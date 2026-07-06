# uv run --dev python examples/play_server.py --model Overworld/Waypoint-1.5-1B --port 17860
"""
Interactive browser demo for Waypoint-1.5: WASD + mouse-look drive the world model in
real time. Pure standard-library HTTP server (no web deps):

  GET  /         -> the player page (canvas + input capture)
  GET  /stream   -> multipart/x-mixed-replace MJPEG of generated frames
  POST /ctrl     -> {buttons:[keycodes], mouse:[dx,dy], scroll:n}  (30 Hz from the page)
  POST /reseed   -> jump to the next seed scene
  GET  /stats    -> {fps, gen_ms, frame}

A single worker thread owns the CUDA WorldEngine and generates frames continuously from
the latest control state; the first frame triggers torch.compile (~2-3 min) during which
the page shows the seed image with a "warming up" overlay.
"""
import argparse
import collections
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

from world_engine import WorldEngine, CtrlInput  # installed package (see gen_sample.py)

MOUSE_SCALE = 0.0035     # px of mouse move -> model yaw/pitch velocity
MOUSE_CLAMP = 0.45
JPEG_Q = 80


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.buttons = set()
        self.mouse = [0.0, 0.0]      # accumulated since last gen (consume-and-zero)
        self.scroll = 0
        self.last_ctrl = 0.0
        self.cond = threading.Condition()
        self.jpeg = None
        self.fid = 0
        self.reseed = False
        self.fps = 0.0          # displayed frames / sec (measured by the pacer)
        self.gen_ms = 0.0       # ms per denoise+decode step
        self.ready = False
        # decoupled display pacer: gen thread fills buf with raw frames, pacer thread
        # emits them evenly so the 4-frame bursts play as continuous motion.
        self.buf = collections.deque()
        self.buf_lock = threading.Lock()
        self.frame_dt = 0.033   # EMA target seconds per displayed frame (~ gen_dur/4)


S = Shared()


def encode(rgb_hwc_uint8):
    bgr = cv2.cvtColor(rgb_hwc_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    return buf.tobytes()


def publish(jpeg):
    with S.cond:
        S.jpeg = jpeg
        S.fid += 1
        S.cond.notify_all()


def load_seed(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(cv2.resize(bgr, (1280, 720)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.repeat(rgb[None], 4, 0))


def gen_worker(model_uri, device, seeds, quant):
    try:
        engine = WorldEngine(model_uri, quant=quant, device=device)
        print(f"engine ready (quant={quant})", flush=True)
    except Exception as e:                          # quant backend missing/unsupported -> bf16
        print(f"quant={quant} failed ({e}); falling back to bf16", flush=True)
        engine = WorldEngine(model_uri, quant=None, device=device)
    si = 0
    torch.manual_seed(1234)
    engine.reset()
    seed_x4 = load_seed(seeds[si])
    engine.append_frame(seed_x4.to(engine.device))
    publish(encode(seed_x4[-1].numpy()))     # show seed while the first gen compiles

    while True:
        if S.reseed:
            si = (si + 1) % len(seeds)
            torch.manual_seed(1234 + si)
            engine.reset()
            seed_x4 = load_seed(seeds[si])
            engine.append_frame(seed_x4.to(engine.device))
            with S.buf_lock:
                S.buf.clear()
            with S.lock:
                S.reseed = False
                S.mouse = [0.0, 0.0]
            publish(encode(seed_x4[-1].numpy()))

        with S.lock:
            if time.time() - S.last_ctrl > 0.2:
                S.mouse = [0.0, 0.0]                     # release look when page goes quiet
            btn = {int(b) for b in S.buttons if 0 <= int(b) < 256}
            mx = max(-MOUSE_CLAMP, min(MOUSE_CLAMP, S.mouse[0]))
            my = max(-MOUSE_CLAMP, min(MOUSE_CLAMP, S.mouse[1]))
            scroll = S.scroll
            S.mouse = [0.0, 0.0]
            S.scroll = 0

        t0 = time.time()
        four = engine.gen_frame(ctrl=CtrlInput(button=btn, mouse=(mx, my), scroll_wheel=scroll))
        frames = four.detach().cpu().numpy()          # (4, H, W, 3) — the native 60fps burst
        gd = time.time() - t0
        with S.buf_lock:
            S.buf.extend(frames)                      # pacer emits these evenly
            while len(S.buf) > 12:                    # bound end-to-end latency
                S.buf.popleft()
        with S.lock:
            S.gen_ms = gd * 1000.0
            S.frame_dt = 0.85 * S.frame_dt + 0.15 * (gd / 4.0)
            S.ready = True


def pacer_worker():
    """Emit buffered frames at an even cadence (~gen_dur/4) so motion is continuous
    instead of 4-frame bursts. Encodes off the gen thread; catches up if it falls behind."""
    pub_t, pub_n = time.time(), 0
    while True:
        with S.buf_lock:
            fr = S.buf.popleft() if S.buf else None
            backlog = len(S.buf)
        if fr is None:
            time.sleep(0.004)
            continue
        publish(encode(fr))
        pub_n += 1
        now = time.time()
        if now - pub_t >= 0.5:
            with S.lock:
                S.fps = pub_n / (now - pub_t)
            pub_t, pub_n = now, 0
        dt = S.frame_dt
        if backlog > 6:                               # behind -> speed up, keep latency low
            dt *= 0.4
        elif backlog < 2:                             # starving -> ease off to avoid stutter
            dt *= 1.2
        time.sleep(max(0.006, min(dt, 0.05)))


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Waypoint-1.5 — play</title>
<style>
  html,body{margin:0;height:100%;background:#07090c;color:#cdd6e0;
    font:14px/1.5 ui-monospace,Menlo,Consolas,monospace;overflow:hidden}
  #stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center}
  #view{max-width:100%;max-height:100%;image-rendering:auto;background:#000;
    box-shadow:0 0 0 1px #1c2330, 0 20px 60px rgba(0,0,0,.6);cursor:crosshair}
  #hud{position:fixed;top:12px;left:12px;background:rgba(9,12,16,.7);border:1px solid #1c2330;
    border-radius:8px;padding:8px 12px;backdrop-filter:blur(6px);pointer-events:none}
  #hud b{color:#46d4c4}
  #help{position:fixed;bottom:12px;left:12px;right:12px;text-align:center;color:#7d858f;
    background:rgba(9,12,16,.6);border-radius:8px;padding:7px;pointer-events:none}
  kbd{background:#161c26;border:1px solid #2a3340;border-bottom-width:2px;border-radius:4px;
    padding:1px 6px;color:#cdd6e0;margin:0 1px}
  #over{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
    flex-direction:column;gap:14px;background:rgba(7,9,12,.72);backdrop-filter:blur(3px);
    font-size:18px;color:#e9edf2;z-index:5}
  #over.hidden{display:none}
  .dot{width:9px;height:9px;border-radius:50%;background:#46d4c4;display:inline-block;
    animation:p 1s infinite}
  @keyframes p{0%,100%{opacity:.25}50%{opacity:1}}
  #reseed{position:fixed;top:12px;right:12px;background:#161c26;color:#cdd6e0;
    border:1px solid #2a3340;border-radius:8px;padding:8px 14px;cursor:pointer;font:inherit;z-index:6}
  #reseed:hover{border-color:#46d4c4;color:#46d4c4}
</style></head><body>
<div id=stage><img id=view src="/stream"></div>
<div id=hud>Waypoint-1.5 · <span id=st>connecting…</span></div>
<button id=reseed>new scene ⟳</button>
<div id=help><kbd>click</kbd> to capture mouse · <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move ·
  <kbd>Space</kbd> / <kbd>Shift</kbd> · <kbd>mouse</kbd> look · <kbd>scroll</kbd> · <kbd>Esc</kbd> release</div>
<div id=over><div><span class=dot></span> warming up — compiling the model on first frame…</div>
  <div style="color:#7d858f;font-size:14px">this takes a couple minutes once, then it's real-time</div></div>
<script>
const KEY={KeyW:87,KeyA:65,KeyS:83,KeyD:68,Space:32,ShiftLeft:16,ShiftRight:16,
  KeyE:69,KeyQ:81,ArrowUp:38,ArrowDown:40,ArrowLeft:37,ArrowRight:39,Digit1:49,KeyF:70};
const pressed=new Set(); let mdx=0,mdy=0,scroll=0,locked=false;
const view=document.getElementById('view');
view.addEventListener('click',()=>view.requestPointerLock());
document.addEventListener('pointerlockchange',()=>{locked=document.pointerLockElement===view;});
addEventListener('keydown',e=>{if(KEY[e.code]!==undefined){pressed.add(KEY[e.code]);e.preventDefault();}});
addEventListener('keyup',e=>{if(KEY[e.code]!==undefined){pressed.delete(KEY[e.code]);e.preventDefault();}});
addEventListener('mousemove',e=>{if(locked){mdx+=e.movementX;mdy+=e.movementY;}});
addEventListener('wheel',e=>{scroll+=Math.sign(e.deltaY);},{passive:true});
document.getElementById('reseed').addEventListener('click',()=>{
  fetch('/reseed',{method:'POST'});pressed.clear();});
setInterval(async()=>{
  const body=JSON.stringify({buttons:[...pressed],mouse:[mdx,mdy],scroll:scroll});
  mdx=0;mdy=0;scroll=0;
  try{await fetch('/ctrl',{method:'POST',body});}catch(e){}
},33);
setInterval(async()=>{
  try{const s=await(await fetch('/stats')).json();
    document.getElementById('st').innerHTML=
      `<b>${s.fps.toFixed(1)}</b> fps · ${s.gen_ms.toFixed(0)} ms/frame · frame ${s.frame}`;
    document.getElementById('over').classList.toggle('hidden', s.frame>1);
  }catch(e){}
},500);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif self.path == "/stats":
            with S.lock:
                body = json.dumps({"fps": S.fps, "gen_ms": S.gen_ms, "frame": S.fid}).encode()
            self._send(200, "application/json", body)
        elif self.path == "/stream":
            self._stream()
        else:
            self._send(404, "text/plain", b"not found")

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        while True:
            with S.cond:
                while S.fid == last:
                    S.cond.wait(timeout=2.0)
                jpg = S.jpeg
                last = S.fid
            if jpg is None:
                continue
            try:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        if self.path == "/reseed":
            with S.lock:
                S.reseed = True
            self._send(200, "application/json", b'{"ok":true}')
            return
        if self.path == "/ctrl":
            try:
                d = json.loads(raw or b"{}")
                with S.lock:
                    S.buttons = set(d.get("buttons", []))
                    S.mouse[0] += float(d.get("mouse", [0, 0])[0]) * MOUSE_SCALE
                    S.mouse[1] += float(d.get("mouse", [0, 0])[1]) * MOUSE_SCALE
                    S.scroll += int(d.get("scroll", 0))
                    S.last_ctrl = time.time()
            except Exception:
                pass
            self._send(200, "application/json", b'{"ok":true}')
            return
        self._send(404, "text/plain", b"not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=17860)
    ap.add_argument("--assets", default="bench_assets")
    ap.add_argument("--quant", default="intw8a8", help="intw8a8 | fp8w8a8 | nvfp4 | none")
    a = ap.parse_args()

    quant = None if a.quant.lower() in ("none", "") else a.quant
    seeds = sorted(glob.glob(os.path.join(a.assets, "*.png")))
    if not seeds:
        print(f"no seed pngs in {a.assets}", file=sys.stderr)
        sys.exit(1)

    threading.Thread(target=gen_worker, args=(a.model, a.device, seeds, quant), daemon=True).start()
    threading.Thread(target=pacer_worker, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"serving on http://127.0.0.1:{a.port}  ({len(seeds)} seeds)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
