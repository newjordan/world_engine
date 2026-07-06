# uv run --dev python examples/permanence_bench.py --self-test           # CPU, no model
# uv run --dev python examples/permanence_bench.py --model Overworld/Waypoint-1.5-1B \
#     --arms still,revisit,oracle --K 8,16,32,64,128,256 --seeds 5 --out bench_out/
"""
Revisit-consistency benchmark for object/scene permanence in Waypoint-1.5.

Measures how much the world model "forgets" a scene once the camera pans away and
comes back, as a function of excursion length K (in latent frames). See
docs/PERMANENCE_PLAN.md (Phase 2) and docs/PERMANENCE_RESULTS.md.

Three arms per (scene, seed, K):
  still    settle -> K no-op frames                 -> autoregressive drift floor
  revisit  settle -> pan-away K/2 -> pan-back K/2    -> drift + forgetting
  oracle   settle -> [get_state, run revisit, load_state] -> 1 frame  -> sampling floor

Headline result: the gap (revisit - still) as K crosses the local (16) and global
(128) horizons is the measured forgetting curve.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import urllib.request

import cv2
import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Metrics (numpy/cv2 only -- no new deps)
# --------------------------------------------------------------------------- #
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two uint8 HxWx3 images. inf if identical."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def _to_gray(x: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float64)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Single-scale SSIM on luma with an 11x11 Gaussian window (sigma 1.5)."""
    a = _to_gray(a)
    b = _to_gray(b)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    k = (11, 11)
    mu_a = cv2.GaussianBlur(a, k, 1.5)
    mu_b = cv2.GaussianBlur(b, k, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a2
    sb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_ab
    ssim_map = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return float(ssim_map.mean())


def make_lpips(device):
    """Return an LPIPS(a,b)->float callable, or None if the package is missing."""
    try:
        import lpips as _lpips  # noqa
    except Exception:
        return None
    net = _lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def _fn(a: np.ndarray, b: np.ndarray) -> float:
        def prep(x):
            t = torch.from_numpy(x).to(device).float().permute(2, 0, 1)[None] / 127.5 - 1.0
            return t
        return float(net(prep(a), prep(b)).item())

    return _fn


# --------------------------------------------------------------------------- #
# Trajectories: build a per-step list of control specs (dicts).
#   Fresh CtrlInput is constructed per step in the run loop, because
#   engine.prep_inputs mutates CtrlInput fields in place (world_engine.py:172).
# --------------------------------------------------------------------------- #
NOOP = {"button": set(), "mouse": (0.0, 0.0)}


def _noop(n):
    return [dict(button=set(), mouse=(0.0, 0.0)) for _ in range(n)]


def build_controls(arm: str, K: int, trajectory: str, yaw_mag: float = 0.2):
    """Controls executed AFTER the settle phase (which is handled separately)."""
    assert K % 2 == 0, "K must be even (pan-away K/2 + pan-back K/2)"
    if arm == "still":
        return _noop(K)

    half = K // 2
    if trajectory == "yaw":
        away = [dict(button=set(), mouse=(+yaw_mag, 0.0)) for _ in range(half)]
        back = [dict(button=set(), mouse=(-yaw_mag, 0.0)) for _ in range(half)]
    elif trajectory == "strafe":  # fallback: D (68) then A (65)
        away = [dict(button={68}, mouse=(0.0, 0.0)) for _ in range(half)]
        back = [dict(button={65}, mouse=(0.0, 0.0)) for _ in range(half)]
    else:
        raise ValueError(f"unknown trajectory {trajectory!r}")
    # revisit and oracle execute the identical excursion; the arms differ only in
    # how the scored frame is produced (see run_trial).
    return away + back


# --------------------------------------------------------------------------- #
# Engine wrappers
# --------------------------------------------------------------------------- #
class RealEngine:
    """Thin adapter over world_engine.WorldEngine."""

    def __init__(self, model_uri, device, quant, pin_frames, pin_all_layers=False):
        from world_engine import WorldEngine, CtrlInput  # local import: touches CUDA
        self._CtrlInput = CtrlInput
        overrides = None
        if pin_frames:
            overrides = {"n_pin_frames": pin_frames, "pin_all_layers": pin_all_layers}
        self.engine = WorldEngine(model_uri, quant=quant, device=device,
                                  model_config_overrides=overrides)
        self.device = self.engine.device

    def reset(self):
        self.engine.reset()

    def append_seed(self, seed_x4_uint8):
        self.engine.append_frame(seed_x4_uint8.to(self.device))

    def gen(self, spec):
        ctrl = self._CtrlInput(button=set(spec["button"]), mouse=list(spec["mouse"]))
        four = self.engine.gen_frame(ctrl=ctrl)  # (4, H, W, 3) uint8 on device
        return four

    def last_rgb(self, four):
        return four[-1].detach().cpu().numpy()

    def all_rgb(self, four):
        return four.detach().cpu().numpy()  # (4, H, W, 3)

    def get_state(self):
        return self.engine.get_state()

    def load_state(self, s):
        self.engine.load_state(s)

    def pin_frame(self):
        self.engine.pin_frame()

    def restamp_memory(self, offset, align_pose=False):
        if hasattr(self.engine, 'restamp_memory'):
            self.engine.restamp_memory(offset, align_pose=align_pose)
        else:
            self.engine.restamp_memory(offset)

    # --- pose-indexed world atlas (Phase 6) -------------------------------------
    def attach_atlas(self, store):
        self._atlas = store

    def atlas_capture(self):
        from atlas import atlas_capture as _cap
        return _cap(self.engine, self._atlas)

    def atlas_activate(self, offset=8, k=1, align_pose=True, retrieve="nearest"):
        from atlas import atlas_activate as _act
        return _act(self.engine, self._atlas, offset=offset, k=k,
                    align_pose=align_pose, retrieve=retrieve)


class FakeEngine:
    """
    CPU synthetic world for --self-test. Integrates controls into a camera pose and
    renders a deterministic frame from that pose, injecting "forgetting" error that
    grows once the excursion exceeds a synthetic horizon. Lets us validate the entire
    harness + analysis pipeline (metrics, symmetry, CSV, plots) without a GPU.
    """
    H, W = 90, 160  # small, for speed

    def __init__(self, horizon=16, forget=0.9, drift=0.02, seed=0):
        self.horizon = horizon
        self.forget = forget
        self.drift = drift
        self.device = torch.device("cpu")
        self._rng = np.random.default_rng(seed)
        self._base = None
        self.reset()
    def reset(self):
        self.yaw = 0.0
        self.pos = np.zeros(2)
        self.t = 0
        self.max_excursion = 0.0

    def append_seed(self, seed_x4_uint8):
        # Use the seed image as the scene's base appearance.
        img = seed_x4_uint8[-1].cpu().numpy()
        self._base = cv2.resize(img, (self.W, self.H)).astype(np.float64)

    def _render(self):
        # View = base shifted by yaw/pos, plus accumulated drift + forgetting error.
        shift = int(round(self.yaw * 40 + self.pos[0] * 40))
        view = np.roll(self._base, shift, axis=1)
        drift_sigma = self.drift * 255 * math.sqrt(max(self.t, 1))
        err = 0.0
        if self.max_excursion > self.horizon:
            over = (self.max_excursion - self.horizon) / self.horizon
            err = self.forget * 255 * min(over, 1.0)
        # Atlas relief: a retrieved near-heading keyframe suppresses forgetting error for
        # this frame (synthetic analogue of restamping an in-distribution memory). Set by
        # atlas_activate(), consumed once, reset in gen().
        err *= (1.0 - getattr(self, "_mem_relief", 0.0))
        noise = self._rng.normal(0, drift_sigma + err, view.shape)
        return np.clip(view + noise, 0, 255).astype(np.uint8)

    def gen(self, spec):
        self.yaw += spec["mouse"][0]
        if 68 in spec["button"]:
            self.pos[0] += 0.2
        if 65 in spec["button"]:
            self.pos[0] -= 0.2
        self.t += 1
        # excursion measured in FRAMES away from origin (peak), so `horizon` is in the
        # same latent-frame units as K -> synthetic forgetting appears once K/2 > horizon.
        steps_away = (abs(self.yaw) + abs(self.pos[0])) / 0.2
        self.max_excursion = max(self.max_excursion, steps_away)
        frame = self._render()
        self._mem_relief = 0.0  # relief applies to exactly the frame after activate
        return np.stack([frame] * 4, 0)  # mimic (4,H,W,3)

    def last_rgb(self, four):
        return four[-1]

    def all_rgb(self, four):
        return four

    def get_state(self):
        return dict(yaw=self.yaw, pos=self.pos.copy(), t=self.t, exc=self.max_excursion)

    def load_state(self, s):
        self.yaw, self.pos, self.t, self.max_excursion = s["yaw"], s["pos"].copy(), s["t"], s["exc"]

    def pin_frame(self):
        pass

    def restamp_memory(self, offset, align_pose=False):
        pass  # synthetic world has no KV to re-stamp; restamp arm ~= revisit here

    # --- pose-indexed world atlas (Phase 6): simulate memory relief ------------
    def attach_atlas(self, store):
        self._atlas = store
        self._mem_relief = 0.0

    def atlas_capture(self):
        # payload is irrelevant in the synthetic world; only the pose tag matters.
        return self._atlas.insert(kv=None, yaw=self.yaw, frame_ts=self.t)

    def atlas_activate(self, offset=8, k=1, align_pose=True, retrieve="nearest",
                       relief_scale=0.8):
        hits = self._atlas.query(self.yaw, k=k, mode=retrieve)
        if not hits:
            self._mem_relief = 0.0
            return 0
        # A near-heading keyframe gives strong relief; a far one gives none. relief_scale
        # is in the same yaw-accumulator units as the capture spacing.
        resid = abs(self.yaw - hits[0].yaw)
        self._mem_relief = max(0.0, 1.0 - resid / relief_scale)
        return len(hits)


# --------------------------------------------------------------------------- #
# Seed images
# --------------------------------------------------------------------------- #
def load_seeds(n, cache_dir, self_test=False):
    """Return a list of (name, uint8 (4,720,1280,3) tensor). Caches under cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    if self_test:
        seeds = []
        for i in range(n):
            rng = np.random.default_rng(1000 + i)
            base = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
            base = cv2.resize(base, (1280, 720), interpolation=cv2.INTER_NEAREST)
            seeds.append((f"synthetic_{i}", torch.from_numpy(np.repeat(base[None], 4, 0))))
        return seeds

    cached = sorted(f for f in os.listdir(cache_dir) if f.endswith(".png"))
    if len(cached) < n:
        api = "https://api.github.com/repos/Overworldai/Biome/contents/seeds?ref=14343a6"
        with urllib.request.urlopen(api) as res:
            urls = [it["download_url"] for it in json.load(res) if it["type"] == "file"]
        random.Random(0).shuffle(urls)
        for i, url in enumerate(urls[:n]):
            raw = urllib.request.urlopen(url).read()
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(cv2.resize(img, (1280, 720)), cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(cache_dir, f"seed_{i:02d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cached = sorted(f for f in os.listdir(cache_dir) if f.endswith(".png"))

    seeds = []
    for f in cached[:n]:
        bgr = cv2.imread(os.path.join(cache_dir, f))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        seeds.append((os.path.splitext(f)[0], torch.from_numpy(np.repeat(rgb[None], 4, 0))))
    return seeds


# --------------------------------------------------------------------------- #
# One trial
# --------------------------------------------------------------------------- #
def run_trial(eng, seed_x4, seed, arm, K, settle, trajectory, pin_at_reference=False,
              restamp_offset=8, n_capture=1):
    torch.manual_seed(seed)
    eng.reset()
    eng.append_seed(seed_x4)

    # Settle: `settle` no-op frames. Reference view A = last RGB of the final 4-pack.
    # Capture the last `n_capture` settle frames into the KV pin slots (round-robin), so
    # the restamp arm can re-inject a multi-frame memory of the reference scene.
    capture = pin_at_reference or arm in ("restamp", "restamp_pose")
    A = None
    specs = _noop(settle)
    for i, spec in enumerate(specs):
        A = eng.last_rgb(eng.gen(spec))
        if capture and i >= settle - n_capture:
            eng.pin_frame()

    if arm == "oracle":
        state = eng.get_state()
        for spec in build_controls("revisit", K, trajectory):
            eng.gen(spec)
        eng.load_state(state)
        B = eng.last_rgb(eng.gen(NOOP))
    elif arm == "restamp":
        # Loop closure: identical revisit excursion, but during the pan-BACK phase the
        # pinned keyframe is re-stamped to an in-distribution recent offset each step so
        # the frozen attention can actually read it. Tests whether that closes the gap.
        controls = build_controls("revisit", K, trajectory)
        half = K // 2
        B = None
        for i, spec in enumerate(controls):
            if i >= half:  # pan-back: bring the memory in-distribution before generating
                eng.restamp_memory(restamp_offset)
            B = eng.last_rgb(eng.gen(spec))
    elif arm == "restamp_pose":
        # Full pose restamp: same as restamp but also rotates the spatial (x/y) RoPE
        # phase of the pinned key by the dead-reckoned camera yaw delta. This is the
        # decisive experiment: does aligning the memory's spatial phase to the current
        # camera heading close more of the revisit→oracle gap than time-only restamp?
        controls = build_controls("revisit", K, trajectory)
        half = K // 2
        B = None
        for i, spec in enumerate(controls):
            if i >= half:  # pan-back: align memory pose (x,y,t) before generating
                eng.restamp_memory(restamp_offset, align_pose=True)
            B = eng.last_rgb(eng.gen(spec))
    else:  # still / revisit
        B = None
        for spec in build_controls(arm, K, trajectory):
            B = eng.last_rgb(eng.gen(spec))

    return A, B


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def contact_sheet(A, B, path, scale=0.5):
    def bgr(x):
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    sheet = np.concatenate([bgr(A), bgr(B)], axis=1)
    if scale != 1.0:
        sheet = cv2.resize(sheet, None, fx=scale, fy=scale)
    cv2.imwrite(path, sheet)


def sweep(args):
    device = "cpu" if args.self_test else args.device
    arms = args.arms.split(",")
    # Pins are needed to capture the reference keyframe for the restamp arm (or when
    # explicitly pinning at reference). Allocated on global layers only by default.
    need_pins = args.pin_at_reference or any(a in arms for a in ("restamp", "restamp_pose"))
    if args.self_test:
        eng = FakeEngine(horizon=16)
    else:
        eng = RealEngine(args.model, device, args.quant,
                         args.pin_frames if need_pins else 0,
                         pin_all_layers=args.pin_all_layers)
    lpips_fn = None if args.self_test or args.no_lpips else make_lpips(eng.device)

    os.makedirs(args.out, exist_ok=True)
    png_dir = os.path.join(args.out, "png")
    os.makedirs(png_dir, exist_ok=True)
    csv_path = os.path.join(args.out, "results.csv")
    new = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="")
    writer = csv.writer(fcsv)
    if new:
        writer.writerow(["scene", "seed", "arm", "K", "psnr", "ssim", "lpips", "png_path"])
        fcsv.flush()

    seeds = load_seeds(args.n_scenes, args.assets, self_test=args.self_test)
    Ks = [int(k) for k in args.K.split(",")]

    n_total = len(seeds) * args.seeds * len(Ks) * len(arms)
    done = 0
    for scene_name, seed_x4 in seeds:
        for s in range(args.seeds):
            seed = args.seed_base + s
            for K in Ks:
                for arm in arms:
                    A, B = run_trial(eng, seed_x4, seed, arm, K, args.settle,
                                     args.trajectory, args.pin_at_reference,
                                     restamp_offset=args.restamp_offset,
                                     n_capture=min(args.pin_frames, args.settle))
                    p, sm = psnr(A, B), ssim(A, B)
                    lp = lpips_fn(A, B) if lpips_fn else ""
                    tag = f"{scene_name}_seed{seed}_{arm}_K{K}"
                    if args.pin_at_reference:
                        tag += "_pinned"
                    png = os.path.join(png_dir, tag + ".png")
                    contact_sheet(A, B, png)
                    writer.writerow([scene_name, seed, arm, K, f"{p:.4f}", f"{sm:.4f}",
                                     (f"{lp:.4f}" if lp != "" else ""), png])
                    fcsv.flush()
                    done += 1
                    print(f"[{done}/{n_total}] {tag}  psnr={p:.2f} ssim={sm:.3f}"
                          + (f" lpips={lp:.3f}" if lp != "" else ""), flush=True)
    fcsv.close()
    print(f"\nWrote {csv_path}")
    return csv_path


# --------------------------------------------------------------------------- #
# Pilot (Phase 2a): validate that the excursion trajectory returns the camera.
# --------------------------------------------------------------------------- #
def pilot(args):
    """Settle -> pan away K/2 -> pan back K/2 for one scene, writing the full video
    plus an A|away|B contact sheet. The trajectory is validated iff panning changes
    the view (PSNR(A, away) low) AND returning restores it (PSNR(A, B) high)."""
    import imageio.v3 as iio

    K = int(args.K.split(",")[0])
    eng = RealEngine(args.model, args.device, args.quant, 0)
    seeds = load_seeds(1, args.assets, self_test=False)
    scene_name, seed_x4 = seeds[0]
    seed = args.seed_base

    torch.manual_seed(seed)
    eng.reset()
    eng.append_seed(seed_x4)

    video = [np.repeat(seed_x4[-1:].cpu().numpy(), 4, 0)]  # seed as first frames
    A = None
    for spec in _noop(args.settle):
        four = eng.all_rgb(eng.gen(spec))
        video.append(four)
        A = four[-1]

    away = None
    excursion = build_controls("revisit", K, args.trajectory)
    for i, spec in enumerate(excursion):
        four = eng.all_rgb(eng.gen(spec))
        video.append(four)
        if i == K // 2 - 1:
            away = four[-1]        # furthest point of the excursion
    B = video[-1][-1]              # returned view

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"pilot_{scene_name}_{args.trajectory}_K{K}")
    frames = np.concatenate(video, axis=0)
    iio.imwrite(stem + ".mp4", frames, fps=60, codec="libx264")
    sheet = np.concatenate([cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in (A, away, B)], axis=1)
    cv2.imwrite(stem + "_A_away_B.png", cv2.resize(sheet, None, fx=0.5, fy=0.5))

    p_away, s_away = psnr(A, away), ssim(A, away)
    p_back, s_back = psnr(A, B), ssim(A, B)
    print(f"\n=== PILOT {scene_name} trajectory={args.trajectory} K={K} ===")
    print(f"  A vs away (furthest):  PSNR {p_away:6.2f}  SSIM {s_away:.3f}   (want LOW: view moved)")
    print(f"  A vs B    (returned):  PSNR {p_back:6.2f}  SSIM {s_back:.3f}   (want HIGH: view returned)")
    moved = p_away < p_back - 1.0
    returned = p_back > p_away + 2.0
    verdict = "PASS" if (moved and returned) else "CHECK"
    print(f"  moved={moved} returned={returned}  ->  {verdict}")
    print(f"  video:  {stem}.mp4")
    print(f"  frames: {stem}_A_away_B.png")
    return stem


# --------------------------------------------------------------------------- #
# Analysis (Phase 2d)
# --------------------------------------------------------------------------- #
def analyze(out_dir):
    csv_path = os.path.join(out_dir, "results.csv")
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print("no rows")
        return

    def fnum(x):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except (ValueError, TypeError):
            return None

    arms = sorted({r["arm"] for r in rows})
    Ks = sorted({int(r["K"]) for r in rows})
    agg = {}  # (arm, K) -> {metric: (mean, std, n)}
    for arm in arms:
        for K in Ks:
            sel = [r for r in rows if r["arm"] == arm and int(r["K"]) == K]
            for m in ("psnr", "ssim"):
                vals = [fnum(r[m]) for r in sel]
                vals = [v for v in vals if v is not None]
                if vals:
                    mean = sum(vals) / len(vals)
                    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                    agg[(arm, K, m)] = (mean, std, len(vals))

    lines = ["# Forgetting curve (mean ± std)\n"]
    for m in ("psnr", "ssim"):
        lines.append(f"\n## {m.upper()}\n")
        header = "| K | " + " | ".join(arms) + " | revisit−still |"
        lines.append(header)
        lines.append("|" + "---|" * (len(arms) + 2))
        for K in Ks:
            cells = []
            for arm in arms:
                v = agg.get((arm, K, m))
                cells.append(f"{v[0]:.3f}±{v[1]:.3f}" if v else "—")
            rv = agg.get(("revisit", K, m))
            st = agg.get(("still", K, m))
            gap = f"{rv[0] - st[0]:+.3f}" if (rv and st) else "—"
            lines.append(f"| {K} | " + " | ".join(cells) + f" | {gap} |")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "curve.md"), "w") as f:
        f.write(report + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, m in zip(axes, ("psnr", "ssim")):
            for arm in arms:
                xs = [K for K in Ks if (arm, K, m) in agg]
                ys = [agg[(arm, K, m)][0] for K in xs]
                es = [agg[(arm, K, m)][1] for K in xs]
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=arm)
            ax.axvline(16, ls="--", c="gray", alpha=0.6, label="local horizon")
            ax.axvline(128, ls=":", c="gray", alpha=0.6, label="global horizon")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("K (latent frames)")
            ax.set_ylabel(m.upper())
            ax.set_title(f"{m.upper()} vs excursion length")
            ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path = os.path.join(out_dir, "forgetting_curve.png")
        fig.savefig(plot_path, dpi=120)
        print(f"\nWrote {plot_path}")
    except Exception as e:
        print(f"(matplotlib unavailable, skipped plot: {e})")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--arms", default="still,revisit,oracle")
    ap.add_argument("--K", default="8,16,32,64,128,256", help="excursion lengths (latent frames), comma-sep")
    ap.add_argument("--seeds", type=int, default=5, help="seeds per (scene,arm,K)")
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--n-scenes", type=int, default=3)
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--trajectory", choices=["yaw", "strafe"], default="yaw")
    ap.add_argument("--out", default="bench_out")
    ap.add_argument("--assets", default="bench_assets")
    ap.add_argument("--no-lpips", action="store_true")
    # Phase 3
    ap.add_argument("--pin-at-reference", action="store_true", help="pin the reference frame (Phase 3)")
    ap.add_argument("--pin-frames", type=int, default=4, help="n_pin_frames when pinning")
    ap.add_argument("--pin-all-layers", action="store_true",
                    help="pin into local layers too (default: global layers only)")
    ap.add_argument("--restamp-offset", type=int, default=8,
                    help="restamp arm: in-distribution temporal offset (frames) to place "
                         "the pinned keyframe behind the current query (default: 8 = one "
                         "global dilation bucket)")
    # modes
    ap.add_argument("--self-test", action="store_true", help="CPU synthetic dry-run (no model)")
    ap.add_argument("--pilot", action="store_true", help="Phase 2a: validate camera-return trajectory")
    ap.add_argument("--analyze", metavar="OUT_DIR", help="only aggregate an existing OUT_DIR/results.csv")
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return
    if args.pilot:
        pilot(args)
        return
    out = sweep(args)
    analyze(os.path.dirname(out) or ".")


if __name__ == "__main__":
    main()
