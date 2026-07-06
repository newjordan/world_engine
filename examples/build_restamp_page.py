# uv run --dev python examples/build_restamp_page.py --out-dir bench_out --html page.html
"""
Build a self-contained results page for the loop-closure (restamp) permanence result:
- aggregates results.csv into per-(arm,K) means,
- composes the real reference / forgotten / recovered / oracle frames into one strip,
- draws the PSNR-vs-excursion recovery curve as inline SVG,
- fills a hand-designed, theme-aware HTML template (no external assets).

The composed frame strip is embedded as a base64 data URI so the page is fully
self-contained (Artifact CSP blocks any external fetch).
"""
import argparse
import base64
import csv
import math
import os

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
# accent colors (BGR) per arm: still=neutral, revisit=amber(forget), restamp=teal(recover), oracle=green(ceiling)
ACCENT_BGR = {"still": (150, 150, 150), "revisit": (60, 170, 245),
              "restamp": (196, 210, 70), "oracle": (140, 205, 90)}
ACCENT_HEX = {"still": "#8a8f98", "revisit": "#f5aa3c", "restamp": "#46d4c4", "oracle": "#5acd8c"}


def aggregate(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    agg, trials = {}, {}
    for r in rows:
        key = (r["arm"], int(r["K"]))
        trials[(r["scene"], r["seed"], r["arm"], int(r["K"]))] = r
        for m in ("psnr", "ssim"):
            try:
                v = float(r[m])
            except (ValueError, TypeError):
                continue
            if math.isfinite(v):
                agg.setdefault((key, m), []).append(v)
    means = {}
    for (key, m), vals in agg.items():
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
        means[(key, m)] = (mu, sd, len(vals))
    return means, trials


def label_panel(bgr, title, sub, accent):
    h, w = bgr.shape[:2]
    bar = 52
    canvas = np.full((h + bar, w, 3), 22, np.uint8)
    canvas[bar:, :] = bgr
    canvas[:5, :] = accent
    cv2.putText(canvas, title, (14, 26), FONT, 0.62, (238, 238, 238), 2, cv2.LINE_AA)
    cv2.putText(canvas, sub, (14, 45), FONT, 0.5, (176, 176, 176), 1, cv2.LINE_AA)
    return canvas


def compose_strip(png_dir, trials, scene, seed, K):
    def sheet(arm):
        p = os.path.join(png_dir, f"{scene}_{seed}_{arm}_K{K}.png")
        im = cv2.imread(p)
        return im
    rev = sheet("revisit")
    if rev is None:
        raise FileNotFoundError(f"no contact sheets for {scene}_{seed}_K{K}")
    half = rev.shape[1] // 2
    A = rev[:, :half]

    def psnr_of(arm):
        r = trials.get((scene, seed, arm, K))
        return float(r["psnr"]) if r else float("nan")

    panels = [label_panel(A, "REFERENCE", "the view to remember", (150, 150, 150))]
    for arm, title in [("revisit", "REVISIT"), ("restamp", "RESTAMP"), ("oracle", "ORACLE")]:
        s = sheet(arm)
        B = s[:, half:]
        tag = {"revisit": "forgotten", "restamp": "recovered", "oracle": "perfect-memory ceiling"}[arm]
        panels.append(label_panel(B, title, f"{tag} · {psnr_of(arm):.1f} dB", ACCENT_BGR[arm]))

    gap = np.full((panels[0].shape[0], 10, 3), 22, np.uint8)
    strip = panels[0]
    for p in panels[1:]:
        strip = np.concatenate([strip, gap, p], axis=1)
    ok, buf = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def svg_chart(means, Ks, arms, w=720, h=380):
    pad = dict(l=58, r=16, t=18, b=44)
    iw, ih = w - pad["l"] - pad["r"], h - pad["t"] - pad["b"]
    ys = [means[((a, K), "psnr")][0] for a in arms for K in Ks if ((a, K), "psnr") in means]
    ymin, ymax = min(ys) - 2, max(ys) + 2
    xs = [math.log2(K) for K in Ks]
    xmin, xmax = min(xs), max(xs)

    def X(K):
        return pad["l"] + (math.log2(K) - xmin) / (xmax - xmin + 1e-9) * iw

    def Y(v):
        return pad["t"] + (1 - (v - ymin) / (ymax - ymin + 1e-9)) * ih

    el = []
    # y gridlines
    step = 5 if (ymax - ymin) > 20 else 2
    v = math.ceil(ymin / step) * step
    while v <= ymax:
        yy = Y(v)
        el.append(f'<line x1="{pad["l"]}" y1="{yy:.1f}" x2="{w-pad["r"]}" y2="{yy:.1f}" class="grid"/>')
        el.append(f'<text x="{pad["l"]-8}" y="{yy+3:.1f}" class="ytick">{v:.0f}</text>')
        v += step
    # horizon markers
    for hz, lab in [(16, "local 16"), (128, "global 128")]:
        if min(Ks) <= hz <= max(Ks):
            xx = X(hz)
            el.append(f'<line x1="{xx:.1f}" y1="{pad["t"]}" x2="{xx:.1f}" y2="{h-pad["b"]}" class="hz"/>')
            el.append(f'<text x="{xx:.1f}" y="{pad["t"]+10:.1f}" class="hzlab">{lab}</text>')
    # x ticks
    for K in Ks:
        xx = X(K)
        el.append(f'<text x="{xx:.1f}" y="{h-pad["b"]+18:.1f}" class="xtick">{K}</text>')
    el.append(f'<text x="{pad["l"]+iw/2:.1f}" y="{h-6:.1f}" class="axis">excursion length K (latent frames)</text>')
    el.append(f'<text transform="translate(15,{pad["t"]+ih/2:.1f}) rotate(-90)" class="axis">PSNR (dB)</text>')
    # series
    for a in arms:
        pts = [(X(K), Y(means[((a, K), "psnr")][0])) for K in Ks if ((a, K), "psnr") in means]
        if not pts:
            continue
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        el.append(f'<polyline points="{d}" fill="none" stroke="{ACCENT_HEX[a]}" stroke-width="2.5" '
                  f'stroke-linejoin="round"/>')
        for x, y in pts:
            el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{ACCENT_HEX[a]}"/>')
    return f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="PSNR vs excursion length">' \
           + "".join(el) + "</svg>"


def build(out_dir, html_path, scene, seed, K):
    csv_path = os.path.join(out_dir, "results.csv")
    means, trials = aggregate(csv_path)
    Ks = sorted({k for ((_, k), _) in means})
    arms = [a for a in ["still", "revisit", "restamp", "oracle"] if any(((a, k), "psnr") in means for k in Ks)]
    strip_uri = compose_strip(os.path.join(out_dir, "png"), trials, scene, seed, K)
    chart = svg_chart(means, Ks, arms, )

    def cell(a, K, m):
        t = means.get(((a, K), m))
        return f"{t[0]:.2f}" if t else "—"

    # recovery + closure per K
    rows_html = []
    best = (0, None)
    for k in Ks:
        st = means.get((("still", k), "psnr"))
        rev = means.get((("revisit", k), "psnr"))
        res = means.get((("restamp", k), "psnr"))
        orc = means.get((("oracle", k), "psnr"))
        rec = (res[0] - rev[0]) if (res and rev) else None
        clo = (rec / (orc[0] - rev[0]) * 100) if (rec is not None and orc and abs(orc[0]-rev[0]) > 1e-6) else None
        if rec and rec > best[0]:
            best = (rec, k)
        rows_html.append(
            "<tr>"
            f'<td class="k">{k}</td>'
            f"<td>{cell('still',k,'psnr')}</td>"
            f"<td>{cell('revisit',k,'psnr')}</td>"
            f'<td class="hl">{cell("restamp",k,"psnr")}</td>'
            f"<td>{cell('oracle',k,'psnr')}</td>"
            f'<td class="rec">{("%+.2f" % rec) if rec is not None else "—"}</td>'
            f'<td>{("%.0f%%" % clo) if clo is not None else "—"}</td>'
            "</tr>")
    n_scenes = len({s for (s, _, _, _) in trials})
    n_seeds = len({sd for (_, sd, _, _) in trials})
    peak_rec, peak_k = best

    legend = "".join(
        f'<span class="lg"><i style="background:{ACCENT_HEX[a]}"></i>{a}</span>' for a in arms)

    html = TEMPLATE.format(
        strip=strip_uri, chart=chart, rows="".join(rows_html), legend=legend,
        peak_rec=f"{peak_rec:.1f}", peak_k=peak_k, n_scenes=n_scenes, n_seeds=n_seeds,
        vis_case=f"{scene} · seed {seed.replace('seed','')} · K={K}",
    )
    with open(html_path, "w") as f:
        f.write(html)
    print("wrote", html_path, f"({len(html)//1024} KB, strip {len(strip_uri)//1024} KB)")


TEMPLATE = """<title>Waypoint permanence — keyframe re-stamp</title>
<style>
:root{{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#14171c; --mut:#5b6472; --line:#e4e7ec;
  --accent:#0e9c8c; --forget:#d98a2b; --ceil:#2f9d63; --neutral:#8a8f98;
  --shadow:0 1px 2px rgba(20,23,28,.05),0 8px 24px rgba(20,23,28,.06);
}}
@media (prefers-color-scheme:dark){{:root{{
  --bg:#0d1013; --panel:#161a20; --ink:#e9edf2; --mut:#98a2b0; --line:#242a33;
  --accent:#46d4c4; --forget:#f5aa3c; --ceil:#5acd8c; --neutral:#7d858f;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}}}}
:root[data-theme=light]{{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#14171c; --mut:#5b6472; --line:#e4e7ec;
  --accent:#0e9c8c; --forget:#d98a2b; --ceil:#2f9d63; --neutral:#8a8f98;
  --shadow:0 1px 2px rgba(20,23,28,.05),0 8px 24px rgba(20,23,28,.06);
}}
:root[data-theme=dark]{{
  --bg:#0d1013; --panel:#161a20; --ink:#e9edf2; --mut:#98a2b0; --line:#242a33;
  --accent:#46d4c4; --forget:#f5aa3c; --ceil:#5acd8c; --neutral:#7d858f;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  line-height:1.55;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:920px;margin:0 auto;padding:56px 24px 80px}}
.mono{{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}}
.eyebrow{{font-family:ui-monospace,monospace;font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);margin:0 0 14px}}
h1{{font-size:clamp(28px,4.5vw,40px);line-height:1.08;letter-spacing:-.02em;margin:0 0 14px;text-wrap:balance}}
.dek{{font-size:18px;color:var(--mut);margin:0 0 8px;max-width:64ch}}
.dek b{{color:var(--ink);font-weight:600}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:34px 0 6px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 18px 16px;box-shadow:var(--shadow)}}
.card .n{{font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums;font-size:30px;font-weight:600;letter-spacing:-.01em}}
.card .n.acc{{color:var(--accent)}}
.card .l{{font-size:12.5px;color:var(--mut);margin-top:4px}}
section{{margin-top:44px}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:var(--mut);
  font-weight:600;margin:0 0 16px;padding-bottom:10px;border-bottom:1px solid var(--line)}}
figure{{margin:0}}
.frames{{width:100%;border-radius:12px;border:1px solid var(--line);display:block;box-shadow:var(--shadow)}}
figcaption{{font-size:13px;color:var(--mut);margin-top:12px}}
figcaption .mono{{color:var(--ink)}}
.chartwrap{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 10px 6px;box-shadow:var(--shadow);overflow-x:auto}}
.chart{{width:100%;height:auto;min-width:520px;display:block}}
.chart .grid{{stroke:var(--line);stroke-width:1}}
.chart .hz{{stroke:var(--mut);stroke-width:1;stroke-dasharray:3 4;opacity:.6}}
.chart .hzlab,.chart .ytick,.chart .xtick{{fill:var(--mut);font:11px ui-monospace,monospace}}
.chart .ytick,.chart .xtick{{text-anchor:middle}}.chart .ytick{{text-anchor:end}}
.chart .hzlab{{text-anchor:middle}}
.chart .axis{{fill:var(--mut);font:12px ui-sans-serif,system-ui;text-anchor:middle}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:12px 2px 0;font-size:13px;color:var(--mut)}}
.lg{{display:inline-flex;align-items:center;gap:7px}}
.lg i{{width:12px;height:3px;border-radius:2px;display:inline-block}}
table{{width:100%;border-collapse:collapse;font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums;font-size:14px}}
th,td{{text-align:right;padding:9px 10px;border-bottom:1px solid var(--line)}}
th{{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
td.k,th:first-child{{text-align:left}}
td.hl{{color:var(--accent);font-weight:600}}
td.rec{{color:var(--accent)}}
.tblnote{{font-size:12.5px;color:var(--mut);margin-top:10px}}
.prose{{color:var(--mut);max-width:68ch}}
.prose p{{margin:0 0 14px}}.prose b{{color:var(--ink);font-weight:600}}
.prose code{{font-family:ui-monospace,monospace;background:var(--panel);border:1px solid var(--line);
  border-radius:5px;padding:1px 6px;font-size:13px;color:var(--ink)}}
footer{{margin-top:52px;padding-top:20px;border-top:1px solid var(--line);font-size:12.5px;color:var(--mut)}}
footer code{{font-family:ui-monospace,monospace;color:var(--ink)}}
@media(max-width:640px){{.cards{{grid-template-columns:1fr}}}}
</style>
<div class="wrap">
  <p class="eyebrow">Waypoint-1.5-1B · scene permanence · frozen weights</p>
  <h1>Recovering forgotten scenes with an in-distribution keyframe re-stamp</h1>
  <p class="dek">The world model forgets a view once the camera pans away past its KV horizon.
  Re-inserting the old keyframe at an <b>in-distribution recent position</b> — a pure temporal
  rotation of its RoPE, weights untouched — lets the frozen attention read it again and
  <b>pulls the returned view back toward the truth</b>.</p>

  <div class="cards">
    <div class="card"><div class="n acc mono">+{peak_rec} dB</div><div class="l">peak PSNR recovered vs. revisit (at K={peak_k})</div></div>
    <div class="card"><div class="n mono">0</div><div class="l">weights changed — inference-side, flag-gated, default-off</div></div>
    <div class="card"><div class="n mono">{n_scenes}×{n_seeds}</div><div class="l">scenes × seeds per point; oracle = perfect-memory ceiling</div></div>
  </div>

  <section>
    <h2>One trial, four ways to return to a scene</h2>
    <figure>
      <img class="frames" src="{strip}" alt="reference, revisit, restamp, oracle frames"/>
      <figcaption>Same seed and camera path throughout — only the memory differs.
      <span class="mono">Revisit</span> forgets and hallucinates new detail;
      <span class="mono">restamp</span> re-anchors to the pinned keyframe;
      <span class="mono">oracle</span> restores full state (the ceiling).
      <br>Representative case: <span class="mono">{vis_case}</span>.</figcaption>
    </figure>
  </section>

  <section>
    <h2>Recovery vs. excursion length</h2>
    <div class="chartwrap">{chart}</div>
    <div class="legend">{legend}</div>
    <p class="tblnote">Re-stamp lifts the <b style="color:var(--accent)">restamp</b> curve off the
    forgetting floor at short/medium excursions and relaxes back toward it as drift becomes
    irrecoverable past the global horizon.</p>
  </section>

  <section>
    <h2>Numbers (mean PSNR, dB)</h2>
    <table>
      <thead><tr><th>K</th><th>still</th><th>revisit</th><th>restamp</th><th>oracle</th><th>recover</th><th>closure</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <p class="tblnote"><b>still</b> = drift floor (camera never moves) · <b>revisit</b> = pan away &amp; back ·
    <b>restamp</b> = revisit + re-stamped keyframe memory · <b>oracle</b> = full state restore ·
    <b>closure</b> = fraction of the revisit→oracle gap recovered.</p>
  </section>

  <section>
    <h2>How it works · what's next</h2>
    <div class="prose">
      <p><b>Why pinning alone failed:</b> the KV keys are stored after RoPE, so a frame pinned in
      place drifts to an out-of-distribution temporal distance as generation continues and the
      frozen attention can no longer read it.</p>
      <p><b>The fix:</b> <code>restamp_temporal_k</code> composes one extra rotation onto the key's
      temporal RoPE band only — moving it in <i>time</i> to a recent, in-distribution offset while
      keeping its spatial identity. Injected across all layers as a short keyframe burst, the
      frozen model attends to it and partially reconstructs the scene.</p>
      <p><b>Next step — a location matrix:</b> a private, rolling side-process that dead-reckons
      camera pose from the control stream, stores per-location keyframe memory, and on loop closure
      re-stamps and rolls that memory into the frozen graph — light overlay on a weak match,
      context replacement on a strong one, sliding toward the oracle ceiling.</p>
    </div>
  </section>

  <footer>
    Reproduce: <code>examples/permanence_bench.py --arms still,revisit,oracle,restamp --pin-all-layers
    --restamp-offset 2 --pin-frames 4</code>. The frame strip is one real trial; the curve and table
    aggregate all scenes × seeds. Verified by CPU tests in <code>examples/test_kv_restamp.py</code>.
  </footer>
</div>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="bench_out")
    ap.add_argument("--html", default="restamp_page.html")
    ap.add_argument("--scene", default="seed_00")
    ap.add_argument("--seed", default="seed1235")
    ap.add_argument("--K", type=int, default=32)
    a = ap.parse_args()
    build(a.out_dir, a.html, a.scene, a.seed, a.K)
