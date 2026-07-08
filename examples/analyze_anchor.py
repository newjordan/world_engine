# uv run --dev python examples/analyze_anchor.py bench_out/anchor/anchor.csv --out bench_out/anchor
"""Summarize Phase 10 anchor probe CSVs into a compact review page."""
import argparse
import csv
import html
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(x, nd=2):
    return "nan" if x is None or math.isnan(x) else f"{x:.{nd}f}"


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["idx"] = int(r["idx"])
        r["heading"] = float(r["heading"])
        r["psnr"] = float(r["psnr"])
        r["ssim"] = float(r["ssim"])
        r["projected"] = int(r["projected"])
        r["anchor"] = int(r["anchor"])
        r["anchor_count"] = int(r["anchor_count"])
        r["alpha"] = float(r["alpha"])
        r["dx_px"] = None if r["dx_px"] == "" else float(r["dx_px"])
        r["resp"] = None if r["resp"] == "" else float(r["resp"])
    return rows


def paired(trial, tgt, base):
    keys = sorted(set(trial[tgt]) & set(trial[base]))
    diffs = [trial[tgt][k] - trial[base][k] for k in keys]
    if not diffs:
        return None
    return {"target": tgt, "base": base, "mean": mean(diffs), "std": std(diffs),
            "wins": sum(1 for d in diffs if d > 0), "n": len(diffs)}


def svg_curve(curves, title, ylabel):
    width, height = 900, 320
    left, right, top, bottom = 54, 18, 24, 38
    vals = [v for pts in curves.values() for _x, v in pts if not math.isnan(v)]
    ymax = max(vals + [1.0])
    ymin = min(vals + [0.0])
    pad = max(1.0, (ymax - ymin) * 0.08)
    ymax += pad
    ymin -= pad
    xs = [x for pts in curves.values() for x, _v in pts]
    xmax = max(xs + [1])

    def sx(x):
        return left + (width - left - right) * x / max(xmax, 1)

    def sy(y):
        return height - bottom - (height - top - bottom) * (y - ymin) / max(ymax - ymin, 1e-6)

    palette = ["#2b6cb0", "#c53030", "#2f855a", "#805ad5", "#dd6b20", "#4a5568"]
    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#444"/>',
        f'<text x="{left}" y="16" font-size="12">{html.escape(ylabel)}</text>',
        f'<text x="{width-right-70}" y="{height-8}" font-size="12">frame</text>',
    ]
    legend_x = width - right - 150
    for i, (arm, pts) in enumerate(curves.items()):
        if not pts:
            continue
        color = palette[i % len(palette)]
        ly = top + 16 + 18 * i
        out.append(f'<line x1="{legend_x}" y1="{ly-4}" x2="{legend_x+22}" y2="{ly-4}" stroke="{color}" stroke-width="2.5"/>')
        out.append(f'<text x="{legend_x+28}" y="{ly:.1f}" font-size="12" fill="{color}">{html.escape(arm)}</text>')
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hard-heading", type=float, default=3.2)
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.out) if args.out else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(csv_path)
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    trials = sorted({(r["scene"], r["seed"]) for r in rows})

    back_trial = {a: defaultdict(list) for a in arms}
    post16 = {a: {} for a in arms}
    for r in rows:
        key = (r["scene"], r["seed"])
        if r["phase"] == "back" and r["heading"] <= args.hard_heading:
            back_trial[r["arm"]][key].append(r["psnr"])
        if r["phase"] == "post" and r["idx"] == 16:
            post16[r["arm"]][key] = r["psnr"]
    back_hard = {a: {k: mean(v) for k, v in vals.items()} for a, vals in back_trial.items()}

    paired_back = []
    if "atlas_nowb" in arms and "revisit" in arms:
        p = paired(back_hard, "atlas_nowb", "revisit")
        if p:
            paired_back.append(p)

    paired_post = []
    for a in arms:
        if a in ("revisit", "atlas_nowb"):
            continue
        for base in ("atlas_nowb", "revisit"):
            if base in arms:
                p = paired(post16, a, base)
                if p:
                    paired_post.append(p)

    curves = {}
    for a in arms:
        pts = []
        for idx in sorted({r["idx"] for r in rows if r["phase"] == "post"}):
            vals = [r["psnr"] for r in rows if r["arm"] == a and r["phase"] == "post" and r["idx"] == idx]
            if vals:
                pts.append((idx, mean(vals)))
        curves[a] = pts

    dx_curves = {}
    for a in arms:
        pts = []
        for idx in sorted({r["idx"] for r in rows if r["phase"] == "back"}):
            vals = [abs(r["dx_px"]) for r in rows
                    if r["arm"] == a and r["phase"] == "back" and r["dx_px"] is not None]
            if vals:
                pts.append((idx, mean(vals)))
        dx_curves[a] = pts

    stats = {}
    for a in arms:
        arm_rows = [r for r in rows if r["arm"] == a]
        info_rows = [r for r in arm_rows if r["dx_px"] is not None or r["resp"] is not None]
        acc = [r for r in info_rows if r["projected"]]
        stats[a] = {
            "post16": mean(list(post16.get(a, {}).values())),
            "anchors": mean([r["anchor_count"] for r in arm_rows if r["phase"] == "post" and r["idx"] == 1]),
            "accept": sum(r["projected"] for r in info_rows) / len(info_rows) if info_rows else None,
            "dx": mean([abs(r["dx_px"]) for r in acc if r["dx_px"] is not None]) if acc else None,
            "resp": mean([r["resp"] for r in info_rows if r["resp"] is not None]),
        }

    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_head = "unknown"

    summary = {
        "csv": str(csv_path),
        "git_head": git_head,
        "n_rows": len(rows),
        "n_trials": len(trials),
        "arms": arms,
        "paired_back": paired_back,
        "paired_post16": paired_post,
        "stats": stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def pair_rows(items):
        return "\n".join(
            f"<tr><td>{html.escape(p['target'])} - {html.escape(p['base'])}</td>"
            f"<td>{p['mean']:+.2f} +/- {p['std']:.2f}</td><td>{p['wins']}/{p['n']}</td></tr>"
            for p in items
        )

    stat_cells = []
    for a, s in stats.items():
        accept = "" if s["accept"] is None else f"{100 * s['accept']:.0f}%"
        dx = "" if s["dx"] is None else fmt(s["dx"])
        resp = "" if s["resp"] is None else fmt(s["resp"], 3)
        stat_cells.append(
            f"<tr><td>{html.escape(a)}</td><td>{fmt(s['post16'])}</td>"
            f"<td>{fmt(s['anchors'])}</td><td>{accept}</td>"
            f"<td>{dx}</td><td>{resp}</td></tr>"
        )
    stat_rows = "\n".join(stat_cells)

    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Phase 10 Anchor Probe</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 28px; color: #1f2933; }}
.boundary {{ background: #fff7e6; border: 1px solid #e8c36a; padding: 10px 12px; max-width: 1000px; }}
table {{ border-collapse: collapse; margin-top: 14px; min-width: 760px; }}
th, td {{ border: 1px solid #d7dde5; padding: 7px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #edf2f7; }}
svg {{ max-width: 1100px; width: 100%; height: auto; border: 1px solid #d7dde5; margin-top: 12px; }}
code {{ background: #eef2f7; padding: 1px 4px; }}
</style>
<h1>Phase 10 Anchor Probe</h1>
<p class="boundary">Review artifact generated from <code>{html.escape(str(csv_path))}</code>.
This page is analysis of recorded CSV rows only; it is not evidence unless the linked
CSV/log came from the controlled Phase 10 runner.</p>

<h2>Post-Intervention Curve</h2>
{svg_curve(curves, "Post PSNR vs start", "PSNR vs start")}

<h2>Post Frame 16 Comparisons</h2>
<table><tr><th>comparison</th><th>delta PSNR</th><th>wins</th></tr>
{pair_rows(paired_post)}
</table>

<h2>Canary</h2>
<table><tr><th>comparison</th><th>delta PSNR</th><th>wins</th></tr>
{pair_rows(paired_back)}
</table>

<h2>Diagnostics</h2>
<table><tr><th>arm</th><th>post16</th><th>anchors</th><th>accept</th><th>mean |dx|</th><th>mean resp</th></tr>
{stat_rows}
</table>

<h2>Back-Phase dx</h2>
{svg_curve(dx_curves, "Mean absolute dx", "mean |dx_px|")}
"""
    (out_dir / "index.html").write_text(html_doc)
    print(f"wrote {out_dir / 'index.html'}")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
