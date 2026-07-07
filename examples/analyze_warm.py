# uv run --dev python examples/analyze_warm.py bench_out/warm/warm.csv --out bench_out/warm
"""
Summarize Phase 9 warmstart probe CSVs into a compact review page.

The probe itself owns the controlled run. This script only reads its CSV/log outputs
and creates static review artifacts for inspection.
"""
import argparse
import csv
import html
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


PAIRS = [
    ("atlas", "revisit"),
    ("atlas_nowb", "revisit"),
    ("redream", "revisit"),
    ("redream_nowb", "revisit"),
    ("redream_far", "revisit"),
    ("atlas_redream", "revisit"),
    ("redream", "redream_nowb"),
    ("redream", "redream_far"),
    ("atlas_nowb", "atlas"),
    ("atlas_redream", "atlas"),
    ("atlas_redream", "redream"),
]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(x, nd=2):
    if x is None or math.isnan(x):
        return "nan"
    return f"{x:.{nd}f}"


def pct(x):
    return "" if x is None else f"{100 * x:.0f}%"


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["back_idx"] = int(r["back_idx"])
        r["heading"] = float(r["heading"])
        r["psnr"] = float(r["psnr"])
        r["ssim"] = float(r["ssim"])
        r["projected"] = int(r["projected"])
        r["wb"] = int(r["wb"])
        r["dx_px"] = None if r["dx_px"] == "" else float(r["dx_px"])
        r["resp"] = None if r["resp"] == "" else float(r["resp"])
        r["sigma_re"] = None if r["sigma_re"] == "" else float(r["sigma_re"])
    return rows


def paired_stats(trial, tgt, base):
    keys = sorted(set(trial[tgt]) & set(trial[base]))
    diffs = [trial[tgt][k] - trial[base][k] for k in keys]
    if not diffs:
        return None
    return {
        "target": tgt,
        "base": base,
        "mean": mean(diffs),
        "std": std(diffs),
        "wins": sum(1 for d in diffs if d > 0),
        "n": len(diffs),
    }


def svg_dx(curves):
    width, height = 880, 300
    left, right, top, bottom = 54, 18, 20, 38
    vals = [v for pts in curves.values() for _x, v in pts if v is not None and not math.isnan(v)]
    ymax = max(vals + [1.0])
    ymax = math.ceil(ymax / 25.0) * 25.0
    xmax = max([x for pts in curves.values() for x, _v in pts] + [31])

    def sx(x):
        return left + (width - left - right) * x / max(xmax, 1)

    def sy(y):
        return height - bottom - (height - top - bottom) * y / ymax

    colors = {
        "atlas_nowb": "#2b6cb0",
        "redream": "#c53030",
        "redream_nowb": "#2f855a",
        "redream_far": "#805ad5",
        "atlas_redream": "#dd6b20",
    }
    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Mean absolute dx by pan-back frame">',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#444"/>',
        f'<text x="{left}" y="14" font-size="12">mean |dx_px|</text>',
        f'<text x="{width-right-70}" y="{height-8}" font-size="12">back_idx</text>',
    ]
    for y in range(0, int(ymax) + 1, 50):
        out.append(f'<line x1="{left}" y1="{sy(y):.1f}" x2="{width-right}" y2="{sy(y):.1f}" stroke="#eee"/>')
        out.append(f'<text x="6" y="{sy(y)+4:.1f}" font-size="11">{y}</text>')
    for arm, pts in curves.items():
        if not pts:
            continue
        color = colors.get(arm, "#333")
        d = " ".join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in pts)
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{d}"/>')
        lx, ly = pts[-1]
        out.append(f'<text x="{sx(lx)+4:.1f}" y="{sy(ly)+4:.1f}" font-size="12" fill="{color}">{html.escape(arm)}</text>')
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

    by_trial = {a: defaultdict(list) for a in arms}
    for r in rows:
        if r["heading"] <= args.hard_heading:
            by_trial[r["arm"]][(r["scene"], r["seed"])].append(r["psnr"])
    hard_trial = {a: {k: mean(v) for k, v in vals.items()} for a, vals in by_trial.items()}

    paired = [p for p in (paired_stats(hard_trial, a, b) for a, b in PAIRS) if p]

    stats = {}
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        with_info = [r for r in arm_rows if r["dx_px"] is not None or r["resp"] is not None]
        per_trial_info = []
        for key in trials:
            rs = [r for r in with_info if (r["scene"], r["seed"]) == key]
            if not rs:
                continue
            accepted = [r for r in rs if r["projected"]]
            per_trial_info.append({
                "accept": sum(r["projected"] for r in rs) / len(rs),
                "wb": sum(r["wb"] for r in rs) / len(rs),
                "reject_resp": sum(1 for r in rs if r["reject"] == "resp") / len(rs),
                "reject_shift": sum(1 for r in rs if r["reject"] == "shift") / len(rs),
                "dx_accept_mean": mean([abs(r["dx_px"]) for r in accepted if r["dx_px"] is not None]) if accepted else None,
                "resp_mean": mean([r["resp"] for r in rs if r["resp"] is not None]),
            })

        def trial_mean(field):
            vals = [s[field] for s in per_trial_info if s[field] is not None]
            return mean(vals) if vals else None

        stats[arm] = {
            "hard_mean": mean(list(hard_trial.get(arm, {}).values())),
            "hard_std": std(list(hard_trial.get(arm, {}).values())),
            "accept": trial_mean("accept"),
            "wb": trial_mean("wb"),
            "reject_resp": trial_mean("reject_resp"),
            "reject_shift": trial_mean("reject_shift"),
            "dx_accept_mean": trial_mean("dx_accept_mean"),
            "resp_mean": trial_mean("resp_mean"),
        }

    curves = {}
    for arm in ["atlas_nowb", "redream", "redream_nowb", "redream_far", "atlas_redream"]:
        pts = []
        for back_idx in sorted({r["back_idx"] for r in rows}):
            vals = [abs(r["dx_px"]) for r in rows
                    if r["arm"] == arm and r["back_idx"] == back_idx and r["dx_px"] is not None]
            if vals:
                pts.append((back_idx, mean(vals)))
        curves[arm] = pts

    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_head = "unknown"

    log_path = csv_path.with_suffix(".log")
    summary = {
        "csv": str(csv_path),
        "log": str(log_path),
        "git_head": git_head,
        "n_rows": len(rows),
        "n_trials": len(trials),
        "arms": arms,
        "hard_heading": args.hard_heading,
        "paired": paired,
        "stats": stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def link(path, label):
        p = Path(path)
        if p.exists():
            return f'<a href="{html.escape(p.name)}">{html.escape(label)}</a>'
        return html.escape(label)

    pair_rows = "\n".join(
        f"<tr><td>{html.escape(p['target'])} - {html.escape(p['base'])}</td>"
        f"<td>{p['mean']:+.2f} +/- {p['std']:.2f}</td><td>{p['wins']}/{p['n']}</td></tr>"
        for p in paired
    )
    stat_rows = "\n".join(
        f"<tr><td>{html.escape(a)}</td><td>{fmt(s['hard_mean'])} +/- {fmt(s['hard_std'])}</td>"
        f"<td>{pct(s['accept'])}</td>"
        f"<td>{pct(s['wb'])}</td>"
        f"<td>{'' if s['dx_accept_mean'] is None else fmt(s['dx_accept_mean'])}</td>"
        f"<td>{'' if s['resp_mean'] is None else fmt(s['resp_mean'], 3)}</td></tr>"
        for a, s in stats.items()
    )
    pair_lookup = {(p["target"], p["base"]): p for p in paired}
    redream_delta = pair_lookup.get(("redream", "redream_nowb"), {})
    atlas_delta = pair_lookup.get(("atlas_nowb", "revisit"), {})
    redream_dx = stats.get("redream", {}).get("dx_accept_mean")
    nowb_dx = stats.get("redream_nowb", {}).get("dx_accept_mean")

    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Phase 9 Warmstart Sweep</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 28px; color: #1f2933; }}
h1 {{ margin-bottom: 4px; }}
.boundary {{ background: #fff7e6; border: 1px solid #e8c36a; padding: 10px 12px; max-width: 980px; }}
.result {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; max-width: 1100px; }}
.tile {{ border: 1px solid #d7dde5; padding: 10px; border-radius: 6px; }}
.tile b {{ display: block; font-size: 20px; margin-top: 4px; }}
table {{ border-collapse: collapse; margin-top: 14px; min-width: 760px; }}
th, td {{ border: 1px solid #d7dde5; padding: 7px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #edf2f7; }}
svg {{ max-width: 1100px; width: 100%; height: auto; border: 1px solid #d7dde5; margin-top: 12px; }}
code {{ background: #eef2f7; padding: 1px 4px; }}
</style>
<h1>Phase 9 Warmstart Sweep</h1>
<p class="boundary">Controlled sweep for <code>examples/warm_probe.py</code> on
Waypoint-1.5B yaw return-to-start, K=64, 2 scenes x 3 seeds. This is not a new record
attempt; it answers whether laundered write-back restores durability.</p>

<h2>Headline</h2>
<div class="result">
  <div class="tile">redream - redream_nowb<b>{redream_delta.get('mean', float('nan')):+.2f} dB</b>{redream_delta.get('wins', 0)}/{redream_delta.get('n', 0)} positive</div>
  <div class="tile">redream write-back |dx|<b>{fmt(redream_dx, 1)} px</b>accepted frames</div>
  <div class="tile">redream_nowb |dx|<b>{fmt(nowb_dx, 1)} px</b>accepted frames</div>
  <div class="tile">atlas_nowb - revisit<b>{atlas_delta.get('mean', float('nan')):+.2f} dB</b>record replication anchor</div>
</div>

<h2>Artifacts</h2>
<p>{link(csv_path, "warm.csv")} · {link(log_path, "warm.log")} ·
{link(out_dir / "summary.json", "summary.json")} ·
{link(out_dir / "pilot.csv", "pilot.csv")} ·
{link(out_dir / "pilot_failed_offgrid_20260707T0012.log", "failed off-grid pilot log")}</p>
<p>Git head at analysis: <code>{html.escape(git_head)}</code></p>

<h2>Paired Hard-Regime Deltas</h2>
<table><tr><th>comparison</th><th>delta PSNR</th><th>wins</th></tr>{pair_rows}</table>

<h2>Arm Stats</h2>
<table><tr><th>arm</th><th>hard PSNR</th><th>accept</th><th>write-back</th><th>mean |dx| accepted</th><th>mean resp</th></tr>{stat_rows}</table>

<h2>dx Fingerprint</h2>
{svg_dx(curves)}

<h2>Interpretation Boundary</h2>
<p>The controlled result is limited to this runner, model, scheduler grid, yaw protocol,
K=64, and hard-regime metric. It shows the write-back channel fails under laundered
redream at sigma≈0.3; it does not test adaptive sigma, warm_fast, or training-time
persistence.</p>
"""
    (out_dir / "index.html").write_text(html_doc)
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
