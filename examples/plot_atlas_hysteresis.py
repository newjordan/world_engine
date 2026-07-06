# uv run --dev python examples/plot_atlas_hysteresis.py bench_out/atlas/ablation_K64.csv
"""
Plot the atlas hysteresis curves from an atlas_probe CSV (columns:
scene,seed,arm,heading,psnr,ssim). One line per arm: mean PSNR vs heading with a ±std
band across (scene, seed) trials. The heading axis doubles as a time-since-reference
axis (gap = 2*(K/2) - (2/yaw_mag)*heading frames); low heading = the forgetting regime,
shaded. See docs/ATLAS_PLAN.md (Phase 6.4).
"""
import csv
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_STYLE = {  # label, color, z-order (atlas on top)
    "revisit":       ("revisit (baseline)",      "#888888", 2),
    "restamp":       ("restamp (single pin)",    "#d1495b", 3),
    "atlas":         ("atlas (nearest+align)",   "#2e7d32", 6),
    "atlas_noalign": ("atlas (nearest, no-align)", "#66bb6a", 5),
    "atlas_far":     ("atlas_far (control)",     "#c9a227", 4),
}


def main(csv_path, out_path=None, hard_heading=3.2):
    rows = list(csv.DictReader(open(csv_path)))
    # (arm, heading) -> [psnr]
    by = defaultdict(list)
    arms_seen = []
    for r in rows:
        arm, h, p = r["arm"], float(r["heading"]), float(r["psnr"])
        by[(arm, h)].append(p)
        if arm not in arms_seen:
            arms_seen.append(arm)
    headings = sorted({h for (_, h) in by})

    fig, ax = plt.subplots(figsize=(9, 5.5))

    order = [a for a in ARM_STYLE if a in arms_seen] + \
            [a for a in arms_seen if a not in ARM_STYLE]
    for arm in order:
        label, color, z = ARM_STYLE.get(arm, (arm, None, 3))
        m = np.array([np.mean(by[(arm, h)]) if (arm, h) in by else np.nan for h in headings])
        sd = np.array([np.std(by[(arm, h)]) if (arm, h) in by else np.nan for h in headings])
        hs = np.array(headings)
        ax.plot(hs, m, "-o", ms=3, lw=1.8, color=color, label=label, zorder=z)
        ax.fill_between(hs, m - sd, m + sd, color=color, alpha=0.12, zorder=z - 0.5)

    # hard-regime band + label (after plotting so autoscale keeps the data limits;
    # y in axes fraction via a blended transform).
    ax.axvspan(min(headings), hard_heading, color="#000000", alpha=0.05, zorder=0)
    ax.text(hard_heading / 2, 0.03, "hard regime (long gap → forgetting)",
            transform=ax.get_xaxis_transform(), va="bottom", ha="center",
            fontsize=9, color="#555555")

    ax.set_xlabel("return heading  (low = long time since reference = hard)")
    ax.set_ylabel("PSNR vs outbound frame at same heading  (dB)")
    ax.set_title("Pose-indexed atlas: return-path consistency vs heading  (K=64)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = out_path or csv_path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench_out/atlas/ablation_K64.csv",
         sys.argv[2] if len(sys.argv) > 2 else None)
