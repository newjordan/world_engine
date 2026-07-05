# uv run --dev python examples/combine_curves.py \
#     --out bench_out/combined \
#     still,revisit,oracle=bench_out/sweep/results.csv \
#     "revisit+pin=bench_out/sweep_pinned/results.csv"
"""
Merge several benchmark CSVs into one PSNR/SSIM-vs-K comparison (table + plot).

Each positional arg is `LABELSPEC=path/to/results.csv`, where LABELSPEC is either
a comma-separated list of arm names to pull from that CSV (kept under their own
names), or `newlabel=` renaming a CSV's single arm. Used for Phase 3/4 to overlay
the pinned revisit curve on the baseline still/revisit/oracle curves.
"""
import argparse
import csv
import math
import os


def load(spec):
    """Return {label: {K: [values...]}} for psnr and ssim from one `arms=csv` spec."""
    label_part, path = spec.split("=", 1)
    wanted = label_part.split(",")
    rows = list(csv.DictReader(open(path)))
    arms_in_csv = sorted({r["arm"] for r in rows})
    # rename mode: a single label mapping onto a CSV that has exactly one arm
    rename = None
    if len(wanted) == 1 and wanted[0] not in arms_in_csv:
        assert len(arms_in_csv) == 1, f"{path}: rename needs exactly one arm, has {arms_in_csv}"
        rename = {arms_in_csv[0]: wanted[0]}
        wanted = arms_in_csv
    out = {}
    for r in rows:
        if r["arm"] not in wanted:
            continue
        label = rename[r["arm"]] if rename else r["arm"]
        K = int(r["K"])
        d = out.setdefault(label, {}).setdefault(K, {"psnr": [], "ssim": []})
        for m in ("psnr", "ssim"):
            try:
                v = float(r[m])
                if math.isfinite(v):
                    d[m].append(v)
            except (ValueError, KeyError):
                pass
    return out


def mean_std(xs):
    if not xs:
        return None
    m = sum(xs) / len(xs)
    s = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return m, s, len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_out/combined")
    ap.add_argument("specs", nargs="+", help="LABELS=results.csv")
    args = ap.parse_args()

    data = {}
    for spec in args.specs:
        for label, byK in load(spec).items():
            data.setdefault(label, {}).update(byK)

    # canonical column order if present
    order = ["oracle", "still", "revisit", "revisit+pin"]
    labels = [l for l in order if l in data] + [l for l in data if l not in order]
    Ks = sorted({K for byK in data.values() for K in byK})

    os.makedirs(args.out, exist_ok=True)
    lines = ["# Baseline vs pinned forgetting curves (mean ± std)\n"]
    for m in ("psnr", "ssim"):
        lines.append(f"\n## {m.upper()}\n")
        cols = labels[:]
        extra = []
        if "revisit" in data and "still" in data:
            extra.append("revisit−still")
        if "revisit+pin" in data and "still" in data:
            extra.append("pin−still")
        if "revisit+pin" in data and "revisit" in data:
            extra.append("pin−revisit")
        lines.append("| K | " + " | ".join(cols + extra) + " |")
        lines.append("|" + "---|" * (len(cols) + len(extra) + 1))
        for K in Ks:
            cells = []
            ms = {}
            for l in cols:
                v = mean_std(data.get(l, {}).get(K, {}).get(m, []))
                ms[l] = v[0] if v else None
                cells.append(f"{v[0]:.3f}±{v[1]:.3f}" if v else "—")
            for e in extra:
                a, b = {"revisit−still": ("revisit", "still"),
                        "pin−still": ("revisit+pin", "still"),
                        "pin−revisit": ("revisit+pin", "revisit")}[e]
                cells.append(f"{ms[a] - ms[b]:+.3f}" if (ms.get(a) is not None and ms.get(b) is not None) else "—")
            lines.append(f"| {K} | " + " | ".join(cells) + " |")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out, "combined_curve.md"), "w") as f:
        f.write(report + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        style = {"oracle": ("C2", "s"), "still": ("C0", "o"),
                 "revisit": ("C3", "^"), "revisit+pin": ("C1", "D")}
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, m in zip(axes, ("psnr", "ssim")):
            for l in labels:
                xs = [K for K in Ks if data.get(l, {}).get(K, {}).get(m)]
                pts = [mean_std(data[l][K][m]) for K in xs]
                ys = [p[0] for p in pts]
                es = [p[1] for p in pts]
                c, mk = style.get(l, ("C7", "x"))
                ax.errorbar(xs, ys, yerr=es, marker=mk, color=c, capsize=3, label=l)
            ax.axvline(16, ls="--", c="gray", alpha=0.6)
            ax.axvline(128, ls=":", c="gray", alpha=0.6)
            ax.annotate("local\nhorizon", (16, ax.get_ylim()[0]), fontsize=7, color="gray")
            ax.annotate("global\nhorizon", (128, ax.get_ylim()[0]), fontsize=7, color="gray")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("K — excursion length (latent frames)")
            ax.set_ylabel(m.upper())
            ax.set_title(f"{m.upper()}: revisit consistency vs excursion length")
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(args.out, "combined_curve.png")
        fig.savefig(p, dpi=120)
        print(f"\nWrote {p}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
