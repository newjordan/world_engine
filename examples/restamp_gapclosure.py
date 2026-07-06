# uv run --dev python examples/restamp_gapclosure.py bench_out/results.csv
"""
Gap-closure analysis for the loop-closure (restamp) arm of permanence_bench.

The revisit arm forgets; the oracle arm is the perfect-memory ceiling. The question is
how much of the revisit->oracle gap the in-distribution keyframe re-stamp recovers:

    closure = (restamp - revisit) / (oracle - revisit)

Reported per excursion length K for PSNR and SSIM, with a verdict:
  >= ~15% closure at K past the horizons  -> primitive works, build the pose-indexed store
  ~0% (within noise)                      -> inference-side permanence falsified (train it in)
"""
import csv
import math
import sys


def load(path):
    rows = list(csv.DictReader(open(path)))
    agg = {}  # (arm, K, metric) -> list of values
    for r in rows:
        for m in ("psnr", "ssim"):
            try:
                v = float(r[m])
            except (ValueError, KeyError, TypeError):
                continue
            if math.isfinite(v):
                agg.setdefault((r["arm"], int(r["K"]), m), []).append(v)
    return agg


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main(path):
    agg = load(path)
    Ks = sorted({k for (_, k, _) in agg})
    arms = {a for (a, _, _) in agg}
    have = {"revisit", "oracle", "restamp"} <= arms
    print(f"# Loop-closure gap analysis  ({path})\n")
    if not have:
        print(f"need arms revisit,oracle,restamp; have {sorted(arms)}")
        return

    for m in ("psnr", "ssim"):
        unit = "dB" if m == "psnr" else ""
        print(f"## {m.upper()}")
        print(f"| K | still | revisit | restamp | oracle | forget gap | restamp−revisit | closure |")
        print("|---|---|---|---|---|---|---|---|")
        closures = []
        for K in Ks:
            still = mean(agg.get(("still", K, m)))
            rev = mean(agg.get(("revisit", K, m)))
            orc = mean(agg.get(("oracle", K, m)))
            res = mean(agg.get(("restamp", K, m)))
            if None in (rev, orc, res):
                continue
            gap = orc - rev
            recov = res - rev
            closure = (recov / gap) if abs(gap) > 1e-6 else float("nan")
            if abs(gap) > (0.3 if m == "psnr" else 0.01):  # only score meaningful gaps
                closures.append(closure)
            s = f"{still:.2f}" if still is not None else "—"
            print(f"| {K} | {s} | {rev:.2f} | {res:.2f} | {orc:.2f} | "
                  f"{gap:+.2f}{unit} | {recov:+.2f}{unit} | {closure*100:+.0f}% |")
        if closures:
            avg = sum(closures) / len(closures)
            print(f"\n  mean closure over K with a real gap: {avg*100:+.1f}%")
            verdict = ("PRIMITIVE WORKS — build the pose-indexed store" if avg >= 0.15
                       else "NO-OP — inference-side permanence falsified; train it in"
                       if avg < 0.05 else "PARTIAL — worth a deeper look")
            print(f"  verdict ({m}): {verdict}")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench_out/results.csv")
