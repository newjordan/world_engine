# uv run --dev python examples/atlas_tune.py --self-test          # CPU smoke test
# uv run --dev python examples/atlas_tune.py --K 64 --seeds 5 --n-scenes 3
"""
Tuning sweep for the pose-indexed atlas (Phase 6.4). One engine / one compile; sweeps
capture density M, temporal offset, and retrieval count k, each scored as the paired
hard-regime gain over the shared no-memory `revisit` baseline.

Primary question — the design thesis: denser capture (smaller M) shrinks the retrieval
residual, so the gain should grow as M -> 1. If it plateaus instead, the frozen-attention
ceiling (Phase 3) is binding, not the residual. Both outcomes are informative.
"""
import argparse
import sys

sys.path.insert(0, "examples")
from atlas_probe import run_hysteresis
from permanence_bench import RealEngine, FakeEngine, load_seeds

# label,            M, offset, k, align
CONFIGS = [
    ("M8  off8  k1", 8, 8, 1, True),
    ("M4  off8  k1", 4, 8, 1, True),
    ("M2  off8  k1", 2, 8, 1, True),
    ("M1  off8  k1", 1, 8, 1, True),   # densest capture -> smallest residual (thesis peak)
    ("M2  off4  k1", 2, 4, 1, True),
    ("M2  off16 k1", 2, 16, 1, True),
    ("M2  off8  k2", 2, 8, 2, True),
    ("M2  off8  k4", 2, 8, 4, True),
]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def hard_mean(rows, hard_h):
    ps = [p for h, p, _ in rows if h <= hard_h]
    return mean(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--n-scenes", type=int, default=3)
    ap.add_argument("--hard-heading", type=float, default=3.2)
    ap.add_argument("--tau-insert", type=float, default=0.05)
    ap.add_argument("--assets", default="bench_assets")
    args = ap.parse_args()

    max_k = max(c[3] for c in CONFIGS)
    if args.self_test:
        eng = FakeEngine(horizon=16)
        seeds = load_seeds(args.n_scenes, args.assets, self_test=True)
    else:
        eng = RealEngine(args.model, "cuda", None, max_k, pin_all_layers=True)
        seeds = load_seeds(args.n_scenes, args.assets)

    trials = [(name, sx, args.seed_base + s)
              for name, sx in seeds for s in range(args.seeds)]
    n = len(trials)
    print(f"K={args.K} trials={n} hard<= {args.hard_heading}  (paired gain over revisit)\n")

    # Shared baseline: revisit is independent of atlas config -> compute once per trial.
    base = []
    for scene, sx, seed in trials:
        rows = run_hysteresis(eng, sx, seed, "revisit", args.K, args.settle,
                              tau_insert=args.tau_insert)
        base.append(hard_mean(rows, args.hard_heading))
    print(f"revisit hard baseline: {mean(base):.2f} +/- {std(base):.2f} dB\n", flush=True)

    results = []
    for label, M, off, k, align in CONFIGS:
        diffs = []
        for i, (scene, sx, seed) in enumerate(trials):
            rows = run_hysteresis(eng, sx, seed, "atlas", args.K, args.settle,
                                  restamp_offset=off, atlas_M=M, atlas_k=k,
                                  atlas_align=align, atlas_retrieve="nearest",
                                  tau_insert=args.tau_insert)
            diffs.append(hard_mean(rows, args.hard_heading) - base[i])
        m, sd = mean(diffs), std(diffs)
        wins = sum(1 for d in diffs if d > 0)
        results.append((label, m, sd, wins))
        print(f"  {label}: {m:+.2f} +/- {sd:.2f} dB  ({wins}/{n} pos)", flush=True)

    print(f"\n=== ranked by hard-regime gain over revisit (n={n}) ===")
    for label, m, sd, wins in sorted(results, key=lambda r: -r[1]):
        flag = "  <-- best" if (label, m, sd, wins) == max(results, key=lambda r: r[1]) else ""
        print(f"  {label}: {m:+.2f} +/- {sd:.2f} dB  ({wins}/{n}){flag}")

    m_series = [(int(lbl.split()[0][1:]), m) for lbl, m, _, _ in results if "off8  k1" in lbl]
    if len(m_series) >= 2:
        m_series.sort()  # ascending by M; m_series[0]=densest (M=1), [-1]=sparsest
        trend = "GROWS as M->1 (thesis holds)" if m_series[0][1] > m_series[-1][1] + 0.1 \
            else "FLAT/INVERTED (residual not the bottleneck)"
        print(f"\nM-density trend (k=1,off8): "
              + "  ".join(f"M{m}={g:+.2f}" for m, g in m_series) + f"   -> {trend}")


if __name__ == "__main__":
    main()
