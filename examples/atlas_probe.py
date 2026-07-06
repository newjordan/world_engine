# uv run --dev python examples/atlas_probe.py --self-test              # CPU smoke test
# uv run --dev python examples/atlas_probe.py --K 64 --seeds 3         # GPU decisive run
"""
Decisive experiment for the pose-indexed world atlas (Phase 6; docs/ATLAS_PLAN.md).

Return-to-start, full-sweep HYSTERESIS scoring: settle at heading 0, pan away K/2, pan
back K/2, and at each pan-back step score the returned frame against the pan-away frame
at the SAME heading (its own outbound counterpart). The metric is return-path
consistency per heading.

Why this isolates the atlas: on the returned frame at heading 0, the single-slot restamp
and the atlas both have the reference view, so they tie. But at every INTERMEDIATE
heading of the return sweep, the single-slot restamp has only the heading-0 reference
(a large, lossy spatial residual), whereas the atlas has a keyframe captured near that
heading on the way out (residual ~0). So the atlas should track the outbound sweep better
than single-frame restamp at headings > 0, and match it at heading 0.

Arms:
  revisit  no memory aid (baseline forgetting)
  restamp  pin the heading-0 reference once, restamp it in-distribution each pan-back step
  atlas    capture keyframes along the pan-away, retrieve+restamp the nearest each step

Caveat (by design): the per-heading reference is the arm's own outbound frame, which
itself drifts. The comparison across arms is fair (same trajectory, same reference
construction); read the numbers as return-path consistency, not absolute ground truth.
"""
import argparse
import csv
import sys

sys.path.insert(0, "examples")
import numpy as np
import torch

from permanence_bench import RealEngine, FakeEngine, psnr, ssim, _noop, load_seeds
from atlas import AtlasStore

# atlas arm variants: name -> (retrieval mode, spatial-align flag). "atlas" is the real
# method; "atlas_noalign" isolates the spatial restamp; "atlas_far" is the pose-
# specificity control (retrieve the WRONG-pose keyframe -> should NOT help if the benefit
# is genuinely pose-indexing rather than just extra attended memory).
ATLAS_VARIANTS = {
    "atlas":         {"retrieve": "nearest",  "align": True},
    "atlas_noalign": {"retrieve": "nearest",  "align": False},
    "atlas_far":     {"retrieve": "farthest", "align": True},
}


def run_hysteresis(eng, seed_x4, seed, arm, K, settle, yaw_mag=0.2,
                   restamp_offset=8, atlas_M=4, atlas_k=1, atlas_align=True,
                   atlas_retrieve="nearest", tau_insert=0.05):
    """One trial. Returns list of (heading, psnr, ssim): each pan-back frame vs the
    pan-away frame at the same heading. Any arm name starting with 'atlas' is an atlas
    variant parameterized by atlas_retrieve/atlas_align."""
    is_atlas = arm.startswith("atlas")
    H = K // 2
    torch.manual_seed(seed)
    eng.reset()
    eng.append_seed(seed_x4)
    if is_atlas:
        eng.attach_atlas(AtlasStore(tau_insert=tau_insert, n_max=100_000))

    # Settle -> reference A at heading 0 (last RGB of the final 4-pack).
    A = None
    for spec in _noop(settle):
        A = eng.last_rgb(eng.gen(spec))
    if arm == "restamp":
        eng.pin_frame()          # pin the heading-0 reference
    elif is_atlas:
        eng.atlas_capture()      # first keyframe: heading 0

    # Pan away, saving each frame and periodically capturing keyframes.
    away = []
    for i in range(H):
        away.append(eng.last_rgb(eng.gen(dict(button=set(), mouse=(+yaw_mag, 0.0)))))
        if is_atlas and (i % atlas_M == 0):
            eng.atlas_capture()

    # Reference-by-heading grid (own outbound frames + A at heading 0).
    ref = {0.0: A}
    for i, f in enumerate(away):
        ref[round((i + 1) * yaw_mag, 6)] = f

    # Pan back: bring memory in-distribution, generate, score vs matched heading.
    rows = []
    for b in range(H):
        if arm == "restamp":
            eng.restamp_memory(restamp_offset, align_pose=True)
        elif is_atlas:
            eng.atlas_activate(offset=restamp_offset, k=atlas_k,
                               align_pose=atlas_align, retrieve=atlas_retrieve)
        f = eng.last_rgb(eng.gen(dict(button=set(), mouse=(-yaw_mag, 0.0))))
        heading = round((H - 1 - b) * yaw_mag, 6)
        r = ref.get(heading)
        if r is not None:
            rows.append((heading, psnr(r, f), ssim(r, f)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--yaw-mag", type=float, default=0.2)
    ap.add_argument("--restamp-offset", type=int, default=8)
    ap.add_argument("--atlas-M", type=int, default=4, help="capture every M pan-away frames")
    ap.add_argument("--atlas-k", type=int, default=1, help="keyframes retrieved per step")
    ap.add_argument("--tau-insert", type=float, default=0.05)
    ap.add_argument("--no-align", action="store_true", help="disable spatial (yaw) restamp")
    ap.add_argument("--arms", default="revisit,restamp,atlas")
    ap.add_argument("--assets", default="bench_assets")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--pin-all-layers", action="store_true", default=True)
    ap.add_argument("--n-scenes", type=int, default=1)
    ap.add_argument("--hard-heading", type=float, default=3.2,
                    help="headings <= this are the 'hard' regime (large time-since-ref)")
    ap.add_argument("--csv", default=None, help="write per-(scene,seed,arm,heading) rows")
    args = ap.parse_args()

    arms = args.arms.split(",")
    if args.self_test:
        eng = FakeEngine(horizon=16)
        seeds = load_seeds(args.n_scenes, args.assets, self_test=True)
    else:
        eng = RealEngine(args.model, "cuda", args.quant,
                         max(args.atlas_k, 1), pin_all_layers=args.pin_all_layers)
        seeds = load_seeds(args.n_scenes, args.assets)

    H = args.K // 2
    print(f"K={args.K} scenes={len(seeds)} seeds={args.seeds} yaw_mag={args.yaw_mag} "
          f"atlas(M={args.atlas_M},k={args.atlas_k},align={not args.no_align})")
    print(f"hysteresis: back[heading] vs away[heading]; heading h -> time gap "
          f"{2*H}-{int(2/args.yaw_mag)}*h frames (low heading = hard).")
    print(f"hard regime = heading <= {args.hard_heading}\n")

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def std(xs):
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    # per-heading aggregation (table) + per-trial regime means (paired significance)
    agg = {a: {} for a in arms}                 # arm -> heading -> [(psnr, ssim)]
    trial = {a: {"hard": [], "easy": []} for a in arms}  # per-trial mean PSNR by regime
    csv_rows = []
    for scene_name, seed_x4 in seeds:
        for s in range(args.seeds):
            seed = args.seed_base + s
            for arm in arms:
                spec = ATLAS_VARIANTS.get(arm, {})
                rows = run_hysteresis(
                    eng, seed_x4, seed, arm, args.K, args.settle,
                    yaw_mag=args.yaw_mag, restamp_offset=args.restamp_offset,
                    atlas_M=args.atlas_M, atlas_k=args.atlas_k,
                    atlas_align=spec.get("align", not args.no_align),
                    atlas_retrieve=spec.get("retrieve", "nearest"),
                    tau_insert=args.tau_insert)
                hard, easy = [], []
                for heading, p, sm in rows:
                    agg[arm].setdefault(heading, []).append((p, sm))
                    (hard if heading <= args.hard_heading else easy).append(p)
                    csv_rows.append((scene_name, seed, arm, heading, p, sm))
                if hard:
                    trial[arm]["hard"].append(mean(hard))
                if easy:
                    trial[arm]["easy"].append(mean(easy))
            print(f"  {scene_name} seed {seed} done", flush=True)

    # per-heading table (means over all trials)
    headings = sorted({h for a in arms for h in agg[a]})
    print(f"\n{'heading':>8} | " + " ".join(f"{a:>8}" for a in arms) + "   (PSNR dB)")
    print("-" * (11 + 9 * len(arms)))
    for h in headings:
        cells = " ".join(f"{mean([p for p, _ in agg[a].get(h, [])]):8.2f}" for a in arms)
        print(f"{h:8.2f} | {cells}")

    # regime summary with per-trial std
    n_trials = len(seeds) * args.seeds
    print(f"\nper-trial mean PSNR by regime (mean +/- std over {n_trials} trials):")
    print(f"{'regime':>10} " + " ".join(f"{a:>14}" for a in arms))
    for reg in ("hard", "easy"):
        cells = " ".join(f"{mean(trial[a][reg]):7.2f}+/-{std(trial[a][reg]):4.2f}" for a in arms)
        print(f"{reg:>10} {cells}")

    # paired target-vs-baseline in the hard (forgetting) regime -> the decisive numbers
    atlas_arms = [a for a in arms if a.startswith("atlas")]
    if atlas_arms:
        def paired(tgt, base):
            diffs = [x - y for x, y in zip(trial[tgt]["hard"], trial[base]["hard"])]
            m, sd = mean(diffs), std(diffs)
            wins = sum(1 for d in diffs if d > 0)
            verdict = ("WINS" if m > 0.3 and m > sd else "LEANS" if m > 0.1
                       else "TIE" if abs(m) <= 0.1 else "LOSES")
            print(f"  {tgt:>13} - {base:<8}: {m:+.2f} +/- {sd:.2f} dB  "
                  f"({wins}/{len(diffs)} pos)  {verdict}")

        print(f"\nPAIRED target - baseline, HARD regime (per-trial, n={n_trials}):")
        for tgt in atlas_arms:
            for base in ("revisit", "restamp"):
                if base in arms:
                    paired(tgt, base)
        # pose-specificity control: is it the nearest keyframe, or just extra memory?
        if "atlas" in arms and "atlas_far" in arms:
            print("  --- pose-specificity control (nearest vs farthest retrieval) ---")
            paired("atlas", "atlas_far")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene", "seed", "arm", "heading", "psnr", "ssim"])
            for r in csv_rows:
                w.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.4f}", f"{r[5]:.4f}"])
        print(f"\nwrote {args.csv} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
