# uv run --dev python examples/restamp_pose_probe.py
"""
Decisive experiment: pose-aligned (x,y,t) restamp vs time-only restamp.

Extends restamp_probe.py to compare THREE restamp modes during loop closure:
  restamp       : temporal-only (the current primitive — rotates ONLY the t phase)
  restamp_pose  : full (x,y,t)  (rotates the x/y/t phase by the dead-reckoned camera
                  yaw delta so the memory is spatially aligned to the current heading)

The question: does aligning the spatial RoPE phase of the pinned keyframe close MORE
of the revisit->oracle gap than time-only restamping?

This is the experiment the goal demands: rotating the full (x,y,t) RoPE phase of a
pinned key by the dead-reckoned camera pose delta, run as the decisive test.

Loads ONE all-layers engine (compiles once) and sweeps:
  - temporal offset (how many frames behind the query the memory is placed)
  - number of captured keyframes
for each of the two restamp modes, against fixed revisit/oracle baselines.
"""
import argparse
import sys

sys.path.insert(0, "examples")
from permanence_bench import RealEngine, run_trial, psnr, ssim, load_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--offsets", default="1,2,4,8,16")
    ap.add_argument("--captures", default="1,2,4,8")
    ap.add_argument("--assets", default="bench_assets")
    args = ap.parse_args()

    offsets = [int(x) for x in args.offsets.split(",")]
    captures = [int(x) for x in args.captures.split(",")]
    npin = max(captures)

    eng = RealEngine(args.model, "cuda", None, pin_frames=npin, pin_all_layers=True)
    name, seed_x4 = load_seeds(1, args.assets)[0]
    K, seed, settle = args.K, args.seed, args.settle

    def score(arm, **kw):
        A, B = run_trial(eng, seed_x4, seed, arm, K, settle, "yaw", **kw)
        return psnr(A, B), ssim(A, B)

    # Baselines
    rev_p, rev_s = score("revisit")
    orc_p, orc_s = score("oracle")
    gap_p, gap_s = orc_p - rev_p, orc_s - rev_s
    print(f"scene={name} K={K} seed={seed}  (all-layers, npin={npin})")
    print(f"revisit  PSNR {rev_p:6.2f}  SSIM {rev_s:.3f}")
    print(f"oracle   PSNR {orc_p:6.2f}  SSIM {orc_s:.3f}   forget gap {gap_p:.2f} dB / {gap_s:.3f}\n")

    # Header
    print(f"{'mode':>12} {'offset':>6} {'ncap':>5} | {'PSNR':>6} {'dPSNR':>6} {'closP':>6} | "
          f"{'SSIM':>6} {'closS':>6}")
    print("-" * 70)

    best = {"restamp": None, "restamp_pose": None}

    for mode in ("restamp", "restamp_pose"):
        for ncap in captures:
            for off in offsets:
                p, s = score(mode, restamp_offset=off, n_capture=ncap)
                cp = (p - rev_p) / gap_p if abs(gap_p) > 1e-6 else float("nan")
                cs = (s - rev_s) / gap_s if abs(gap_s) > 1e-6 else float("nan")
                flag = ""
                if best[mode] is None or p > best[mode][0]:
                    best[mode] = (p, off, ncap)
                    flag = "  <-"
                print(f"{mode:>12} {off:6d} {ncap:5d} | {p:6.2f} {p-rev_p:+6.2f} {cp*100:5.0f}% | "
                      f"{s:6.3f} {cs*100:5.0f}%{flag}")

    print()
    for mode in ("restamp", "restamp_pose"):
        b = best[mode]
        if b:
            cp = (b[0] - rev_p) / gap_p * 100 if abs(gap_p) > 1e-6 else float("nan")
            print(f"best {mode:>12}: PSNR {b[0]:.2f} at offset={b[1]} n_capture={b[2]}  "
                  f"({cp:+.0f}% closure)")

    # Headline comparison
    if best["restamp"] and best["restamp_pose"]:
        diff = best["restamp_pose"][0] - best["restamp"][0]
        print(f"\n=== POSE vs TIME-ONLY: {diff:+.2f} dB "
              f"({'POSE WINS' if diff > 0.5 else 'TIE/NO DIFFERENCE' if abs(diff) <= 0.5 else 'TIME-ONLY WINS'}) ===")


if __name__ == "__main__":
    main()
