# uv run --dev python examples/restamp_probe.py
"""
Driver ablation for the loop-closure (restamp) permanence primitive.

Loads ONE all-layers engine (compiles once) and sweeps the two runtime levers --
temporal offset and number of captured keyframes -- against fixed revisit/oracle
baselines, at a fixed scene/seed/K. Answers "which knob buys the gap-closure?" without
paying a torch.compile per configuration (offset and n_capture are runtime-only; only
global-vs-all-layers needs a separate compile, already bracketed by the main sweep).
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

    rev_p, rev_s = score("revisit")
    orc_p, orc_s = score("oracle")
    gap_p, gap_s = orc_p - rev_p, orc_s - rev_s
    print(f"scene={name} K={K} seed={seed}  (all-layers, npin={npin})")
    print(f"revisit  PSNR {rev_p:6.2f}  SSIM {rev_s:.3f}")
    print(f"oracle   PSNR {orc_p:6.2f}  SSIM {orc_s:.3f}   forget gap {gap_p:.2f} dB / {gap_s:.3f}\n")

    print(f"{'offset':>6} {'ncap':>5} | {'PSNR':>6} {'dPSNR':>6} {'closP':>6} | "
          f"{'SSIM':>6} {'closS':>6}")
    print("-" * 56)
    best = None
    for ncap in captures:
        for off in offsets:
            p, s = score("restamp", restamp_offset=off, n_capture=ncap)
            cp = (p - rev_p) / gap_p if abs(gap_p) > 1e-6 else float("nan")
            cs = (s - rev_s) / gap_s if abs(gap_s) > 1e-6 else float("nan")
            flag = ""
            if best is None or p > best[0]:
                best = (p, off, ncap)
                flag = "  <-"
            print(f"{off:6d} {ncap:5d} | {p:6.2f} {p-rev_p:+6.2f} {cp*100:5.0f}% | "
                  f"{s:6.3f} {cs*100:5.0f}%{flag}")
    print(f"\nbest: PSNR {best[0]:.2f} at offset={best[1]} n_capture={best[2]}  "
          f"({(best[0]-rev_p)/gap_p*100:.0f}% closure)")


if __name__ == "__main__":
    main()
