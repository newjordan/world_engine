# uv run --dev python examples/scheduler_probe.py
"""
Decisive experiment: does denoiser step-count / schedule shape affect frame quality?

Current 4-step scheduler [1.0, 0.9, 0.75, 0.3, 0.0]: step 3 has |Δ|=0.45, 4.5x larger
than step 1 (|Δ|=0.10). Covers the perceptually-critical mid-range in one Euler leap.

Tests: same seed frame under different sigma schedules (config override, zero code change).
Oracle = finest schedule. Schedule sensitivity = quality lever.
"""
import argparse, sys
sys.path.insert(0, "examples")
import numpy as np, torch
from permanence_bench import psnr, ssim, load_seeds

SCHEDULES = {
    "current": [1.0, 0.9, 0.75, 0.3, 0.0],
    "unif4":   [1.0, 0.75, 0.5, 0.25, 0.0],
    "fine8":   [1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0.0],
}

def run_schedule(model_uri, device, seed_x4, sigmas, n_frames=12, seed=1234):
    from world_engine import WorldEngine, CtrlInput
    eng = WorldEngine(model_uri, quant=None, device=device,
                      model_config_overrides={"scheduler_sigmas": list(sigmas)})
    torch.manual_seed(seed)
    eng.reset()
    eng.append_frame(seed_x4.to(eng.device))
    frames = []
    for _ in range(n_frames):
        four = eng.gen_frame(ctrl=CtrlInput())
        frames.append(four[-1].detach().cpu().numpy())
    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--assets", default="bench_assets")
    args = ap.parse_args()

    name, seed_x4 = load_seeds(1, args.assets)[0]
    print(f"scene={name} seed={args.seed} frames={args.frames}")
    print(f"oracle=finest(fine8). Schedule sensitivity = quality lever.\n")

    all_frames = {}
    for sched_name, sigmas in SCHEDULES.items():
        print(f"  generating {sched_name} ({len(sigmas)-1} evals)...", flush=True)
        all_frames[sched_name] = run_schedule(
            args.model, "cuda", seed_x4, sigmas,
            n_frames=args.frames, seed=args.seed)

    oracle = "fine8"
    print(f"\n{'schedule':>10} {'evals':>5} | {'PSNR vs oracle':>14} {'SSIM vs oracle':>14}")
    print("-" * 55)
    for sched_name, sigmas in SCHEDULES.items():
        if sched_name == oracle: continue
        psnrs, ssims = [], []
        for i in range(min(len(all_frames[sched_name]), len(all_frames[oracle]))):
            a, b = all_frames[sched_name][i], all_frames[oracle][i]
            psnrs.append(psnr(a, b)); ssims.append(ssim(a, b))
        print(f"{sched_name:>10} {len(sigmas)-1:5d} | {np.mean(psnrs):14.2f} {np.mean(ssims):14.3f}")

    # Same-cost pairwise: current vs uniform4
    print(f"\n{'pairwise (same cost)':>22} | {'PSNR':>8} {'SSIM':>8}")
    print("-" * 44)
    psnrs, ssims = [], []
    for i in range(args.frames):
        psnrs.append(psnr(all_frames["current"][i], all_frames["unif4"][i]))
        ssims.append(ssim(all_frames["current"][i], all_frames["unif4"][i]))
    print(f"{'current vs unif4':>22} | {np.mean(psnrs):8.2f} {np.mean(ssims):8.3f}")

    print("\nInterpretation:")
    print("  Low PSNR current-vs-fine8 => schedule is a quality lever (more steps = better).")
    print("  Low PSNR current-vs-unif4 => schedule SHAPE matters at same cost.")

if __name__ == "__main__":
    main()
