# uv run --dev python examples/rigid_probe.py --K 64 --seeds 3          # GPU decisive run
# uv run --dev python examples/rigid_probe.py --K 32 --seeds 1 --arms revisit,rigid  # pilot
"""
Decisive experiment for rigid latent projection (Phase 8; examples/rigid.py).

Protocol: identical to atlas_probe.py's return-to-start hysteresis (settle, pan away
K/2, pan back K/2, score each pan-back frame against the pan-away frame at the same
heading), so the numbers are directly comparable to the Phase 6 atlas results
(atlas beat revisit by +0.33 dB default / +1.67 dB tuned, hard regime).

Arms:
  revisit    no memory aid (baseline forgetting)
  atlas      Phase 6 incumbent: KV keyframes paged into pin slots + restamp
             (attention-side memory: lobbies v via context)
  rigid      Phase 8: latent keyframes hard-projected (lam) into the certified band
             of the sampler output, registered by appearance, written back into the
             KV context (output-side memory: constrains the diffusion line itself)
  rigid_far  falsification control: retrieval forced to the WRONG-pose keyframe. The
             response gate should mostly reject it; if it projects anyway and still
             helps, the benefit is not pose-registered memory. Its accept rate is the
             gate's honesty metric.

Also reported per rigid arm: projection accept rate, mean |dx| (the registration
shift actually applied), mean response.
"""
import argparse
import csv
import sys

sys.path.insert(0, "examples")
import numpy as np
import torch

from permanence_bench import RealEngine, psnr, ssim, _noop, load_seeds
from atlas import AtlasStore
from rigid import RigidStore, band_mask, rigid_step

RIGID_VARIANTS = {
    # overrides applied on top of the CLI defaults, per arm (None = fill from CLI)
    "rigid":        {},                                      # as configured (lam, resp_min from CLI)
    "rigid_far":    {"retrieve": "farthest"},                # wrong-pose falsification control
    "rigid_nowb":   {"write_back": False},                   # project+score but cache the raw x0:
                                                             #   isolates the KV write-back cascade
    "nowb_far":     {"write_back": False, "retrieve": "farthest"},  # pose control for the paste
    "rigid_soft":   {"lam": 0.35},                           # soft blend: paste-shock control
    "rigid_gate":   {"resp_min": None},                      # resp_min from --resp-hi (tuned gate)
    "rigid_anchor": {"wb_dx_max": None},                     # anchor rule: write back only when
                                                             #   |dx| <= --wb-dx-max (aligned)
    "anchor_pure":  {"wb_dx_max": None, "score_projected": False},  # anchor context effect only:
                                                             #   scored frame is the raw decode
    "atlas_nowb":   {"write_back": False},                   # stacked: atlas in attention (via
                                                             #   name prefix) + paste at output
}

CLI_FILL = {"resp_min": "resp_hi", "wb_dx_max": "wb_dx_max"}


def run_trial(eng, seed_x4, seed, arm, args):
    """One (scene, seed, arm) hysteresis trial. Returns (rows, info):
    rows = [(heading, psnr, ssim, info-or-None)]; info = projection stats."""
    is_atlas = arm.startswith("atlas")
    is_rigid = arm in RIGID_VARIANTS
    ov = dict(RIGID_VARIANTS.get(arm, {}))
    for k, v in ov.items():
        if v is None:
            ov[k] = getattr(args, CLI_FILL[k])
    kw = {"retrieve": "nearest", "lam": args.lam, "resp_min": args.resp_min,
          "write_back": True, "wb_dx_max": None, "score_projected": True}
    kw.update(ov)
    H = args.K // 2
    torch.manual_seed(seed)
    eng.reset()
    eng.append_seed(seed_x4)

    store = mask = None
    if is_atlas:
        eng.attach_atlas(AtlasStore(tau_insert=args.tau_insert, n_max=100_000))
    if is_rigid:
        store = RigidStore(tau_insert=args.tau_insert)
        _, _, _, lh, lw = eng.engine.frm_shape
        mask = band_mask(lh, lw, device=eng.device, feather=args.feather)

    def gen(mouse, capture=False, project=False):
        spec = dict(button=set(), mouse=mouse)
        if is_rigid:
            ctrl = eng._CtrlInput(button=set(spec["button"]), mouse=list(spec["mouse"]))
            four, info = rigid_step(
                eng.engine, ctrl, store, mask, project=project, capture=capture, **kw)
            return eng.last_rgb(four), info
        four = eng.gen(spec)
        return eng.last_rgb(four), None

    # Settle -> reference A at heading 0; capture the heading-0 keyframe.
    A = None
    for i, _spec in enumerate(_noop(args.settle)):
        A, _ = gen((0.0, 0.0), capture=(is_rigid and i == args.settle - 1))
    if is_atlas:
        eng.atlas_capture()

    # Pan away, saving frames and capturing keyframes every M steps.
    away = []
    for i in range(H):
        f, _ = gen((+args.yaw_mag, 0.0), capture=(is_rigid and i % args.M == 0))
        away.append(f)
        if is_atlas and (i % args.M == 0):
            eng.atlas_capture()

    ref = {0.0: A}
    for i, f in enumerate(away):
        ref[round((i + 1) * args.yaw_mag, 6)] = f

    # Pan back with the arm's memory mechanism active; score vs matched heading.
    rows, infos = [], []
    for b in range(H):
        if is_atlas:
            eng.atlas_activate(offset=args.restamp_offset, k=args.atlas_k,
                               align_pose=True, retrieve="nearest")
        f, info = gen((-args.yaw_mag, 0.0), project=is_rigid)
        if info is not None:
            infos.append(info)
        heading = round((H - 1 - b) * args.yaw_mag, 6)
        r = ref.get(heading)
        if r is not None:
            rows.append((heading, psnr(r, f), ssim(r, f), info))

    stats = {}
    if infos:
        acc = [i for i in infos if i["projected"]]
        stats = {"accept": len(acc) / len(infos),
                 "wb": sum(1 for i in infos if i["wrote_back"]) / len(infos),
                 "rej_resp": sum(1 for i in infos if i["reject"] == "resp") / len(infos),
                 "rej_shift": sum(1 for i in infos if i["reject"] == "shift") / len(infos),
                 "dx_mean": float(np.mean([abs(i["dx_px"]) for i in acc])) if acc else float("nan"),
                 "resp_mean": float(np.mean([i["resp"] for i in infos if i["resp"] is not None]))}
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--yaw-mag", type=float, default=0.2)
    ap.add_argument("--M", type=int, default=2, help="capture every M pan-away frames")
    ap.add_argument("--tau-insert", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0, help="projection hardness")
    ap.add_argument("--feather", type=int, default=1)
    ap.add_argument("--resp-min", type=float, default=0.05)
    ap.add_argument("--resp-hi", type=float, default=0.20,
                    help="tuned response gate used by the rigid_gate arm")
    ap.add_argument("--wb-dx-max", type=float, default=12.0,
                    help="anchor arms: write back only when |dx| <= this (px; "
                         "~half a latent cell)")
    ap.add_argument("--restamp-offset", type=int, default=8)
    ap.add_argument("--atlas-k", type=int, default=1)
    ap.add_argument("--arms", default="revisit,atlas,rigid,rigid_far")
    ap.add_argument("--assets", default="bench_assets")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--n-scenes", type=int, default=1)
    ap.add_argument("--hard-heading", type=float, default=3.2)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    arms = args.arms.split(",")
    eng = RealEngine(args.model, "cuda", args.quant,
                     max(args.atlas_k, 1), pin_all_layers=True)
    seeds = load_seeds(args.n_scenes, args.assets)

    print(f"K={args.K} scenes={len(seeds)} seeds={args.seeds} lam={args.lam} "
          f"M={args.M} resp_min={args.resp_min} arms={arms}", flush=True)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def std(xs):
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    agg = {a: {} for a in arms}
    trial = {a: {"hard": [], "easy": []} for a in arms}
    proj_stats = {a: [] for a in arms}
    csv_rows = []
    for scene_name, seed_x4 in seeds:
        for s in range(args.seeds):
            seed = args.seed_base + s
            for arm in arms:
                rows, stats = run_trial(eng, seed_x4, seed, arm, args)
                if stats:
                    proj_stats[arm].append(stats)
                hard, easy = [], []
                for heading, p, sm, info in rows:
                    agg[arm].setdefault(heading, []).append((p, sm))
                    (hard if heading <= args.hard_heading else easy).append(p)
                    csv_rows.append((scene_name, seed, arm, heading, p, sm, info))
                if hard:
                    trial[arm]["hard"].append(mean(hard))
                if easy:
                    trial[arm]["easy"].append(mean(easy))
                extra = (f"  [accept {stats['accept']:.0%} "
                         f"rej resp/shift {stats['rej_resp']:.0%}/{stats['rej_shift']:.0%} "
                         f"|dx| {stats['dx_mean']:.1f}px resp {stats['resp_mean']:.3f}]"
                         if stats else "")
                print(f"  {scene_name} seed {seed} {arm}: hard {mean(hard):.2f} "
                      f"easy {mean(easy):.2f} dB{extra}", flush=True)

    headings = sorted({h for a in arms for h in agg[a]})
    print(f"\n{'heading':>8} | " + " ".join(f"{a:>10}" for a in arms) + "   (PSNR dB)")
    print("-" * (11 + 11 * len(arms)))
    for h in headings:
        cells = " ".join(f"{mean([p for p, _ in agg[a].get(h, [])]):10.2f}" for a in arms)
        print(f"{h:8.2f} | {cells}")

    n_trials = len(seeds) * args.seeds
    print(f"\nper-trial mean PSNR by regime (mean +/- std over {n_trials} trials):")
    for reg in ("hard", "easy"):
        cells = " ".join(f"{a}={mean(trial[a][reg]):.2f}+/-{std(trial[a][reg]):.2f}"
                         for a in arms)
        print(f"  {reg:>5}: {cells}")

    print(f"\nPAIRED target - baseline, HARD regime (per-trial, n={n_trials}):")

    def paired(tgt, base):
        diffs = [x - y for x, y in zip(trial[tgt]["hard"], trial[base]["hard"])]
        if not diffs:
            return
        m, sd = mean(diffs), std(diffs)
        wins = sum(1 for d in diffs if d > 0)
        verdict = ("WINS" if m > 0.3 and m > sd else "LEANS" if m > 0.1
                   else "TIE" if abs(m) <= 0.1 else "LOSES")
        print(f"  {tgt:>10} - {base:<8}: {m:+.2f} +/- {sd:.2f} dB  "
              f"({wins}/{len(diffs)} pos)  {verdict}")

    for tgt in [a for a in arms if a.startswith("rigid") or a.startswith("atlas")]:
        if tgt != "revisit" and "revisit" in arms:
            paired(tgt, "revisit")
    if "rigid" in arms and "atlas" in arms:
        paired("rigid", "atlas")
    for tgt in [a for a in arms if a in RIGID_VARIANTS and a not in ("rigid", "rigid_far")]:
        if "rigid" in arms:
            paired(tgt, "rigid")
    for tgt, base in (("rigid_anchor", "rigid_nowb"), ("atlas_nowb", "atlas"),
                      ("anchor_pure", "revisit")):
        if tgt in arms and base in arms:
            paired(tgt, base)
    print("  --- pose-specificity / gate-honesty controls ---")
    for tgt, base in (("rigid", "rigid_far"), ("rigid_nowb", "nowb_far")):
        if tgt in arms and base in arms:
            paired(tgt, base)
    for a in arms:
        if proj_stats[a]:
            acc = mean([s["accept"] for s in proj_stats[a]])
            wb = mean([s["wb"] for s in proj_stats[a]])
            rr = mean([s["rej_resp"] for s in proj_stats[a]])
            rs = mean([s["rej_shift"] for s in proj_stats[a]])
            print(f"  {a}: accept {acc:.0%}  wrote-back {wb:.0%}  "
                  f"rej resp {rr:.0%}  rej shift {rs:.0%}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene", "seed", "arm", "heading", "psnr", "ssim",
                        "projected", "dx_px", "resp", "reject", "wb"])
            for r in csv_rows:
                info = r[6] or {}
                w.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.4f}", f"{r[5]:.4f}",
                            int(bool(info.get("projected"))),
                            "" if info.get("dx_px") is None else f"{info['dx_px']:.2f}",
                            "" if info.get("resp") is None else f"{info['resp']:.4f}",
                            info.get("reject") or "",
                            int(bool(info.get("wrote_back")))])
        print(f"\nwrote {args.csv} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
