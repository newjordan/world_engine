# uv run --dev python examples/warm_probe.py --arms revisit,redream,redream_nowb --n-scenes 1 --seeds 1 --csv bench_out/warm/pilot.csv
# uv run --dev python examples/warm_probe.py --csv bench_out/warm/warm.csv
"""
Phase 9 warmstart/re-dream probe.

Protocol: the same return-to-start hysteresis used by atlas_probe.py and
rigid_probe.py. The warmstart arms composite a registered memory band into the
sampler proposal, renoise to a scheduler-grid sigma, resume the Euler tail under
the same KV context, and optionally write the re-dreamed latent back.

The decisive comparison is redream - redream_nowb in the hard regime, with the
per-frame dx_px curve as the fingerprint. Positive delta with bounded dx supports
the durability claim; negative delta with growing dx means even on-manifold memory
edits stall the rollout dynamics.
"""
import argparse
import csv
import hashlib
import os
import pathlib
import shlex
import sys

sys.path.insert(0, "examples")

import numpy as np
import torch

from atlas import AtlasStore
from permanence_bench import RealEngine, _noop, load_seeds, psnr, ssim
from rigid import RigidStore, band_mask, rigid_step
from warmstart import redream_step


WARM_VARIANTS = {
    "redream": {},
    "redream_nowb": {"write_back": False},
    "redream_far": {"retrieve": "farthest"},
    "redream75": {"sigma_re": 0.75},
    "atlas_redream": {},
}

RIGID_OVERLAY_VARIANTS = {
    "atlas_nowb": {"write_back": False},
}

ATLAS_ARMS = {"atlas", "atlas_nowb", "atlas_redream"}


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _sha256(path):
    try:
        data = pathlib.Path(path).read_bytes()
    except OSError:
        return "missing"
    return hashlib.sha256(data).hexdigest()[:16]


def _info_stats(infos):
    if not infos:
        return {}
    acc = [i for i in infos if i["projected"]]
    resp = [i["resp"] for i in infos if i["resp"] is not None]
    sig = [i["sigma_re"] for i in infos if i.get("sigma_re") is not None]
    return {
        "accept": len(acc) / len(infos),
        "wb": sum(1 for i in infos if i["wrote_back"]) / len(infos),
        "rej_resp": sum(1 for i in infos if i["reject"] == "resp") / len(infos),
        "rej_shift": sum(1 for i in infos if i["reject"] == "shift") / len(infos),
        "dx_mean": float(np.mean([abs(i["dx_px"]) for i in acc])) if acc else float("nan"),
        "resp_mean": float(np.mean(resp)) if resp else float("nan"),
        "sigma_re": float(np.mean(sig)) if sig else float("nan"),
    }


def run_trial(eng, seed_x4, seed, arm, args):
    """One (scene, seed, arm) hysteresis trial.

    Returns rows = [(back_idx, heading, psnr, ssim, info-or-None)] plus per-trial
    projection stats for warm/overlay arms.
    """
    is_atlas = arm in ATLAS_ARMS
    is_warm = arm in WARM_VARIANTS
    is_overlay = arm in RIGID_OVERLAY_VARIANTS

    warm_kw = {
        "retrieve": "nearest",
        "lam": args.lam,
        "resp_min": args.resp_min,
        "max_shift_frac": args.max_shift_frac,
        "write_back": True,
        "sigma_re": args.sigma_re,
    }
    warm_kw.update(WARM_VARIANTS.get(arm, {}))

    overlay_kw = {
        "retrieve": "nearest",
        "lam": args.lam,
        "resp_min": args.resp_min,
        "max_shift_frac": args.max_shift_frac,
        "write_back": True,
    }
    overlay_kw.update(RIGID_OVERLAY_VARIANTS.get(arm, {}))

    H = args.K // 2
    torch.manual_seed(seed)
    eng.reset()
    eng.append_seed(seed_x4)

    store = mask = None
    if is_atlas:
        eng.attach_atlas(AtlasStore(tau_insert=args.tau_insert, n_max=100_000))
    if is_warm or is_overlay:
        store = RigidStore(tau_insert=args.tau_insert)
        _, _, _, lh, lw = eng.engine.frm_shape
        mask = band_mask(lh, lw, device=eng.device, feather=args.feather)

    def gen(mouse, capture=False, project=False):
        spec = dict(button=set(), mouse=mouse)
        if is_warm:
            ctrl = eng._CtrlInput(button=set(spec["button"]), mouse=list(spec["mouse"]))
            four, info = redream_step(
                eng.engine, ctrl, store, mask, project=project, capture=capture, **warm_kw)
            return eng.last_rgb(four), info
        if is_overlay:
            ctrl = eng._CtrlInput(button=set(spec["button"]), mouse=list(spec["mouse"]))
            four, info = rigid_step(
                eng.engine, ctrl, store, mask, project=project, capture=capture,
                **overlay_kw)
            return eng.last_rgb(four), info
        four = eng.gen(spec)
        return eng.last_rgb(four), None

    # Settle -> reference A at heading 0. Capture the heading-0 latent/keyframe.
    A = None
    for i, _spec in enumerate(_noop(args.settle)):
        A, _ = gen((0.0, 0.0), capture=((is_warm or is_overlay) and i == args.settle - 1))
    if is_atlas:
        eng.atlas_capture()

    # Pan away, saving reference frames and capturing keyframes every M steps.
    away = []
    for i in range(H):
        f, _ = gen((+args.yaw_mag, 0.0), capture=((is_warm or is_overlay) and i % args.M == 0))
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
        f, info = gen((-args.yaw_mag, 0.0), project=(is_warm or is_overlay))
        if info is not None:
            infos.append(info)
        heading = round((H - 1 - b) * args.yaw_mag, 6)
        r = ref.get(heading)
        if r is not None:
            rows.append((b, heading, psnr(r, f), ssim(r, f), info))

    return rows, _info_stats(infos)


def _paired(trial, tgt, base):
    diffs = [x - y for x, y in zip(trial[tgt]["hard"], trial[base]["hard"])]
    if not diffs:
        return None
    m, sd = _mean(diffs), _std(diffs)
    wins = sum(1 for d in diffs if d > 0)
    verdict = ("WINS" if m > 0.3 and m > sd else "LEANS" if m > 0.1
               else "TIE" if abs(m) <= 0.1 else "LOSES")
    return m, sd, wins, len(diffs), verdict


def _print_paired(trial, tgt, base):
    got = _paired(trial, tgt, base)
    if got is None:
        return
    m, sd, wins, n, verdict = got
    print(f"  {tgt:>13} - {base:<13}: {m:+.2f} +/- {sd:.2f} dB  "
          f"({wins}/{n} pos)  {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--yaw-mag", type=float, default=0.2)
    ap.add_argument("--M", type=int, default=1, help="capture every M pan-away frames")
    ap.add_argument("--tau-insert", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0, help="memory blend hardness")
    ap.add_argument("--feather", type=int, default=1)
    ap.add_argument("--resp-min", type=float, default=0.05)
    ap.add_argument("--max-shift-frac", type=float, default=0.35)
    ap.add_argument("--sigma-re", type=float, default=0.3)
    ap.add_argument("--restamp-offset", type=int, default=4)
    ap.add_argument("--atlas-k", type=int, default=1)
    ap.add_argument("--arms", default="revisit,atlas,atlas_nowb,redream,redream_nowb,redream_far,atlas_redream")
    ap.add_argument("--assets", default="bench_assets")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--n-scenes", type=int, default=2)
    ap.add_argument("--hard-heading", type=float, default=3.2)
    ap.add_argument("--csv", default="bench_out/warm/warm.csv")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    valid = {"revisit", "atlas"} | ATLAS_ARMS | set(WARM_VARIANTS) | set(RIGID_OVERLAY_VARIANTS)
    unknown = sorted(set(arms) - valid)
    if unknown:
        raise ValueError(f"unknown arm(s): {unknown}; valid={sorted(valid)}")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)

    print("Condition source: docs/WARMSTART_PLAN.md")
    print("Run label: new_experiment")
    print("Read metric: hard-regime paired PSNR; dx_px per pan-back frame")
    print(f"Command: {shlex.join(sys.argv)}")
    print("Source hashes: "
          f"warm_probe={_sha256(__file__)} "
          f"warmstart={_sha256('examples/warmstart.py')} "
          f"plan={_sha256('docs/WARMSTART_PLAN.md')}")

    eng = RealEngine(args.model, "cuda", args.quant,
                     max(args.atlas_k, 1), pin_all_layers=True)
    seeds = load_seeds(args.n_scenes, args.assets)
    sigmas = [float(x) for x in eng.engine.scheduler_sigmas.detach().cpu()]
    print(f"Resolved condition: model={args.model} quant={args.quant} device=cuda "
          f"K={args.K} settle={args.settle} scenes={len(seeds)} seeds={args.seeds} "
          f"seed_base={args.seed_base} yaw_mag={args.yaw_mag} M={args.M} "
          f"restamp_offset={args.restamp_offset} sigma_re={args.sigma_re} "
          f"lam={args.lam} resp_min={args.resp_min} scheduler_sigmas={sigmas} "
          f"arms={arms}", flush=True)

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
                for back_idx, heading, p, sm, info in rows:
                    agg[arm].setdefault(heading, []).append((p, sm))
                    (hard if heading <= args.hard_heading else easy).append(p)
                    csv_rows.append((scene_name, seed, arm, back_idx, heading, p, sm, info))
                if hard:
                    trial[arm]["hard"].append(_mean(hard))
                if easy:
                    trial[arm]["easy"].append(_mean(easy))
                extra = (f"  [accept {stats['accept']:.0%} "
                         f"wb {stats['wb']:.0%} "
                         f"rej resp/shift {stats['rej_resp']:.0%}/{stats['rej_shift']:.0%} "
                         f"|dx| {stats['dx_mean']:.1f}px resp {stats['resp_mean']:.3f} "
                         f"sigma {stats['sigma_re']:.2f}]"
                         if stats else "")
                print(f"  {scene_name} seed {seed} {arm}: hard {_mean(hard):.2f} "
                      f"easy {_mean(easy):.2f} dB{extra}", flush=True)

    headings = sorted({h for a in arms for h in agg[a]})
    print(f"\n{'heading':>8} | " + " ".join(f"{a:>13}" for a in arms) + "   (PSNR dB)")
    print("-" * (11 + 14 * len(arms)))
    for h in headings:
        cells = " ".join(f"{_mean([p for p, _ in agg[a].get(h, [])]):13.2f}" for a in arms)
        print(f"{h:8.2f} | {cells}")

    n_trials = len(seeds) * args.seeds
    print(f"\nper-trial mean PSNR by regime (mean +/- std over {n_trials} trials):")
    for reg in ("hard", "easy"):
        cells = " ".join(f"{a}={_mean(trial[a][reg]):.2f}+/-{_std(trial[a][reg]):.2f}"
                         for a in arms)
        print(f"  {reg:>5}: {cells}")

    print(f"\nPAIRED target - baseline, HARD regime (per-trial, n={n_trials}):")
    if "revisit" in arms:
        for tgt in [a for a in arms if a != "revisit"]:
            _print_paired(trial, tgt, "revisit")

    print("  --- Phase 9 decision comparisons ---")
    for tgt, base in (
        ("redream", "redream_nowb"),
        ("redream", "redream_far"),
        ("atlas_nowb", "atlas"),
        ("atlas_redream", "atlas"),
        ("atlas_redream", "redream"),
    ):
        if tgt in arms and base in arms:
            _print_paired(trial, tgt, base)

    for a in arms:
        if proj_stats[a]:
            acc = _mean([s["accept"] for s in proj_stats[a]])
            wb = _mean([s["wb"] for s in proj_stats[a]])
            rr = _mean([s["rej_resp"] for s in proj_stats[a]])
            rs = _mean([s["rej_shift"] for s in proj_stats[a]])
            dx = _mean([s["dx_mean"] for s in proj_stats[a]])
            print(f"  {a}: accept {acc:.0%}  wrote-back {wb:.0%}  "
                  f"rej resp {rr:.0%}  rej shift {rs:.0%}  mean |dx| {dx:.1f}px")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene", "seed", "arm", "back_idx", "heading", "psnr", "ssim",
                        "projected", "dx_px", "resp", "reject", "wb", "sigma_re"])
            for r in csv_rows:
                info = r[7] or {}
                w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.2f}", f"{r[5]:.4f}",
                            f"{r[6]:.4f}", int(bool(info.get("projected"))),
                            "" if info.get("dx_px") is None else f"{info['dx_px']:.2f}",
                            "" if info.get("resp") is None else f"{info['resp']:.4f}",
                            info.get("reject") or "",
                            int(bool(info.get("wrote_back"))),
                            "" if info.get("sigma_re") is None else f"{info['sigma_re']:.2f}"])
        print(f"\nwrote {args.csv} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
