# uv run --dev python examples/anchor_probe.py --arms revisit,atlas_nowb,anchor,anchor_blend --n-scenes 1 --seeds 1 --csv bench_out/anchor/pilot.csv
"""
Phase 10 front-door anchoring probe.

This runner keeps the Phase 9 yaw return-to-start protocol, then measures a
post-intervention durability curve after the memory aid is turned off. Anchor arms
synthesize a registered RGB atlas reconstruction and feed it through the engine's
append-frame path; they do not write edited latents directly into KV.
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

from anchor import (AnchorStore, blend_u8, confidence_alpha, micro_yaw_grid,
                    reconstruct_from_keyframe, reconstruct_temporal_batch,
                    repeat_as_x4)
from atlas import AtlasStore, _clear_pins
from permanence_bench import RealEngine, _noop, load_seeds, psnr, ssim
from rigid import RigidStore, band_mask, rigid_step


ANCHOR_ARMS = {
    "anchor", "anchor_blend", "anchor_k1", "anchor_k4",
    "anchor4_linear", "anchor4_blend", "anchor4_k4",
    "anchor_full", "anchor_full_blend", "anchor_confmap", "anchor_full_far",
    "anchor4_full", "anchor4_full_blend", "anchor4_full_far",
    "anchor4_full_dx64", "anchor4_full_dx96", "anchor4_full_dx128",
    "anchor4_full_resp25", "anchor4_full_dx96_resp25",
    "anchor4_full_dx64_far", "anchor4_full_dx96_far", "anchor4_full_dx128_far",
    "anchor4_full_resp25_far", "anchor4_full_dx96_resp25_far",
    "anchor4_full_pose02", "anchor4_full_pose04",
    "anchor4_full_pose02_far", "anchor4_full_pose04_far",
}
ATLAS_ARMS = {"atlas_nowb"}


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


def _cadence(arm):
    if arm == "anchor_k1":
        return 1
    if arm in ("anchor_k4", "anchor4_k4"):
        return 4
    return None


def _is_anchor4(arm):
    return arm.startswith("anchor4_")


def _retrieve_mode(arm):
    return "farthest" if arm.endswith("_far") else "nearest"


def _mask_mode(arm):
    if arm.startswith("anchor_full") or arm.startswith("anchor4_full"):
        return "full"
    if arm == "anchor_confmap":
        return "confmap"
    return "band"


def _admission_failure(arm, micro_infos):
    has_pose = any(tok in arm for tok in ("pose02", "pose04"))
    has_gate = has_pose or any(tok in arm for tok in ("dx64", "dx96", "dx128", "resp25"))
    if not arm.startswith("anchor4_full_") or not has_gate:
        return None
    projected = [i for i in micro_infos if i.get("projected")]
    if not projected:
        return None
    if not has_pose and len(projected) != len(micro_infos):
        return "admit_partial"
    dx_abs = [abs(i["dx_px"]) for i in projected if i.get("dx_px") is not None]
    resp = [i["resp"] for i in projected if i.get("resp") is not None]
    dx_mean = float(np.mean(dx_abs)) if dx_abs else float("inf")
    resp_mean = float(np.mean(resp)) if resp else 0.0
    if has_pose:
        pose_errs = []
        for i in projected:
            if i.get("target_yaw") is None or i.get("kf_frame_yaw") is None:
                return "admit_pose_unknown"
            pose_errs.append(abs(float(i["target_yaw"]) - float(i["kf_frame_yaw"])))
        pose_max = max(pose_errs) if pose_errs else float("inf")
        if "pose02" in arm and pose_max > 0.2:
            return "admit_pose02"
        if "pose04" in arm and pose_max > 0.4:
            return "admit_pose04"
    if "dx64" in arm and dx_mean > 64.0:
        return "admit_dx64"
    if "dx96" in arm and dx_mean > 96.0:
        return "admit_dx96"
    if "dx128" in arm and dx_mean > 128.0:
        return "admit_dx128"
    if "resp25" in arm and resp_mean < 0.25:
        return "admit_resp25"
    return None


def _summarize_micro_infos(micro_infos, anchor_count):
    projected = [i for i in micro_infos if i.get("projected")]
    resp = [i["resp"] for i in micro_infos if i.get("resp") is not None]
    alpha = [i.get("alpha", 0.0) for i in projected]
    all_projected = len(projected) == len(micro_infos)
    return {
        "projected": bool(projected),
        "dx_px": float(np.mean([i["dx_px"] for i in projected])) if projected else None,
        "resp": float(np.mean(resp)) if resp else None,
        "reject": None if all_projected else ("empty" if not projected else "partial"),
        "anchor": bool(projected),
        "alpha": float(np.mean(alpha)) if alpha else 0.0,
        "anchor_count": anchor_count,
        "micro_infos": micro_infos,
    }


def _info_stats(infos):
    if not infos:
        return {}
    acc = [i for i in infos if i.get("projected")]
    resp = [i["resp"] for i in infos if i.get("resp") is not None]
    return {
        "accept": len(acc) / len(infos),
        "dx_mean": float(np.mean([abs(i["dx_px"]) for i in acc])) if acc else float("nan"),
        "resp_mean": float(np.mean(resp)) if resp else float("nan"),
        "rej_resp": sum(1 for i in infos if i.get("reject") == "resp") / len(infos),
        "rej_shift": sum(1 for i in infos if i.get("reject") == "shift") / len(infos),
    }


def _append_anchor(eng, store, current_rgb, arm, args, anchor_count,
                   current_x4=None, target_yaws=None):
    kf = store.query(float(eng.engine.camera_yaw), mode=_retrieve_mode(arm))
    info = {"projected": False, "dx_px": None, "resp": None, "reject": "empty",
            "anchor": False, "alpha": 0.0, "anchor_count": anchor_count}
    if kf is None:
        return current_rgb, info

    if _is_anchor4(arm):
        if current_x4 is None:
            current_x4 = repeat_as_x4(current_rgb)
        if target_yaws is None:
            target_yaws = [float(eng.engine.camera_yaw)] * len(current_x4)
        recon_x4, micro_infos = reconstruct_temporal_batch(
            store, current_x4, target_yaws,
            retrieve=_retrieve_mode(arm),
            blend=(arm in ("anchor4_blend", "anchor4_full_blend")),
            resp_min=args.resp_min,
            resp_full=args.resp_full,
            max_shift_frac=args.max_shift_frac,
            feather=args.pixel_feather,
            mask_mode=_mask_mode(arm),
        )
        admit_failure = _admission_failure(arm, micro_infos)
        if not any(i.get("projected") for i in micro_infos) or admit_failure:
            info = _summarize_micro_infos(micro_infos, anchor_count)
            if admit_failure:
                info.update(projected=False, reject=admit_failure)
            high_level = dict(info)
            high_level.pop("micro_infos", None)
            info.update(anchor=False, micro_infos=[high_level])
            return current_rgb, info

        ctrl = eng._CtrlInput(button=set(), mouse=(0.0, 0.0))
        four = eng.engine.append_frame(torch.from_numpy(recon_x4).to(eng.device), ctrl=ctrl)
        appended = eng.last_rgb(four)
        anchor_count += 1
        info = _summarize_micro_infos(micro_infos, anchor_count)
        info.update(anchor=True)
        return appended, info

    recon, info = reconstruct_from_keyframe(
        kf, current_rgb,
        resp_min=args.resp_min,
        max_shift_frac=args.max_shift_frac,
        feather=args.pixel_feather,
        mask_mode=_mask_mode(arm),
    )
    if not info["projected"]:
        info.update(anchor=False, alpha=0.0, anchor_count=anchor_count)
        return current_rgb, info

    alpha = 1.0
    if arm in ("anchor_blend", "anchor_full_blend"):
        alpha = confidence_alpha(info.get("resp"), args.resp_min, args.resp_full)
        recon = blend_u8(recon, current_rgb, alpha)

    ctrl = eng._CtrlInput(button=set(), mouse=(0.0, 0.0))
    four = eng.engine.append_frame(torch.from_numpy(repeat_as_x4(recon)).to(eng.device), ctrl=ctrl)
    appended = eng.last_rgb(four)
    anchor_count += 1
    info.update(anchor=True, alpha=alpha, anchor_count=anchor_count)
    return appended, info


def run_trial(eng, seed_x4, seed, arm, args):
    is_anchor = arm in ANCHOR_ARMS
    is_atlas = arm in ATLAS_ARMS
    H = args.K // 2
    torch.manual_seed(seed)
    eng.reset()
    eng.append_seed(seed_x4)

    anchor_store = AnchorStore(tau_insert=args.tau_insert) if is_anchor else None
    rigid_store = None
    rigid_mask = None
    if is_atlas:
        eng.attach_atlas(AtlasStore(tau_insert=args.tau_insert, n_max=100_000))
        rigid_store = RigidStore(tau_insert=args.tau_insert)
        _, _, _, lh, lw = eng.engine.frm_shape
        rigid_mask = band_mask(lh, lw, device=eng.device, feather=args.feather)

    def gen_plain(mouse=(0.0, 0.0)):
        return eng.gen(dict(button=set(), mouse=mouse))

    def gen_atlas_nowb(mouse, capture=False, project=False):
        ctrl = eng._CtrlInput(button=set(), mouse=list(mouse))
        four, info = rigid_step(
            eng.engine, ctrl, rigid_store, rigid_mask,
            project=project,
            capture=capture,
            retrieve="nearest",
            lam=args.lam,
            resp_min=args.resp_min,
            max_shift_frac=args.max_shift_frac,
            write_back=False,
        )
        return four, info

    rows, infos = [], []
    anchor_count = 0

    # Settle -> reference A at heading 0. Capture the start frame.
    A = None
    last_yaw_delta = 0.0
    for i, _spec in enumerate(_noop(args.settle)):
        yaw_before = float(eng.engine.camera_yaw)
        if is_atlas:
            four, _ = gen_atlas_nowb((0.0, 0.0), capture=(i == args.settle - 1))
        else:
            four = gen_plain((0.0, 0.0))
        yaw_after = float(eng.engine.camera_yaw)
        last_yaw_delta = yaw_after - yaw_before
        A = eng.last_rgb(four)
    if is_anchor:
        anchor_store.insert(eng.all_rgb(four), float(eng.engine.camera_yaw),
                            yaw_delta=last_yaw_delta)
    if is_atlas:
        eng.atlas_capture()

    # Pan away, saving reference frames and capturing keyframes every M steps.
    away = []
    for i in range(H):
        yaw_before = float(eng.engine.camera_yaw)
        if is_atlas:
            four, _ = gen_atlas_nowb((+args.yaw_mag, 0.0), capture=(i % args.M == 0))
        else:
            four = gen_plain((+args.yaw_mag, 0.0))
        yaw_after = float(eng.engine.camera_yaw)
        yaw_delta = yaw_after - yaw_before
        away.append(eng.last_rgb(four))
        if is_anchor and i % args.M == 0:
            anchor_store.insert(eng.all_rgb(four), float(eng.engine.camera_yaw),
                                yaw_delta=yaw_delta)
        if is_atlas and i % args.M == 0:
            eng.atlas_capture()

    ref = {0.0: A}
    for i, f in enumerate(away):
        ref[round((i + 1) * args.yaw_mag, 6)] = f

    cadence = _cadence(arm)
    last_rgb = None
    last_x4 = None
    last_target_yaws = None

    # Pan back. atlas_nowb uses cosmetic overlay only here; anchor arms generate
    # normally and optionally insert explicit append-frame interventions.
    for b in range(H):
        yaw_before = float(eng.engine.camera_yaw)
        if is_atlas:
            eng.atlas_activate(offset=args.restamp_offset, k=args.atlas_k,
                               align_pose=True, retrieve="nearest")
            four, info = gen_atlas_nowb((-args.yaw_mag, 0.0), project=True)
        else:
            four = gen_plain((-args.yaw_mag, 0.0))
            info = None
        yaw_after = float(eng.engine.camera_yaw)
        yaw_delta = yaw_after - yaw_before
        last_x4 = eng.all_rgb(four)
        last_target_yaws = micro_yaw_grid(yaw_after, yaw_delta, n=len(last_x4))
        last_rgb = eng.last_rgb(four)
        if info is not None:
            infos.append(info)

        heading = round((H - 1 - b) * args.yaw_mag, 6)
        r = ref.get(heading)
        if r is not None:
            rows.append({
                "phase": "back", "idx": b, "heading": heading,
                "psnr": psnr(r, last_rgb), "ssim": ssim(r, last_rgb),
                "info": info or {}, "anchor_count": anchor_count,
            })

        if is_anchor and cadence and (b % cadence == 0):
            last_rgb, ainfo = _append_anchor(
                eng, anchor_store, last_rgb, arm, args, anchor_count,
                current_x4=last_x4, target_yaws=last_target_yaws)
            anchor_count = ainfo["anchor_count"]
            infos.extend(ainfo.get("micro_infos", [ainfo]))
            if ainfo["anchor"]:
                rows.append({
                    "phase": "intervention", "idx": anchor_count, "heading": heading,
                    "psnr": psnr(A, last_rgb), "ssim": ssim(A, last_rgb),
                    "info": ainfo, "anchor_count": anchor_count,
                })

    # Single loop-closure anchor for non-cadence anchor arms. Cadence arms also anchor
    # at the end if their last cadence did not already land on the final return step.
    needs_final_anchor = is_anchor and (cadence is None or ((H - 1) % cadence != 0))
    if needs_final_anchor:
        last_rgb, ainfo = _append_anchor(
            eng, anchor_store, last_rgb, arm, args, anchor_count,
            current_x4=last_x4, target_yaws=last_target_yaws)
        anchor_count = ainfo["anchor_count"]
        infos.extend(ainfo.get("micro_infos", [ainfo]))
        if ainfo["anchor"]:
            rows.append({
                "phase": "intervention", "idx": anchor_count, "heading": 0.0,
                "psnr": psnr(A, last_rgb), "ssim": ssim(A, last_rgb),
                "info": ainfo, "anchor_count": anchor_count,
            })

    if is_atlas:
        _clear_pins(eng.engine.kv_cache)

    # Post-intervention durability curve: no overlay, no anchors, no controls.
    for p in range(1, args.post_frames + 1):
        four = gen_plain((0.0, 0.0))
        f = eng.last_rgb(four)
        rows.append({
            "phase": "post", "idx": p, "heading": 0.0,
            "psnr": psnr(A, f), "ssim": ssim(A, f),
            "info": {}, "anchor_count": anchor_count,
        })

    return rows, _info_stats(infos)


def _paired(trial, phase, idx, tgt, base):
    t = trial.get((tgt, phase, idx), {})
    b = trial.get((base, phase, idx), {})
    keys = sorted(set(t) & set(b))
    diffs = [t[k] - b[k] for k in keys]
    if not diffs:
        return None
    return _mean(diffs), _std(diffs), sum(1 for d in diffs if d > 0), len(diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--post-frames", type=int, default=64)
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--yaw-mag", type=float, default=0.2)
    ap.add_argument("--M", type=int, default=1, help="capture every M pan-away frames")
    ap.add_argument("--tau-insert", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--feather", type=int, default=1, help="latent feather for atlas_nowb canary")
    ap.add_argument("--pixel-feather", type=int, default=8)
    ap.add_argument("--resp-min", type=float, default=0.05)
    ap.add_argument("--resp-full", type=float, default=0.55)
    ap.add_argument("--max-shift-frac", type=float, default=0.35)
    ap.add_argument("--restamp-offset", type=int, default=4)
    ap.add_argument("--atlas-k", type=int, default=1)
    ap.add_argument("--arms", default="revisit,atlas_nowb,anchor,anchor_blend")
    ap.add_argument("--assets", default="bench_assets")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--n-scenes", type=int, default=2)
    ap.add_argument("--hard-heading", type=float, default=3.2)
    ap.add_argument("--csv", default="bench_out/anchor/anchor.csv")
    ap.add_argument("--condition-source", default="docs/ANCHOR_PLAN.md")
    ap.add_argument("--run-label", default="new_experiment",
                    choices=["new_experiment", "scout", "mechanics_proxy"],
                    help="printed provenance label for the run")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    valid = {"revisit"} | ATLAS_ARMS | ANCHOR_ARMS
    unknown = sorted(set(arms) - valid)
    if unknown:
        raise ValueError(f"unknown arm(s): {unknown}; valid={sorted(valid)}")
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)

    print(f"Condition source: {args.condition_source}")
    print(f"Run label: {args.run_label}")
    print("Read metric: post-intervention PSNR vs start; hard-regime atlas_nowb canary")
    print(f"Command: {shlex.join(sys.argv)}")
    print("Source hashes: "
          f"script={_sha256(sys.argv[0])} "
          f"anchor_probe={_sha256(__file__)} "
          f"anchor={_sha256('examples/anchor.py')} "
          f"plan={_sha256(args.condition_source)}")

    eng = RealEngine(args.model, "cuda", args.quant, max(args.atlas_k, 1),
                     pin_all_layers=True)
    seeds = load_seeds(args.n_scenes, args.assets)
    sigmas = [float(x) for x in eng.engine.scheduler_sigmas.detach().cpu()]
    print(f"Resolved condition: model={args.model} quant={args.quant} device=cuda "
          f"K={args.K} post_frames={args.post_frames} settle={args.settle} "
          f"scenes={len(seeds)} seeds={args.seeds} seed_base={args.seed_base} "
          f"yaw_mag={args.yaw_mag} M={args.M} restamp_offset={args.restamp_offset} "
          f"resp_min={args.resp_min} resp_full={args.resp_full} "
          f"max_shift_frac={args.max_shift_frac} scheduler_sigmas={sigmas} "
          f"arms={arms}", flush=True)

    agg = {(a, "back"): {} for a in arms}
    agg.update({(a, "post"): {} for a in arms})
    trial = {}
    proj_stats = {a: [] for a in arms}
    csv_rows = []

    for scene_name, seed_x4 in seeds:
        for s in range(args.seeds):
            seed = args.seed_base + s
            key = (scene_name, seed)
            for arm in arms:
                rows, stats = run_trial(eng, seed_x4, seed, arm, args)
                if stats:
                    proj_stats[arm].append(stats)
                hard = []
                post = []
                for r in rows:
                    info = r["info"]
                    if r["phase"] == "back":
                        agg[(arm, "back")].setdefault(r["heading"], []).append(r["psnr"])
                        if r["heading"] <= args.hard_heading:
                            hard.append(r["psnr"])
                    elif r["phase"] == "post":
                        agg[(arm, "post")].setdefault(r["idx"], []).append(r["psnr"])
                        post.append(r["psnr"])
                    if r["phase"] == "post":
                        trial[(arm, "post", r["idx"])] = trial.get((arm, "post", r["idx"]), {})
                        trial[(arm, "post", r["idx"])][key] = r["psnr"]
                    csv_rows.append((scene_name, seed, arm, r))
                trial[(arm, "back", "hard")] = trial.get((arm, "back", "hard"), {})
                if hard:
                    trial[(arm, "back", "hard")][key] = _mean(hard)
                extra = (f"  [accept {stats['accept']:.0%} "
                         f"rej resp/shift {stats['rej_resp']:.0%}/{stats['rej_shift']:.0%} "
                         f"|dx| {stats['dx_mean']:.1f}px resp {stats['resp_mean']:.3f}]"
                         if stats else "")
                print(f"  {scene_name} seed {seed} {arm}: "
                      f"back-hard {_mean(hard):.2f} post16 "
                      f"{post[15] if len(post) >= 16 else float('nan'):.2f}{extra}",
                      flush=True)

    print("\nBack hard-regime canary:")
    if "revisit" in arms and "atlas_nowb" in arms:
        got = _paired(trial, "back", "hard", "atlas_nowb", "revisit")
        if got:
            m, sd, wins, n = got
            print(f"  atlas_nowb - revisit: {m:+.2f} +/- {sd:.2f} dB ({wins}/{n} pos)")

    print("\nPost-intervention paired deltas at frame 16:")
    for arm in arms:
        if arm in ("revisit", "atlas_nowb"):
            continue
        if "atlas_nowb" in arms:
            got = _paired(trial, "post", 16, arm, "atlas_nowb")
            if got:
                m, sd, wins, n = got
                print(f"  {arm} - atlas_nowb: {m:+.2f} +/- {sd:.2f} dB ({wins}/{n} pos)")
        if "revisit" in arms:
            got = _paired(trial, "post", 16, arm, "revisit")
            if got:
                m, sd, wins, n = got
                print(f"  {arm} - revisit:    {m:+.2f} +/- {sd:.2f} dB ({wins}/{n} pos)")

    for a in arms:
        if proj_stats[a]:
            acc = _mean([s["accept"] for s in proj_stats[a]])
            rr = _mean([s["rej_resp"] for s in proj_stats[a]])
            rs = _mean([s["rej_shift"] for s in proj_stats[a]])
            dx = _mean([s["dx_mean"] for s in proj_stats[a]])
            resp = _mean([s["resp_mean"] for s in proj_stats[a]])
            print(f"  {a}: accept {acc:.0%} rej resp {rr:.0%} rej shift {rs:.0%} "
                  f"mean |dx| {dx:.1f}px resp {resp:.3f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene", "seed", "arm", "phase", "idx", "heading", "psnr",
                        "ssim", "projected", "dx_px", "resp", "reject", "anchor",
                        "alpha", "anchor_count"])
            for scene_name, seed, arm, r in csv_rows:
                info = r["info"] or {}
                w.writerow([
                    scene_name, seed, arm, r["phase"], r["idx"],
                    f"{r['heading']:.2f}", f"{r['psnr']:.4f}", f"{r['ssim']:.4f}",
                    int(bool(info.get("projected"))),
                    "" if info.get("dx_px") is None else f"{info['dx_px']:.2f}",
                    "" if info.get("resp") is None else f"{info['resp']:.4f}",
                    info.get("reject") or "",
                    int(bool(info.get("anchor"))),
                    f"{float(info.get('alpha', 0.0)):.3f}",
                    int(r.get("anchor_count", info.get("anchor_count", 0))),
                ])
        print(f"\nwrote {args.csv} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
