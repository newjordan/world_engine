# uv run --dev python examples/anchor_pan_video.py --out bench_out/anchor_admission/pan180_case0
"""
Controlled full-pan MP4 review for front-door anchoring arms.

This is a visual review runner, not the aggregate benchmark. It uses the same
model path and append-frame anchoring mechanics as anchor_probe.py, but records
the whole trajectory as MP4: settle -> pan away to a content-measured target angle
-> symmetric return -> optional anchor append -> no-op post window.
"""
import argparse
import hashlib
import html
import json
import os
import pathlib
import shutil
import shlex
import subprocess
import sys

sys.path.insert(0, "examples")

import cv2
import numpy as np
import torch

from anchor import (AnchorStore, blend_u8, confidence_alpha, micro_yaw_grid,
                    reacq_reconstruct, reconstruct_from_keyframe,
                    reconstruct_temporal_batch, repeat_as_x4)
from anchor_probe import (ANCHOR_ARMS, _admission_failure, _cadence, _is_reacq,
                          _mask_mode, _needs_final_anchor, _reacq_pool,
                          _retrieve_mode, _summarize_micro_infos)
from atlas import AtlasStore, _clear_pins
from permanence_bench import RealEngine, _noop, load_seeds, psnr
from rigid import RigidStore, band_mask, rigid_step

ATLAS_ARMS = {"atlas_nowb"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _sha256(path):
    try:
        data = pathlib.Path(path).read_bytes()
    except OSError:
        return "missing"
    return hashlib.sha256(data).hexdigest()[:16]


def _jsonable(value):
    """Coerce numpy scalars/arrays (possibly nested) to JSON-serializable types."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _cadence_attempt(b, deg, info, closure):
    """Manifest row for one cadence anchor attempt (per-micro detail elided)."""
    entry = {"b": int(b), "deg": float(deg), "closure": bool(closure)}
    entry.update({k: v for k, v in info.items() if k != "micro_infos"})
    return _jsonable(entry)


def pc_shift(prev, gray):
    h, w = gray.shape
    roi = (slice(int(0.28 * h), int(0.64 * h)), slice(int(0.15 * w), int(0.85 * w)))
    a, b = np.float32(prev[roi]), np.float32(gray[roi])
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, _), _resp = cv2.phaseCorrelate(a, b, win)
    return abs(dx)


def _append_anchor_x4(eng, store, current_rgb, arm, args, current_x4, target_yaws):
    if _is_reacq(arm):
        yaw = float(eng.engine.camera_yaw)
        recon_x4, micro_infos, diag = reacq_reconstruct(
            store, current_x4, yaw,
            k=args.reacq_k, mode=_reacq_pool(arm),
            resp_min=args.resp_min, resp_full=args.resp_full,
            max_shift_frac=args.max_shift_frac, feather=args.pixel_feather,
            mask_mode=_mask_mode(arm))
        diag_fields = {k: v for k, v in diag.items() if k != "reject"}
        if recon_x4 is None:
            return None, {"projected": False, "anchor": False,
                          "reject": diag["reject"], **diag_fields}
        admit_failure = _admission_failure(arm, micro_infos)
        if not any(i.get("projected") for i in micro_infos) or admit_failure:
            info = _summarize_micro_infos(micro_infos, 0)
            if admit_failure:
                info.update(projected=False, reject=admit_failure)
            elif not info["projected"]:
                info.update(reject="nomatch")
            info.update(anchor=False, **diag_fields)
            return None, info
        ctrl = eng._CtrlInput(button=set(), mouse=(0.0, 0.0))
        four = eng.engine.append_frame(torch.from_numpy(recon_x4).to(eng.device), ctrl=ctrl)
        info = _summarize_micro_infos(micro_infos, 1)
        info.update(anchor=True, **diag_fields)
        return eng.all_rgb(four), info

    kf = store.query(float(eng.engine.camera_yaw), mode=_retrieve_mode(arm))
    if kf is None:
        return None, {"projected": False, "anchor": False, "reject": "empty"}

    if arm.startswith("anchor4_"):
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
            info = _summarize_micro_infos(micro_infos, 0)
            if admit_failure:
                info.update(projected=False, reject=admit_failure)
            info.update(anchor=False)
            return None, info
        ctrl = eng._CtrlInput(button=set(), mouse=(0.0, 0.0))
        four = eng.engine.append_frame(torch.from_numpy(recon_x4).to(eng.device), ctrl=ctrl)
        info = _summarize_micro_infos(micro_infos, 1)
        info.update(anchor=True)
        return eng.all_rgb(four), info

    recon, info = reconstruct_from_keyframe(
        kf, current_rgb,
        resp_min=args.resp_min,
        max_shift_frac=args.max_shift_frac,
        feather=args.pixel_feather,
        mask_mode=_mask_mode(arm),
    )
    if not info["projected"]:
        info.update(anchor=False)
        return None, info
    if arm in ("anchor_blend", "anchor_full_blend"):
        alpha = confidence_alpha(info.get("resp"), args.resp_min, args.resp_full)
        recon = blend_u8(recon, current_rgb, alpha)
    else:
        alpha = 1.0
    ctrl = eng._CtrlInput(button=set(), mouse=(0.0, 0.0))
    four = eng.engine.append_frame(torch.from_numpy(repeat_as_x4(recon)).to(eng.device), ctrl=ctrl)
    info.update(anchor=True, alpha=alpha)
    return eng.all_rgb(four), info


class FrameSpool:
    """Disk-backed frame list: PNG spill keeps peak resident frames O(1)."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = []

    def append(self, phase, deg, rgb):
        path = self.root / f"{len(self.meta):06d}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(rgb),
                                                   cv2.COLOR_RGB2BGR)):
            raise OSError(f"frame spill failed: {path}")
        self.meta.append({"phase": phase, "deg": float(deg), "path": str(path)})

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        m = self.meta[i]
        bgr = cv2.imread(m["path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"frame spill unreadable: {m['path']}")
        return {"phase": m["phase"], "deg": m["deg"],
                "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)}


def run_arm(eng, seed_x4, seed, arm, args, spool_dir, n_away=None):
    is_anchor = arm in ANCHOR_ARMS
    is_atlas = arm in ATLAS_ARMS
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
        return rigid_step(
            eng.engine, ctrl, rigid_store, rigid_mask,
            project=project,
            capture=capture,
            retrieve="nearest",
            lam=args.lam,
            resp_min=args.resp_min,
            max_shift_frac=args.max_shift_frac,
            write_back=False,
        )

    video = FrameSpool(spool_dir)
    start = None
    last_yaw_delta = 0.0
    prev_gray = None
    ppr = None
    for i, spec in enumerate(_noop(args.settle)):
        yaw_before = float(eng.engine.camera_yaw)
        if is_atlas:
            four, _info = gen_atlas_nowb((0.0, 0.0), capture=(i == args.settle - 1))
        else:
            four = gen_plain(spec["mouse"])
        yaw_after = float(eng.engine.camera_yaw)
        last_yaw_delta = yaw_after - yaw_before
        for r in eng.all_rgb(four)[::args.subsample]:
            video.append("settle", 0.0, r)
        start = eng.last_rgb(four)
        prev_gray = cv2.cvtColor(start, cv2.COLOR_RGB2GRAY)
        ppr = start.shape[1] * (360.0 / args.fov)

    if is_anchor:
        anchor_store.insert(eng.all_rgb(four), float(eng.engine.camera_yaw),
                            yaw_delta=last_yaw_delta)
    if is_atlas:
        eng.atlas_capture()

    away_steps = 0
    away_deg = 0.0
    while True:
        yaw_before = float(eng.engine.camera_yaw)
        if is_atlas:
            four, _info = gen_atlas_nowb((+args.yaw_mag, 0.0), capture=(away_steps % args.M == 0))
        else:
            four = gen_plain((+args.yaw_mag, 0.0))
        yaw_after = float(eng.engine.camera_yaw)
        yaw_delta = yaw_after - yaw_before
        rgb = eng.last_rgb(four)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        away_deg += pc_shift(prev_gray, gray) / ppr * 360.0
        prev_gray = gray
        for r in eng.all_rgb(four)[::args.subsample]:
            video.append("pan away", away_deg, r)
        if is_anchor and away_steps % args.M == 0:
            anchor_store.insert(eng.all_rgb(four), float(eng.engine.camera_yaw),
                                yaw_delta=yaw_delta)
        if is_atlas and away_steps % args.M == 0:
            eng.atlas_capture()
        away_steps += 1
        if (n_away is not None and away_steps >= n_away) or (
            n_away is None and (away_deg >= args.target_deg or away_steps >= args.cap)
        ):
            break

    cadence = _cadence(arm)
    cadence_attempts = []
    last_rgb = None
    last_x4 = None
    last_target_yaws = None
    for b in range(away_steps):
        yaw_before = float(eng.engine.camera_yaw)
        if is_atlas:
            eng.atlas_activate(offset=args.restamp_offset, k=args.atlas_k,
                               align_pose=True, retrieve="nearest")
            four, _info = gen_atlas_nowb((-args.yaw_mag, 0.0), project=True)
        else:
            four = gen_plain((-args.yaw_mag, 0.0))
        yaw_after = float(eng.engine.camera_yaw)
        yaw_delta = yaw_after - yaw_before
        last_x4 = eng.all_rgb(four)
        last_target_yaws = micro_yaw_grid(yaw_after, yaw_delta, n=len(last_x4))
        last_rgb = eng.last_rgb(four)
        remaining = max(0.0, away_deg * (1.0 - float(b + 1) / max(away_steps, 1)))
        for r in last_x4[::args.subsample]:
            video.append("return", remaining, r)

        if is_anchor and cadence and (b % cadence == 0):
            anchor_x4, ainfo = _append_anchor_x4(
                eng, anchor_store, last_rgb, arm, args, last_x4, last_target_yaws)
            cadence_attempts.append(_cadence_attempt(
                b, remaining, ainfo, closure=(b == away_steps - 1)))
            if anchor_x4 is not None:
                for r in anchor_x4[::args.subsample]:
                    video.append("anchor cadence", remaining, r)

    anchor_info = {}
    if is_anchor and _needs_final_anchor(cadence, away_steps):
        anchor_x4, anchor_info = _append_anchor_x4(
            eng, anchor_store, last_rgb, arm, args, last_x4, last_target_yaws)
        if anchor_x4 is not None:
            for r in anchor_x4[::args.subsample]:
                video.append("anchor append", 0.0, r)
        if cadence:
            cadence_attempts.append(_cadence_attempt(
                away_steps - 1, 0.0, anchor_info, closure=True))

    if is_atlas:
        _clear_pins(eng.engine.kv_cache)

    post = {}
    for p in range(1, args.post_frames + 1):
        four = gen_plain((0.0, 0.0))
        rgb = eng.last_rgb(four)
        if p in (16, 64):
            post[p] = rgb
        for r in eng.all_rgb(four)[::args.subsample]:
            video.append(f"post {p:02d}", 0.0, r)

    return {
        "arm": arm,
        "video": video,
        "start": start,
        "away_steps": away_steps,
        "away_deg": away_deg,
        "anchor_info": anchor_info,
        "cadence_attempts": cadence_attempts,
        "post16_psnr": psnr(start, post[16]) if 16 in post else None,
        "post64_psnr": psnr(start, post[64]) if 64 in post else None,
    }


def panel(entry, title, metric):
    img = cv2.resize(entry["rgb"], (960, 540), interpolation=cv2.INTER_AREA)
    out = img.copy()
    cv2.rectangle(out, (0, 0), (960, 44), (0, 0, 0), -1)
    cv2.rectangle(out, (0, 496), (960, 540), (0, 0, 0), -1)
    cv2.putText(out, title, (14, 29), FONT, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"{entry['phase']} | {entry['deg']:.0f} deg", (14, 526),
                FONT, 0.68, (0, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(out, metric, (560, 526), FONT, 0.6, (210, 235, 255), 2, cv2.LINE_AA)
    return out


def metric_text(res):
    info = res.get("anchor_info") or {}
    parts = []
    if res.get("post16_psnr") is not None:
        parts.append(f"p16 {res['post16_psnr']:.1f}dB")
    if info:
        parts.append("anchored" if info.get("anchor") else f"reject {info.get('reject', '')}")
    attempts = res.get("cadence_attempts") or []
    if attempts:
        accepted = sum(1 for a in attempts if a.get("anchor"))
        parts.append(f"cad {accepted}/{len(attempts)}")
    return " | ".join(parts)


def write_single_mp4(path, res, fps):
    metric = metric_text(res)
    write_mp4(path, lambda: (panel(v, res["arm"], metric) for v in res["video"]), fps)


def write_compare_mp4(path, results, fps):
    order = list(results)
    while len(order) < 4:
        order.append(order[-1])
    order = order[:4]
    max_len = max(len(r["video"]) for r in order)

    def frames():
        for i in range(max_len):
            cells = [panel(r["video"][min(i, len(r["video"]) - 1)], r["arm"],
                           metric_text(r)) for r in order]
            yield np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:4])])

    write_mp4(path, frames, fps)


def write_mp4(path, make_frames, fps):
    # make_frames: zero-arg callable returning a fresh frame iterator, so the
    # cv2 fallback can restart the stream if imageio fails partway.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import imageio
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1) as w:
            for f in make_frames():
                w.append_data(f)
        return
    except Exception:
        pass
    it = iter(make_frames())
    first = next(it)
    h, w = first.shape[:2]
    tmp_path = str(pathlib.Path(path).with_suffix(".tmp.mp4"))
    vw = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    vw.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
    for f in it:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", tmp_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
            "-movflags", "+faststart", path,
        ]
        try:
            subprocess.run(cmd, check=True)
            os.unlink(tmp_path)
            return
        except subprocess.CalledProcessError:
            pass
    os.replace(tmp_path, path)


def cleanup_frames(out, keep_frames):
    if keep_frames:
        return
    shutil.rmtree(os.path.join(out, "tmp_frames"), ignore_errors=True)


def write_index(out, manifest, videos):
    rows = "\n".join(
        f"<tr><td>{html.escape(v['name'])}</td><td><a href='{html.escape(v['file'])}'>{html.escape(v['file'])}</a></td></tr>"
        for v in videos
    )
    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Anchor 180 Pan MP4 Review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 28px; color: #111827; }}
.boundary {{ background: #fff7e6; border: 1px solid #e8c36a; padding: 10px 12px; max-width: 1120px; }}
video {{ width: min(100%, 1120px); display: block; margin: 14px 0 24px; background: #000; }}
table {{ border-collapse: collapse; margin-top: 12px; min-width: 760px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 7px 9px; text-align: left; }}
th {{ background: #e5edf6; }}
code {{ background: #eef2f7; padding: 1px 4px; }}
</style>
<h1>Anchor 180 Pan MP4 Review</h1>
<p class="boundary">Controlled visual review. Videos are real generated frames from
<code>examples/anchor_pan_video.py</code>. They are not the aggregate benchmark;
use the linked sweep CSV/report for aggregate claims.</p>
<h2>Comparison</h2>
<video src="compare_2x2.mp4" controls muted loop></video>
<h2>Files</h2>
<table><tr><th>name</th><th>file</th></tr>{rows}</table>
<h2>Provenance</h2>
<pre>{html.escape(json.dumps(manifest, indent=2))}</pre>
"""
    pathlib.Path(out, "index.html").write_text(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--condition-source", default="docs/ANCHOR_ADMISSION_PLAN.md")
    ap.add_argument("--assets", default="bench_assets")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--arms", default="revisit,atlas_nowb,anchor4_full,anchor4_full_dx96")
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--target-deg", type=float, default=180.0)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--yaw-mag", type=float, default=1.0)
    ap.add_argument("--cap", type=int, default=240)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--post-frames", type=int, default=64)
    ap.add_argument("--tau-insert", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--feather", type=int, default=1)
    ap.add_argument("--pixel-feather", type=int, default=8)
    ap.add_argument("--resp-min", type=float, default=0.05)
    ap.add_argument("--resp-full", type=float, default=0.55)
    ap.add_argument("--max-shift-frac", type=float, default=0.35)
    ap.add_argument("--reacq-k", type=int, default=8, help="reacq candidate pool size")
    ap.add_argument("--restamp-offset", type=int, default=4)
    ap.add_argument("--atlas-k", type=int, default=1)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="bench_out/anchor_admission/pan180_case0")
    ap.add_argument("--keep-frames", action="store_true",
                    help="retain the tmp_frames/ PNG spill after MP4s are written")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    valid = {"revisit"} | ATLAS_ARMS | ANCHOR_ARMS
    unknown = sorted(set(arms) - valid)
    if unknown:
        raise ValueError(f"unknown arm(s): {unknown}; valid={sorted(valid)}")
    os.makedirs(args.out, exist_ok=True)

    command = shlex.join(sys.argv)
    print(f"Condition source: {args.condition_source}")
    print("Run label: visual_review")
    print(f"Command: {command}")
    print("Source hashes: "
          f"anchor_pan_video={_sha256(__file__)} "
          f"anchor_probe={_sha256('examples/anchor_probe.py')} "
          f"anchor={_sha256('examples/anchor.py')} "
          f"plan={_sha256(args.condition_source)}")

    eng = RealEngine(args.model, "cuda", args.quant, max(args.atlas_k, 1),
                     pin_all_layers=True)
    scene_name, seed_x4 = load_seeds(args.scene + 1, args.assets)[args.scene]
    sigmas = [float(x) for x in eng.engine.scheduler_sigmas.detach().cpu()]
    print(f"Resolved condition: model={args.model} quant={args.quant} device=cuda "
          f"scene={scene_name} seed={args.seed} target_deg={args.target_deg} "
          f"yaw_mag={args.yaw_mag} M={args.M} restamp_offset={args.restamp_offset} "
          f"post_frames={args.post_frames} scheduler_sigmas={sigmas} arms={arms}",
          flush=True)

    results = []
    n_away = None
    for arm in arms:
        print(f"  pan180 {scene_name} seed {args.seed} {arm}", flush=True)
        res = run_arm(eng, seed_x4, args.seed, arm, args,
                      os.path.join(args.out, "tmp_frames", arm), n_away=n_away)
        if n_away is None:
            n_away = res["away_steps"]
            print(f"  locked command pan steps: {n_away} ({res['away_deg']:.1f} deg baseline content)")
        results.append(res)

    videos = []
    for res in results:
        fname = f"{res['arm']}.mp4"
        write_single_mp4(os.path.join(args.out, fname), res, args.fps)
        videos.append({"name": res["arm"], "file": fname})
    write_compare_mp4(os.path.join(args.out, "compare_2x2.mp4"), results, args.fps)
    videos.insert(0, {"name": "compare_2x2", "file": "compare_2x2.mp4"})

    manifest = {
        "condition_source": args.condition_source,
        "run_label": "visual_review",
        "command": command,
        "model": args.model,
        "quant": args.quant,
        "device": "cuda",
        "scene": scene_name,
        "seed": args.seed,
        "target_deg": args.target_deg,
        "locked_away_steps": n_away,
        "scheduler_sigmas": sigmas,
        "arms": arms,
        "results": [
            {k: v for k, v in r.items() if k not in ("video", "start")}
            for r in results
        ],
        "source_hashes": {
            "anchor_pan_video": _sha256(__file__),
            "anchor_probe": _sha256("examples/anchor_probe.py"),
            "anchor": _sha256("examples/anchor.py"),
            "plan": _sha256(args.condition_source),
        },
    }
    pathlib.Path(args.out, "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_index(args.out, manifest, videos)
    cleanup_frames(args.out, args.keep_frames)
    print(f"wrote {args.out}/index.html")
    print(f"wrote {args.out}/manifest.json")
    for v in videos:
        print(f"wrote {args.out}/{v['file']}")


if __name__ == "__main__":
    main()
