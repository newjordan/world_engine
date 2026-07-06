# uv run --dev python examples/atlas_demo.py --out bench_out/atlas/demo
"""
Before/after DEMO of the pose-indexed world atlas (Phase 6.5, lightweight).

Runs one pan-away-and-return trajectory TWICE on the same seed, differing only in the
memory mechanism:
  BEFORE = no memory aid (revisit baseline) -> forgets the scene on return
  AFTER  = pose atlas (best config M2 off4 k1) -> retrieves the near-heading keyframe

Emits a side-by-side mp4 over the whole trajectory (a memory system is temporal, so video
is the honest medium) plus a before/after stills triptych at the return heading where the
atlas advantage is largest. One engine / one compile; both arms share it.
"""
import argparse
import os
import sys

sys.path.insert(0, "examples")
import cv2
import numpy as np
import torch

from permanence_bench import RealEngine, load_seeds, _noop, psnr
from atlas import AtlasStore

FONT = cv2.FONT_HERSHEY_SIMPLEX


def run_arm(eng, seed_x4, seed, arm, K, settle, yaw_mag, M, offset, k):
    """Return (ref_A, video_frames[list rgb], away_by_h{dict}, back_by_h{dict})."""
    H = K // 2
    torch.manual_seed(seed)
    eng.reset()
    eng.append_seed(seed_x4)
    if arm == "atlas":
        eng.attach_atlas(AtlasStore(tau_insert=0.05, n_max=100_000))

    video, ref = [], None
    for spec in _noop(settle):
        four = eng.gen(spec)
        for r in eng.all_rgb(four):
            video.append(("at start", r))
        ref = eng.last_rgb(four)
    if arm == "atlas":
        eng.atlas_capture()

    away_by_h = {}
    for i in range(H):
        four = eng.gen(dict(button=set(), mouse=(+yaw_mag, 0.0)))
        for r in eng.all_rgb(four):
            video.append(("panning away", r))
        away_by_h[round((i + 1) * yaw_mag, 4)] = eng.last_rgb(four)
        if arm == "atlas" and i % M == 0:
            eng.atlas_capture()

    back_by_h = {}
    for b in range(H):
        if arm == "atlas":
            eng.atlas_activate(offset=offset, k=k, align_pose=True)
        four = eng.gen(dict(button=set(), mouse=(-yaw_mag, 0.0)))
        for r in eng.all_rgb(four):
            video.append(("returning", r))
        back_by_h[round((H - 1 - b) * yaw_mag, 4)] = eng.last_rgb(four)
    return ref, video, away_by_h, back_by_h


def rz(img, h=360):
    hh, ww = img.shape[:2]
    return cv2.resize(img, (int(round(ww * h / hh)), h))


def decorate(img, title, tcolor, phase, ref):
    img = rz(img).copy()
    Hh, Ww = img.shape[:2]
    # top title bar
    cv2.rectangle(img, (0, 0), (Ww, 34), (0, 0, 0), -1)
    cv2.putText(img, title, (10, 24), FONT, 0.7, tcolor, 2, cv2.LINE_AA)
    # phase caption bottom-left
    cv2.rectangle(img, (0, Hh - 30), (Ww, Hh), (0, 0, 0), -1)
    cv2.putText(img, phase, (10, Hh - 9), FONT, 0.6, (0, 235, 235), 2, cv2.LINE_AA)
    # "start view" reference inset, top-right
    iw, ih = Ww // 4, (Ww // 4) * ref.shape[0] // ref.shape[1]
    ins = cv2.resize(ref, (iw, ih))
    y0, x0 = 40, Ww - iw - 6
    img[y0:y0 + ih, x0:x0 + iw] = ins
    cv2.rectangle(img, (x0, y0), (x0 + iw, y0 + ih), (255, 255, 255), 1)
    cv2.putText(img, "start", (x0 + 3, y0 + 15), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_mp4(path, frames, fps=30):
    try:
        import imageio
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1) as w:
            for f in frames:
                w.append_data(f)
        return "h264"
    except Exception as e:
        Hh, Ww = frames[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (Ww, Hh))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        return f"mp4v (imageio unavailable: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1236)
    ap.add_argument("--yaw-mag", type=float, default=0.2)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--offset", type=int, default=4)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--assets", default="bench_assets")
    ap.add_argument("--out", default="bench_out/atlas/demo")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    eng = RealEngine(args.model, "cuda", None, max(args.k, 1), pin_all_layers=True)
    seeds = load_seeds(args.scene + 1, args.assets)
    name, seed_x4 = seeds[args.scene]
    print(f"scene={name} K={args.K} atlas(M={args.M},off={args.offset},k={args.k})", flush=True)

    ref_b, vid_b, away_b, back_b = run_arm(eng, seed_x4, args.seed, "revisit",
                                           args.K, args.settle, args.yaw_mag,
                                           args.M, args.offset, args.k)
    print("baseline arm done", flush=True)
    ref_a, vid_a, away_a, back_a = run_arm(eng, seed_x4, args.seed, "atlas",
                                           args.K, args.settle, args.yaw_mag,
                                           args.M, args.offset, args.k)
    print("atlas arm done", flush=True)

    # --- side-by-side video over the full trajectory ---
    gap = np.zeros((360, 12, 3), np.uint8)
    panels = []
    for (ph, fb), (_, fa) in zip(vid_b, vid_a):
        pb = decorate(fb, "BEFORE  -  no memory", (120, 120, 255), ph, ref_b)
        pa = decorate(fa, "AFTER  -  pose atlas", (120, 255, 120), ph, ref_a)
        panels.append(np.concatenate([pb, gap, pa], axis=1))
    mp4 = os.path.join(args.out, "atlas_before_after.mp4")
    codec = write_mp4(mp4, panels, fps=args.fps)
    print(f"wrote {mp4}  ({len(panels)} frames, {codec})", flush=True)

    # --- stills triptych at the return heading of maximum atlas advantage ---
    best_h, best_gain = None, -1e9
    for h in sorted(set(back_b) & set(back_a) & set(away_b)):
        if h <= 0:
            continue
        gt = away_b[h]
        gain = psnr(back_a[h], gt) - psnr(back_b[h], gt)
        if gain > best_gain:
            best_gain, best_h = gain, h
    gt = away_b[best_h]
    trip = np.concatenate([
        decorate(gt, "GROUND TRUTH (outbound)", (255, 255, 255), f"heading {best_h}", ref_b),
        gap, decorate(back_b[best_h], "BEFORE  -  forgot", (120, 120, 255),
                      f"{psnr(back_b[best_h], gt):.1f} dB", ref_b),
        gap, decorate(back_a[best_h], "AFTER  -  remembered", (120, 255, 120),
                      f"{psnr(back_a[best_h], gt):.1f} dB", ref_a),
    ], axis=1)
    trip_png = os.path.join(args.out, "atlas_before_after_stills.png")
    cv2.imwrite(trip_png, cv2.cvtColor(trip, cv2.COLOR_RGB2BGR))

    # --- returned-to-start triptych (heading 0) ---
    ret = np.concatenate([
        decorate(ref_b, "START view", (255, 255, 255), "reference", ref_b),
        gap, decorate(back_b[0.0], "BEFORE returned", (120, 120, 255), "no memory", ref_b),
        gap, decorate(back_a[0.0], "AFTER returned", (120, 255, 120), "pose atlas", ref_a),
    ], axis=1)
    ret_png = os.path.join(args.out, "atlas_returned_to_start.png")
    cv2.imwrite(ret_png, cv2.cvtColor(ret, cv2.COLOR_RGB2BGR))

    print(f"wrote {trip_png}  (max-advantage heading {best_h}, +{best_gain:.2f} dB)")
    print(f"wrote {ret_png}")


if __name__ == "__main__":
    main()
