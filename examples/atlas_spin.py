# uv run --dev python examples/atlas_spin.py --out bench_out/atlas/spin
"""
Full-rotation SPIN demo of the pose-indexed atlas with CIRCULAR loop closure (Phase 6.5).

The camera spins continuously in one direction for ~two full revolutions:
  lap 1 = first look (both arms identical; the atlas captures keyframes all the way round)
  lap 2 = revisit  (BEFORE hallucinates a fresh world; AFTER retrieves lap-1 keyframes by
                    CIRCULAR yaw and reconstructs the same world)

"One revolution" is defined operationally by CONTENT, not a fragile degrees/frame guess:
we integrate the per-frame horizontal phase-correlation shift and call it a full turn when
the scene has scrolled by 360/FoV screen-widths. The baseline spin sets the frame budget and
the per-lap length (-> AtlasStore yaw_period); the atlas spin replays the identical trajectory.

Emits a side-by-side mp4 (continuous rotation, running degree + lap readout) and a
loop-closure stills triptych: lap-1 first look vs lap-2 forgot vs lap-2 remembered.
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


def pc_shift(prev, gray):
    """Robust per-frame horizontal shift (px) via windowed phase correlation on a
    textured central ROI (excludes static HUD/sky)."""
    Hh, Ww = gray.shape
    r = (slice(int(0.28 * Hh), int(0.64 * Hh)), slice(int(0.15 * Ww), int(0.85 * Ww)))
    a, b = np.float32(prev[r]), np.float32(gray[r])
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, _), _ = cv2.phaseCorrelate(a, b, win)
    return abs(dx)


def spin(eng, seed_x4, seed, arm, settle, yaw_mag, fov, M, offset, k,
         yaw_period=None, n_frames=None, target_revs=2.0, cap=320, subs=2):
    """Spin one direction. If n_frames is None, run until target_revs (or cap); else run
    exactly n_frames. Returns (ref, video[(deg,lap,rgb)], last[rgb/gen], cum_deg[])."""
    torch.manual_seed(seed)
    eng.reset()
    eng.append_seed(seed_x4)
    if arm == "atlas":
        eng.attach_atlas(AtlasStore(tau_insert=0.02, n_max=100_000, yaw_period=yaw_period))

    video, ref, prev = [], None, None
    for spec in _noop(settle):
        four = eng.gen(spec)
        for r in eng.all_rgb(four)[::4 // subs]:
            video.append((0.0, 1, r))
        ref = eng.last_rgb(four)
        prev = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    if arm == "atlas":
        eng.atlas_capture()
    ppr = prev.shape[1] * (360.0 / fov)  # px of horizontal scroll == one full turn

    last, cum_deg, cum_px, i = [], [], 0.0, 0
    while True:
        if arm == "atlas":
            eng.atlas_activate(offset=offset, k=k, align_pose=False)  # temporal restamp only
        four = eng.gen(dict(button=set(), mouse=(+yaw_mag, 0.0)))
        g = cv2.cvtColor(eng.last_rgb(four), cv2.COLOR_RGB2GRAY)
        cum_px += pc_shift(prev, g)
        prev = g
        deg = cum_px / ppr * 360.0
        lap = 1 + int(deg // 360)
        for r in eng.all_rgb(four)[::4 // subs]:
            video.append((deg % 360, lap, r))
        last.append(eng.last_rgb(four))
        cum_deg.append(deg)
        if arm == "atlas" and i % M == 0:
            eng.atlas_capture()
        i += 1
        if (n_frames is not None and i >= n_frames) or \
           (n_frames is None and (deg >= target_revs * 360.0 or i >= cap)):
            break
    return ref, video, last, cum_deg


def rz(img, h=360):
    hh, ww = img.shape[:2]
    return cv2.resize(img, (int(round(ww * h / hh)), h))


def decorate(img, title, tcolor, readout, ref):
    img = rz(img).copy()
    Hh, Ww = img.shape[:2]
    cv2.rectangle(img, (0, 0), (Ww, 34), (0, 0, 0), -1)
    cv2.putText(img, title, (10, 24), FONT, 0.7, tcolor, 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, Hh - 30), (Ww, Hh), (0, 0, 0), -1)
    cv2.putText(img, readout, (10, Hh - 9), FONT, 0.6, (0, 235, 235), 2, cv2.LINE_AA)
    iw = Ww // 4
    ih = iw * ref.shape[0] // ref.shape[1]
    y0, x0 = 40, Ww - iw - 6
    img[y0:y0 + ih, x0:x0 + iw] = cv2.resize(ref, (iw, ih))
    cv2.rectangle(img, (x0, y0), (x0 + iw, y0 + ih), (255, 255, 255), 1)
    cv2.putText(img, "start", (x0 + 3, y0 + 15), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_mp4(path, frames, fps):
    Hh, Ww = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (Ww, Hh))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1236)
    ap.add_argument("--yaw-mag", type=float, default=1.0)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--offset", type=int, default=4)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--cap", type=int, default=340)
    ap.add_argument("--subs", type=int, default=2, help="decoded subframes/gen kept in video")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--assets", default="bench_assets")
    ap.add_argument("--out", default="bench_out/atlas/spin")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    eng = RealEngine(args.model, "cuda", None, max(args.k, 1), pin_all_layers=True)
    name, seed_x4 = load_seeds(args.scene + 1, args.assets)[args.scene]

    # Baseline sets the frame budget + per-lap length (integrated content revolutions).
    ref_b, vid_b, last_b, deg_b = spin(eng, seed_x4, args.seed, "revisit", args.settle,
                                       args.yaw_mag, args.fov, args.M, args.offset, args.k,
                                       target_revs=2.0, cap=args.cap, subs=args.subs)
    n_frames = len(last_b)
    lap1_len = next((j + 1 for j, d in enumerate(deg_b) if d >= 360.0), n_frames)
    yaw_period = lap1_len * args.yaw_mag
    print(f"scene={name}  yaw_mag={args.yaw_mag}  spun {n_frames} frames, total "
          f"{deg_b[-1]:.0f} deg, lap1={lap1_len} frames (yaw_period={yaw_period:.1f})",
          flush=True)

    ref_a, vid_a, last_a, deg_a = spin(eng, seed_x4, args.seed, "atlas", args.settle,
                                       args.yaw_mag, args.fov, args.M, args.offset, args.k,
                                       yaw_period=yaw_period, n_frames=n_frames, subs=args.subs)
    print("both arms done", flush=True)

    gap = np.zeros((360, 12, 3), np.uint8)
    panels = []
    for (deg, lap, fb), (_, _, fa) in zip(vid_b, vid_a):
        ro = f"{deg:5.0f} deg   lap {lap}"
        pb = decorate(fb, "BEFORE  -  no memory", (120, 120, 255), ro, ref_b)
        pa = decorate(fa, "AFTER  -  atlas loop-closure", (120, 255, 120), ro, ref_a)
        panels.append(np.concatenate([pb, gap, pa], axis=1))
    mp4 = os.path.join(args.out, "atlas_spin_before_after.mp4")
    write_mp4(mp4, panels, args.fps)
    print(f"wrote {mp4}  ({len(panels)} frames)", flush=True)

    # Loop-closure stills: for each lap-2 frame, match the lap-1 frame at the same heading
    # (nearest integrated degree) and pick the heading of maximum atlas advantage.
    lap1 = [(j, deg_b[j]) for j in range(lap1_len)]
    best = None
    for j in range(lap1_len, n_frames):
        tgt = deg_b[j] - 360.0
        i0 = min(lap1, key=lambda t: abs(t[1] - tgt))[0]
        gt = last_b[i0]
        gain = psnr(last_a[j], gt) - psnr(last_b[j], gt)
        if best is None or gain > best[0]:
            best = (gain, i0, j, gt)
    if best:
        gain, i0, j, gt = best
        deg = int(deg_b[i0]) % 360
        trip = np.concatenate([
            decorate(gt, "LAP 1  first look", (255, 255, 255), f"{deg} deg", ref_b), gap,
            decorate(last_b[j], "LAP 2  BEFORE  forgot", (120, 120, 255),
                     f"{psnr(last_b[j], gt):.1f} dB", ref_b), gap,
            decorate(last_a[j], "LAP 2  AFTER  remembered", (120, 255, 120),
                     f"{psnr(last_a[j], gt):.1f} dB", ref_a),
        ], axis=1)
        png = os.path.join(args.out, "atlas_spin_loopclosure.png")
        cv2.imwrite(png, cv2.cvtColor(trip, cv2.COLOR_RGB2BGR))
        print(f"wrote {png}  (heading {deg} deg, +{gain:.2f} dB)")


if __name__ == "__main__":
    main()
