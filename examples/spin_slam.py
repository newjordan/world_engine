# uv run --dev python examples/spin_slam.py --out bench_out/spin_slam
"""
360-degree spin persistence with SLAM loop closure (spin-persistence branch).

Phase 6.5's spin demo was OPEN-LOOP dead reckoning: the atlas was keyed on the
command-yaw accumulator with a units-per-revolution constant borrowed from a SEPARATE
baseline run. This benchmark closes the loop with examples/slam.py — signed visual
odometry on the rendered frames (the slope integration), appearance-detected loop
closure at the sine-wave re-intersection point (measures the TRUE period in-run, drift
included), and post-closure relocalization (drift cannot re-accumulate on lap 2+).

Arms (identical command stream, ~2.05 revolutions of continuous yaw):
  baseline    no memory — the model hallucinates a new world on lap 2
  atlas_dr    Phase 6.5 control: command-yaw key + cross-run period calibration
  atlas_slam  content-space key from YawSLAM + in-run closure; no cross-run oracle

Scoring is per-heading SELF-consistency: each lap-2 frame is scored (PSNR) against the
same arm's own lap-1 frame at the same content heading (heading = each arm's own signed
VO integration on a shared geometric deg axis, so arms are comparable). "Does the world
that comes back match the world that was there?" Cross-arm scores vs the baseline's
lap 1 are also emitted for continuity with the Phase 6.5 numbers.

Single-trial mode emits media: before/after mp4, a loop-closure 2x2 grid, the sine-phase
figure with the closure point, and a per-frame CSV. --no-media for sweep trials.
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, "examples")
import cv2
import numpy as np
import torch

from atlas import AtlasStore
from atlas_spin import decorate, write_mp4
from permanence_bench import RealEngine, load_seeds, _noop, psnr
from slam import YawSLAM

FONT = cv2.FONT_HERSHEY_SIMPLEX


def gray_of(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def run_arm(eng, seed_x4, seed, arm, args, n_frames=None, dr_period=None):
    """Run one spin arm. Every arm carries a YawSLAM observer (baseline/atlas_dr use it
    passively for the heading axis; atlas_slam uses it as the retrieval key + closure).
    Returns a result dict; n_frames=None (baseline) runs to target revs / cap."""
    torch.manual_seed(seed)
    eng.reset()
    eng.append_seed(seed_x4)

    video, ref = [], None
    for spec in _noop(args.settle):
        four = eng.gen(spec)
        for r in eng.all_rgb(four)[::4 // args.subs]:
            video.append((0.0, 1, r))
        ref = eng.last_rgb(four)

    ppr_geom = ref.shape[1] * (360.0 / args.fov)   # geometric px-per-rev prior
    slam = YawSLAM(ppr_guess=ppr_geom, gain=args.gain)
    slam.observe(gray_of(ref))                     # init odometry on the settle frame

    store = None
    if arm == "atlas_dr":
        store = AtlasStore(tau_insert=0.02, n_max=100_000, yaw_period=dr_period)
        eng.attach_atlas(store)
        eng.atlas_capture()                        # command-space key (default)
    elif arm == "atlas_slam":
        store = AtlasStore(tau_insert=args.tau_px, n_max=100_000, yaw_period=None)
        eng.attach_atlas(store)
        if eng.atlas_capture(yaw=slam.pose):       # content-space key
            slam.note_keyframe(gray_of(ref))

    frames, degs, poses, i, t0 = [], [], [], 0, time.time()
    closure_logged = False
    while True:
        if arm == "atlas_dr":
            eng.atlas_activate(offset=args.offset, k=args.k, align_pose=False)
        elif arm == "atlas_slam":
            if slam.period is not None:
                store.yaw_period = slam.period     # circular retrieval once measured
            # predict step: pose is one frame stale; query where we are ABOUT to be
            eng.atlas_activate(offset=args.offset, k=args.k, align_pose=False,
                               yaw=slam.pose + slam.last_dx)
        four = eng.gen(dict(button=set(), mouse=(+args.yaw_mag, 0.0)))
        rgb = eng.last_rgb(four)
        pose = slam.observe(gray_of(rgb), cmd=args.yaw_mag)
        if arm == "atlas_slam":
            ev = slam.pop_revoke_event()
            if ev is not None:
                # slam pruned its own keyframes; prune the SAME tail from the store
                # (inserted 1:1) and drop the bad period so retrieval goes linear again
                del store.keyframes[ev["kf_len"]:]
                store.yaw_period = None
                closure_logged = False
                print(f"  [{arm}] closure REVOKED at frame {i} "
                      f"({ev['reason']}, bad period {ev['period']:.0f}px); "
                      f"store pruned to {len(store.keyframes)} keyframes", flush=True)
            if slam.closure is not None and not closure_logged:
                closure_logged = True
                print(f"  [{arm}] LOOP CLOSED at frame {i}: "
                      f"period={slam.period:.1f}px (geom prior {ppr_geom:.1f}), "
                      f"ncc={slam.closure['ncc']:.2f} "
                      f"votes={slam.closure['n_votes']}", flush=True)
        if arm != "baseline" and i % args.M == 0:
            if arm == "atlas_slam":
                if eng.atlas_capture(yaw=pose):
                    slam.note_keyframe(gray_of(rgb))
            else:
                eng.atlas_capture()
        deg = pose / ppr_geom * 360.0              # shared geometric heading axis
        lap = 1 + int(abs(deg) // 360)
        for r in eng.all_rgb(four)[::4 // args.subs]:
            video.append((abs(deg) % 360, lap, r))
        frames.append(rgb)
        degs.append(abs(deg))
        poses.append(pose)
        i += 1
        if i % 50 == 0:
            print(f"  [{arm}] {i} frames, {degs[-1]:.0f} deg, "
                  f"{i / (time.time() - t0):.1f} fps", flush=True)
        if (n_frames is not None and i >= n_frames) or \
           (n_frames is None and (degs[-1] >= args.target_revs * 360.0 or i >= args.cap)):
            break

    return dict(arm=arm, ref=ref, video=video, frames=frames, degs=degs, poses=poses,
                slam=slam, slam_stats=slam.stats(),
                store_stats=store.stats() if store else None, ppr_geom=ppr_geom)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def lap2_self_scores(res):
    """[(j, i_match, heading_deg, psnr)] for lap-2 frames vs the SAME arm's lap-1 frame
    at the nearest content heading."""
    degs, frames = res["degs"], res["frames"]
    lap1 = [(i, d) for i, d in enumerate(degs) if d < 360.0]
    rows = []
    for j, d in enumerate(degs):
        if 360.0 <= d < 720.0 and lap1:
            i0 = min(lap1, key=lambda t: abs(t[1] - (d - 360.0)))[0]
            rows.append((j, i0, d - 360.0, psnr(frames[j], frames[i0])))
    return rows


def lap2_cross_scores(res, base):
    """Phase-6.5-style: arm's lap-2 frames vs the BASELINE's lap-1 frame at the same
    heading (continuity with the +1.04 dB open-loop number)."""
    lap1b = [(i, d) for i, d in enumerate(base["degs"]) if d < 360.0]
    rows = []
    for j, d in enumerate(res["degs"]):
        if 360.0 <= d < 720.0 and lap1b:
            i0 = min(lap1b, key=lambda t: abs(t[1] - (d - 360.0)))[0]
            rows.append((j, i0, d - 360.0, psnr(res["frames"][j], base["frames"][i0])))
    return rows


def binned(rows, width=10.0):
    """heading-bin -> mean psnr, for paired cross-arm diffs on a shared axis."""
    out = {}
    for _, _, h, p in rows:
        out.setdefault(int(h // width), []).append(p)
    return {b: float(np.mean(v)) for b, v in out.items()}


def paired_diff(bins_a, bins_b):
    common = sorted(set(bins_a) & set(bins_b))
    if not common:
        return float("nan"), 0
    d = [bins_a[b] - bins_b[b] for b in common]
    return float(np.mean(d)), len(common)


# --------------------------------------------------------------------------- #
# media
# --------------------------------------------------------------------------- #
def emit_video(out, base, slam_res, fps):
    gap = np.zeros((360, 12, 3), np.uint8)
    closure_f = (slam_res["slam"].closure or {}).get("frame")
    panels = []
    for idx, ((db, lb, fb), (da, la, fa)) in enumerate(zip(base["video"],
                                                           slam_res["video"])):
        pb = decorate(fb, "BEFORE  -  no memory", (120, 120, 255),
                      f"{db:5.0f} deg   lap {lb}", base["ref"])
        tag = "AFTER  -  SLAM loop-closure"
        pa = decorate(fa, tag, (120, 255, 120),
                      f"{da:5.0f} deg   lap {la}", slam_res["ref"])
        # banner once the closure has fired (video is subs frames per gen)
        gen_i = max(0, idx - args_settle_video_len(base)) // 2
        if closure_f is not None and gen_i >= closure_f:
            cv2.putText(pa, "LOOP CLOSED", (pa.shape[1] - 230, 62), FONT, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
        panels.append(np.concatenate([pb, gap, pa], axis=1))
    mp4 = os.path.join(out, "spin_slam_before_after.mp4")
    write_mp4(mp4, panels, fps)
    print(f"wrote {mp4}  ({len(panels)} frames)", flush=True)


def args_settle_video_len(res):
    """# of video frames emitted during settle (deg==0 prefix)."""
    n = 0
    for d, _, _ in res["video"]:
        if d == 0.0:
            n += 1
        else:
            break
    return n


def emit_grid(out, base, slam_res):
    """2x2: rows = arm (baseline / atlas_slam), cols = own lap-1 vs lap-2 at the heading
    of maximum SLAM self-consistency advantage. Within-arm comparison: did the SAME
    world come back?"""
    rows_b = {round(r[2]): r for r in lap2_self_scores(base)}
    rows_s = lap2_self_scores(slam_res)
    best = None
    for j, i0, h, p in rows_s:
        pb = rows_b.get(round(h))
        if pb is None:
            continue
        gain = p - pb[3]
        if best is None or gain > best[0]:
            best = (gain, h, (pb[1], pb[0]), (i0, j))
    if best is None:
        print("no overlapping lap-2 headings; grid skipped")
        return
    gain, h, (bi, bj), (si, sj) = best
    gap = np.zeros((360, 12, 3), np.uint8)

    def row(res, i, j, name, col):
        p = psnr(res["frames"][j], res["frames"][i])
        return np.concatenate([
            decorate(res["frames"][i], f"{name}  lap 1", (255, 255, 255),
                     f"{h:.0f} deg first look", res["ref"]), gap,
            decorate(res["frames"][j], f"{name}  lap 2", col,
                     f"{h:.0f} deg again   {p:.1f} dB", res["ref"])], axis=1)

    vgap = np.zeros((12, row(base, bi, bj, "x", (0, 0, 0)).shape[1], 3), np.uint8)
    grid = np.concatenate([row(base, bi, bj, "BEFORE (no memory)", (120, 120, 255)),
                           vgap,
                           row(slam_res, si, sj, "AFTER (SLAM atlas)", (120, 255, 120))],
                          axis=0)
    png = os.path.join(out, "spin_slam_loopclosure_grid.png")
    cv2.imwrite(png, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"wrote {png}  (heading {h:.0f} deg, self-consistency gain +{gain:.2f} dB)")


def emit_phase_figure(out, results):
    """The sine-wave picture: sin(2*pi*pose/P) per arm over frames, closure marked; and
    the lap-2 self-consistency PSNR per heading underneath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slam_res = results["atlas_slam"]
    P = slam_res["slam"].period or slam_res["ppr_geom"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                   gridspec_kw={"height_ratios": [1, 1.3]})
    for arm, col in (("baseline", "#7777ff"), ("atlas_slam", "#22aa55")):
        poses = np.array(results[arm]["poses"])
        ax1.plot(np.sin(2 * np.pi * poses / P), lw=1.2, color=col, label=arm)
    cl = slam_res["slam"].closure
    if cl:
        ax1.axvline(cl["frame"], color="#dd3333", ls="--", lw=1.5)
        ax1.annotate(f"loop closed\nP={cl['period']:.0f}px",
                     (cl["frame"], 0.0), textcoords="offset points", xytext=(8, 10),
                     color="#dd3333", fontsize=9)
    ax1.set_ylabel("sin(2π · pose / P)")
    ax1.set_title("heading phase — the sine-wave interaction point is the loop closure")
    ax1.legend(loc="lower left", fontsize=8)

    for arm, col in (("baseline", "#7777ff"), ("atlas_dr", "#ff9922"),
                     ("atlas_slam", "#22aa55")):
        if arm not in results:
            continue
        rows = lap2_self_scores(results[arm])
        if rows:
            hs = [r[2] for r in rows]
            ps = [r[3] for r in rows]
            ax2.plot(hs, ps, ".", ms=3, alpha=0.35, color=col)
            b = binned(rows)
            xs = sorted(b)
            ax2.plot([x * 10 + 5 for x in xs], [b[x] for x in xs], "-", lw=2,
                     color=col, label=arm)
    ax2.set_xlabel("lap-2 heading (deg)")
    ax2.set_ylabel("PSNR vs own lap-1 (dB)")
    ax2.set_title("lap-2 self-consistency: does the same world come back?")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(out, "spin_slam_phase.png")
    fig.savefig(png, dpi=110)
    print(f"wrote {png}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1236)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--yaw-mag", type=float, default=1.5)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--offset", type=int, default=4)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--gain", type=float, default=0.25)
    ap.add_argument("--tau-px", type=float, default=6.0)
    ap.add_argument("--target-revs", type=float, default=2.05)
    ap.add_argument("--cap", type=int, default=1100)
    ap.add_argument("--subs", type=int, default=2)
    ap.add_argument("--arms", default="baseline,atlas_dr,atlas_slam")
    ap.add_argument("--assets", default="bench_assets")
    ap.add_argument("--out", default="bench_out/spin_slam")
    ap.add_argument("--csv", default=None, help="append per-frame rows here (sweeps)")
    ap.add_argument("--no-media", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = args.arms.split(",")
    assert arms[0] == "baseline", "baseline must run first (sets the frame budget)"

    eng = RealEngine(args.model, "cuda", None, max(args.k, 1), pin_all_layers=True)
    name, seed_x4 = load_seeds(args.scene + 1, args.assets)[args.scene]

    results = {}
    base = run_arm(eng, seed_x4, args.seed, "baseline", args)
    results["baseline"] = base
    n_frames = len(base["frames"])
    lap1_len = next((j + 1 for j, d in enumerate(base["degs"]) if d >= 360.0), n_frames)
    dr_period = lap1_len * args.yaw_mag        # Phase 6.5 cross-run calibration
    print(f"scene={name} seed={args.seed}: baseline {n_frames} frames, "
          f"{base['degs'][-1]:.0f} deg, lap1={lap1_len} frames "
          f"(atlas_dr period={dr_period:.1f} cmd-units)", flush=True)

    for arm in arms[1:]:
        results[arm] = run_arm(eng, seed_x4, args.seed, arm, args,
                               n_frames=n_frames, dr_period=dr_period)

    # ---------------- summary ----------------
    base_bins = binned(lap2_self_scores(base))
    summary = dict(scene=name, seed=args.seed, n_frames=n_frames, lap1_len=lap1_len)
    print("\n=== lap-2 self-consistency (PSNR vs own lap-1, dB) ===")
    for arm in arms:
        rows = lap2_self_scores(results[arm])
        mean = float(np.mean([r[3] for r in rows])) if rows else float("nan")
        dmean, nb = paired_diff(binned(rows), base_bins)
        summary[f"{arm}_self_mean"] = mean
        summary[f"{arm}_vs_base"] = dmean
        print(f"  {arm:12s} mean {mean:6.2f}   vs baseline {dmean:+.2f} dB "
              f"({nb} heading bins, n={len(rows)} frames)")
    print("\n=== lap-2 cross-consistency vs baseline lap-1 (Phase-6.5 axis) ===")
    for arm in arms:
        rows = lap2_cross_scores(results[arm], base)
        mean = float(np.mean([r[3] for r in rows])) if rows else float("nan")
        summary[f"{arm}_cross_mean"] = mean
        print(f"  {arm:12s} mean {mean:6.2f} dB (n={len(rows)})")

    if "atlas_slam" in results:
        st = results["atlas_slam"]["slam_stats"]
        summary["slam"] = st
        summary["store"] = results["atlas_slam"]["store_stats"]
        print(f"\nSLAM: {json.dumps(st, default=float)}")
        print(f"store: {results['atlas_slam']['store_stats']}")
        if st["period_px"] and st["slope_px_per_cmd"]:
            p_cmd = st["period_px"] / st["slope_px_per_cmd"]
            print(f"measured period in cmd units: {p_cmd:.1f} "
                  f"(atlas_dr borrowed {dr_period:.1f} — in-run vs cross-run calib)")
            summary["slam_period_cmd"] = p_cmd
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["scene", "seed", "arm", "frame", "heading_deg",
                            "psnr_self", "psnr_cross"])
            for arm in arms:
                cross = {j: p for j, _, _, p in lap2_cross_scores(results[arm], base)}
                for j, _, h, p in lap2_self_scores(results[arm]):
                    w.writerow([name, args.seed, arm, j, f"{h:.1f}",
                                f"{p:.3f}", f"{cross.get(j, float('nan')):.3f}"])
        print(f"appended CSV rows -> {args.csv}")

    if not args.no_media:
        if "atlas_slam" in results:
            emit_video(args.out, base, results["atlas_slam"], args.fps)
            emit_grid(args.out, base, results["atlas_slam"])
        emit_phase_figure(args.out, results)


if __name__ == "__main__":
    main()
