# uv run --dev pytest examples/test_slam.py -v
"""
CPU tests for YawSLAM (examples/slam.py) on a synthetic wrap-around panorama.

The world is a periodic texture strip of width P_TRUE px; the "camera" is a sliding
window. Frames are exactly what the estimator sees in the real system (decoded gray
frames), so this validates the entire estimator — signed odometry, appearance loop
closure, period measurement, relocalization — without a model or GPU, mirroring how
test_atlas.py validates retrieval geometry.
"""
import sys

sys.path.insert(0, "examples")
import cv2
import numpy as np
import pytest

from atlas import yaw_dist
from slam import YawSLAM, wrap

P_TRUE = 2600          # px per revolution (ground truth)
H, W_VIEW = 120, 640   # frame size
RATE = 7               # px scrolled per frame (integer => exact ground truth)


def make_world(seed=0, noise=2.0):
    """Returns view(pose)->uint8 frame: periodic panorama sampled at integer pose."""
    rng = np.random.default_rng(seed)
    tex = rng.random((H, P_TRUE)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), 3)
    tex = cv2.normalize(tex, None, 40, 215, cv2.NORM_MINMAX)
    frame_rng = np.random.default_rng(seed + 1)

    def view(pose: float) -> np.ndarray:
        rolled = np.roll(tex, int(round(pose)), axis=1)  # +pose => content moves right
        f = rolled[:, :W_VIEW] + frame_rng.normal(0.0, noise, (H, W_VIEW))
        return np.clip(f, 0, 255).astype(np.uint8)

    return view


def run_spin(slam, view, n_frames, rate=RATE, kf_every=2, inject=None):
    """Drive the estimator for n_frames; note a keyframe every kf_every frames.
    inject={frame: px_error} perturbs slam.pose (simulated accumulated drift).
    Returns (true_poses, est_poses) aligned per observe() call."""
    true, est = [], []
    pose_true = 0.0
    for t in range(n_frames):
        p = slam.observe(view(pose_true))
        if inject and t in inject:
            slam.pose += inject[t]
            p = slam.pose
        if t % kf_every == 0:
            slam.note_keyframe(view(pose_true))
        true.append(pose_true)
        est.append(p)
        pose_true += rate
    return np.array(true), np.array(est)


# --------------------------------------------------------------------------- #
# odometry (the slope integration)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rate", [+RATE, -RATE])
def test_signed_odometry_tracks_scroll(rate):
    slam = YawSLAM(ppr_guess=W_VIEW * 4.0)
    true, est = run_spin(slam, make_world(), n_frames=150, rate=rate)
    # first observe() only initializes; afterwards pose integrates the signed shift
    err = est[-1] - true[-1]
    assert abs(err) < 0.01 * abs(true[-1]), (est[-1], true[-1])
    assert np.sign(est[-1]) == np.sign(rate)
    assert slam.period is None  # half a lap: closure must NOT fire


def test_slope_diagnostic():
    slam = YawSLAM(ppr_guess=W_VIEW * 4.0)
    view = make_world()
    pose_true = 0.0
    for _ in range(100):
        slam.observe(view(pose_true), cmd=1.5)   # constant command, like the spin demo
        pose_true += RATE
    s = slam.slope()
    assert s is not None and abs(s - RATE / 1.5) < 0.05 * (RATE / 1.5)


# --------------------------------------------------------------------------- #
# loop closure (the sine-wave re-intersection)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("guess_scale", [0.85, 1.0, 1.20])
@pytest.mark.parametrize("rate", [+RATE, -RATE])
def test_closure_measures_true_period(rate, guess_scale):
    """Closure fires shortly after one revolution and measures P within 1.5%, even when
    the geometric prior is 15-20% wrong — the prior only gates candidacy."""
    slam = YawSLAM(ppr_guess=P_TRUE * guess_scale)
    n = int(1.4 * P_TRUE / RATE)
    run_spin(slam, make_world(), n_frames=n, rate=rate)
    assert slam.closure is not None, "closure never fired"
    assert abs(slam.period - P_TRUE) < 0.015 * P_TRUE, slam.period
    lap_frames = P_TRUE / RATE
    assert slam.closure["frame"] <= 1.15 * lap_frames, slam.closure


def test_no_false_closure_before_revisit():
    """Random panorama: mid-lap views must not fire closure (appearance gate)."""
    slam = YawSLAM(ppr_guess=P_TRUE * 0.5)   # generous gate: candidates from ~0.3 lap
    n = int(0.9 * P_TRUE / RATE)             # never actually revisits
    run_spin(slam, make_world(), n_frames=n)
    assert slam.closure is None
    assert slam.period is None


def test_no_closure_without_keyframes():
    slam = YawSLAM(ppr_guess=P_TRUE)
    view = make_world()
    pose_true = 0.0
    for _ in range(int(1.3 * P_TRUE / RATE)):
        slam.observe(view(pose_true))
        pose_true += RATE
    assert slam.closure is None


# --------------------------------------------------------------------------- #
# relocalization (drift control on lap 2+)
# --------------------------------------------------------------------------- #
# NOTE the estimator converges to MAP consistency, not metric ground truth: keyframe
# poses carry the (tiny, integrated) odometry bias, and reloc pulls the estimate toward
# the MAP. That is the correct target — the atlas needs the right KEYFRAME retrieved,
# not ground-truth pixels — so these tests score retrieval CONTENT error: does circular
# retrieval at the current estimate fetch a keyframe showing the current true heading?
def content_retrieval_err(slam, est, true, i, kf_every, max_seq=None):
    kfs = slam.keyframes if max_seq is None else \
        [kf for kf in slam.keyframes if kf.seq < max_seq]
    near = min(kfs, key=lambda kf: yaw_dist(kf.pose, est[i], slam.period))
    kf_true = near.seq * kf_every * RATE      # ground-truth pose when kf was noted
    return abs(wrap(kf_true - true[i], P_TRUE))


def test_reloc_corrects_injected_drift():
    n = int(1.9 * P_TRUE / RATE)
    inject_at = int(1.4 * P_TRUE / RATE)
    kf_every = 2
    spacing = kf_every * RATE

    # score against the LAP-1 map only: keyframes noted after the injection sit at
    # injected coordinates with correct content, so they mask the drift being tested
    lap1_kfs = int(np.ceil((P_TRUE / RATE) / kf_every))

    def final_content_err(gain):
        slam = YawSLAM(ppr_guess=P_TRUE * 0.98, gain=gain)
        true, est = run_spin(slam, make_world(), n_frames=n, kf_every=kf_every,
                             inject={inject_at: +40.0})
        assert slam.closure is not None
        return content_retrieval_err(slam, est, true, n - 1, kf_every,
                                     max_seq=lap1_kfs)

    open_loop = final_content_err(gain=0.0)
    closed = final_content_err(gain=0.25)
    assert open_loop > 25.0, open_loop            # error persists without feedback
    assert closed <= spacing + 3.0, closed        # feedback re-locks retrieval


def test_reloc_keeps_map_consistency_over_laps():
    """With feedback on, retrieval stays content-correct through lap 3, and the
    corrections stay small (the filter is not fighting the map)."""
    kf_every = 2
    spacing = kf_every * RATE
    slam = YawSLAM(ppr_guess=P_TRUE * 0.98, gain=0.25)
    n = int(2.6 * P_TRUE / RATE)
    true, est = run_spin(slam, make_world(), n_frames=n, kf_every=kf_every)
    assert slam.closure is not None
    lap23 = [i for i in range(n) if true[i] > 1.1 * P_TRUE]
    for i in lap23[:: len(lap23) // 10]:
        assert content_retrieval_err(slam, est, true, i, kf_every) <= spacing + 3.0, i
    assert slam.n_reloc > 50
    assert slam.reloc_abs_max < 10.0, slam.reloc_abs_max


def test_aliased_repeat_is_revoked_then_true_closure_found():
    """Perceptual aliasing: a 1000px content segment repeated at a 2400px offset fires
    a false closure that is internally consistent while the copy lasts. Verification
    must catch it once the copy runs out (reloc evidence dries up), revoke it (undoing
    corrections, pruning contaminated keyframes), and the TRUE closure must still be
    found afterward. This is the failure mode observed in the real generative world."""
    P = 4000
    rng = np.random.default_rng(3)
    tex = rng.random((H, P)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), 3)
    tex = cv2.normalize(tex, None, 40, 215, cv2.NORM_MINMAX)
    tex[:, 2500:3500] = tex[:, 100:1100]      # the alias, offset 2400 >= closure gate
    frame_rng = np.random.default_rng(4)

    def view(pose):
        rolled = np.roll(tex, int(round(pose)), axis=1)
        f = rolled[:, :W_VIEW] + frame_rng.normal(0.0, 2.0, (H, W_VIEW))
        return np.clip(f, 0, 255).astype(np.uint8)

    slam = YawSLAM(ppr_guess=P * 0.98, gain=0.25)
    n = int(1.5 * P / RATE)
    run_spin(slam, view, n_frames=n)
    # end state is what matters: the TRUE period, any alias revoked along the way
    assert slam.closure is not None, slam.revoke_log
    assert abs(slam.period - P) < 0.015 * P, (slam.period, slam.revoke_log)


# --------------------------------------------------------------------------- #
# end-to-end keying: SLAM pose + measured period + AtlasStore circular distance
# --------------------------------------------------------------------------- #
def test_wrap_retrieval_finds_same_content_keyframe():
    """After closure, the nearest keyframe by CIRCULAR distance at a lap-2 pose is one
    whose lap-1 content offset is under the keyframe spacing — i.e. the atlas would
    page in the right view. This is the sine-wave interaction point doing real work."""
    slam = YawSLAM(ppr_guess=P_TRUE * 0.98)
    kf_every = 2
    n = int(1.5 * P_TRUE / RATE)
    true, est = run_spin(slam, make_world(), n_frames=n, kf_every=kf_every)
    assert slam.closure is not None

    spacing = kf_every * RATE
    lap2 = [i for i in range(n) if true[i] > 1.1 * P_TRUE]
    for i in lap2[:: len(lap2) // 8]:
        assert content_retrieval_err(slam, est, true, i, kf_every) <= spacing + 3.0, i


def test_wrap_helper():
    assert wrap(10.0, 100.0) == 10.0
    assert wrap(60.0, 100.0) == -40.0
    assert wrap(-60.0, 100.0) == 40.0
    assert abs(wrap(350.0, 100.0) - (-50.0)) < 1e-9
