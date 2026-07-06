# uv run --dev pytest examples/test_slam.py -v
"""
1-DOF visual SLAM for full-revolution loop closure (spin persistence).

Phase 6.5's spin demo was OPEN-LOOP: it integrated |phase-correlation shift| from a
*separate baseline run* to guess units-per-revolution, then keyed the atlas on the
command-space yaw accumulator. That works only as well as the command->content scale
holds across runs — and it is unsigned, uncorrected, and never verified against what the
model actually rendered.

This module closes the loop, in content space, within the run itself:

  ODOMETRY   the SLOPE integration: per-frame SIGNED horizontal phase-correlation shift
             of the rendered frames, integrated into a continuous scroll pose (px). This
             measures the world the model actually drew, not the commands we sent.
  CLOSURE    heading is a PHASE on a circle; after ~one revolution the phase re-crosses
             its starting point (the sine-wave re-intersection). We detect that moment by
             APPEARANCE — the current view fingerprint matches a keyframe recorded far
             away in odometry — and measure the true period P there, drift included.
  RELOC      after closure, every confident appearance match feeds back a small pose
             correction (complementary filter), so drift cannot re-accumulate on lap 2+.

The pose this class maintains is the retrieval key for the pose-indexed atlas
(`examples/atlas.py`): continuous px pose + measured period -> AtlasStore's circular
yaw distance retrieves the lap-1 keyframe for the same CONTENT heading, indefinitely.

Model-agnostic and GPU-free (numpy + cv2 on decoded frames only), so the whole estimator
is testable on a synthetic panorama without a model (`examples/test_slam.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def roi_of(gray: np.ndarray) -> np.ndarray:
    """Textured central band used for all correlation (excludes HUD/sky/ground blur).
    Same crop as the Phase-6.5 spin demo, kept identical so numbers stay comparable."""
    H, W = gray.shape
    return np.float32(gray[int(0.28 * H):int(0.64 * H), int(0.15 * W):int(0.85 * W)])


def signed_shift(prev_roi: np.ndarray, cur_roi: np.ndarray, win: np.ndarray) -> float:
    """SIGNED horizontal scroll (px) between consecutive ROIs via windowed phase
    correlation. Convention (pinned empirically): content moving right => positive.
    Phase 6.5 used abs() here; the sign is what lets pose be a real integrator."""
    (dx, _dy), _resp = cv2.phaseCorrelate(prev_roi, cur_roi, win)
    return float(dx)


def wrap(d: float, period: float) -> float:
    """Signed circular residual of d into [-period/2, +period/2)."""
    return (d + period / 2.0) % period - period / 2.0


def _fingerprint(roi: np.ndarray, size) -> np.ndarray:
    """Normalized (zero-mean, unit-norm) thumbnail for appearance matching (NCC)."""
    fp = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
    fp -= fp.mean()
    n = float(np.linalg.norm(fp))
    return fp / n if n > 0 else fp


@dataclass
class SlamKeyframe:
    """Appearance record of one atlas keyframe: where it was (odometry pose at capture)
    and what it looked like (fingerprint + small ROI for sub-px refinement)."""
    pose: float
    fp: np.ndarray
    roi: np.ndarray       # downscaled ROI (refine_w wide), f32
    seq: int


class YawSLAM:
    """Signed visual odometry + appearance loop closure + relocalization, 1 DOF (yaw).

    Usage per generated frame (see examples/spin_slam.py):
        pose = slam.observe(gray_frame)        # AFTER gen: integrate + correct
        if captured_into_atlas:
            slam.note_keyframe(gray_frame)     # keep SLAM keyframes == atlas keyframes
        # `pose` keys atlas_capture/atlas_activate; once slam.period is set, give it
        # to AtlasStore.yaw_period so retrieval distance becomes circular.

    Parameters
    ----------
    ppr_guess       geometric prior for px-per-revolution (frame_width * 360/FoV). Used
                    ONLY to gate closure candidacy (a match must be at least
                    min_loop_frac * ppr_guess away in odometry); the measured period
                    replaces it entirely once closure fires.
    tau_close       min NCC for a closure match (pre-period, strict).
    confirm         closure VOTES required: independent per-frame matches whose implied
                    periods agree within vote_tol, inside a vote_window-frame window.
                    A generative world re-renders similar content at different headings
                    (perceptual aliasing), so single-match or short-streak evidence
                    false-fires; a consensus of many consistent measurements is the
                    standard SLAM defense. Bumped by +2 after every revocation.
    tau_reloc       min NCC for a relocalization match (post-period, looser).
    gain            complementary-filter gain on the reloc residual (0 = open loop).

    A fired closure is provisional for verify_horizon frames: relocalization evidence
    must keep flowing (>= min_evidence matches per evidence_window frames) and stay
    consistent (median |residual| of the last 10 events <= revoke_frac * period). A
    closure that fails either check is REVOKED — corrections undone, keyframes noted
    since the closure pruned, search resumed with a stricter vote requirement. This is
    what catches an aliased closure that internally tracks for a while (e.g. a repeated
    corridor) and only betrays itself once the copy runs out.
    """

    def __init__(self, ppr_guess: float, min_loop_frac: float = 0.6,
                 tau_close: float = 0.60, confirm: int = 5,
                 tau_reloc: float = 0.50, gain: float = 0.25,
                 fp_size=(48, 20), refine_w: int = 192,
                 max_refine_frac: float = 0.35, reloc_window_frac: float = 0.12,
                 min_resp: float = 0.05, vote_window: int = 60,
                 vote_tol: Optional[float] = None, verify_horizon: int = 240,
                 verify_grace: int = 15, evidence_window: int = 60,
                 min_evidence: int = 3, revoke_frac: float = 0.03):
        assert ppr_guess > 0, ppr_guess
        self.ppr_guess = float(ppr_guess)
        self.min_loop_frac = min_loop_frac
        self.tau_close = tau_close
        self.confirm = max(1, int(confirm))
        self.tau_reloc = tau_reloc
        self.gain = gain
        self.fp_size = fp_size
        self.refine_w = refine_w
        self.max_refine_frac = max_refine_frac
        self.reloc_window_frac = reloc_window_frac
        self.min_resp = min_resp
        self.vote_window = vote_window
        self.vote_tol = vote_tol if vote_tol is not None else 0.012 * ppr_guess
        self.verify_horizon = verify_horizon
        self.verify_grace = verify_grace
        self.evidence_window = evidence_window
        self.min_evidence = min_evidence
        self.revoke_frac = revoke_frac

        self.pose = 0.0                      # continuous signed content scroll (px)
        self.period: Optional[float] = None  # |px per revolution|, measured at closure
        self.keyframes: List[SlamKeyframe] = []
        self.frame = 0
        self.cum_cmd = 0.0                   # optional command integral (slope diag)

        self._prev_roi: Optional[np.ndarray] = None
        self._win: Optional[np.ndarray] = None
        self._roi_full_w: Optional[int] = None
        self._votes: List[tuple] = []        # (frame, signed implied period)
        # provisional-closure verification state
        self._close_frame: Optional[int] = None
        self._kf_len_at_close: Optional[int] = None
        self._reloc_events: List[tuple] = [] # (frame, |resid|) since closure
        self._corr_sum = 0.0                 # summed applied corrections since closure
        self._revoke_event = None            # for pop_revoke_event()

        # diagnostics (no silent behavior: everything observable in stats())
        self.closure = None                  # dict once fired
        self.n_revoked = 0
        self.revoke_log: List[dict] = []
        self.n_reloc = 0
        self.reloc_abs_sum = 0.0
        self.reloc_abs_max = 0.0
        self.last_dx = 0.0

    # ------------------------------------------------------------------ core
    def observe(self, gray: np.ndarray, cmd: float = 0.0) -> float:
        """Integrate one rendered frame into the pose estimate; attempt loop closure
        (pre-period) or relocalization (post-period). Returns the corrected pose."""
        roi = roi_of(gray)
        if self._win is None:
            self._win = cv2.createHanningWindow((roi.shape[1], roi.shape[0]), cv2.CV_32F)
            self._roi_full_w = roi.shape[1]
        if self._prev_roi is not None:
            self.last_dx = signed_shift(self._prev_roi, roi, self._win)
            self.pose += self.last_dx
            self.frame += 1
            self.cum_cmd += cmd
            if self.period is None:
                self._try_close(roi)
            else:
                self._try_reloc(roi)
                self._verify()
        self._prev_roi = roi
        return self.pose

    def note_keyframe(self, gray: np.ndarray) -> None:
        """Record the current view as a keyframe at the current pose. Call exactly when
        a frame is captured into the atlas, so appearance records mirror the store."""
        roi = roi_of(gray)
        small = cv2.resize(roi, (self.refine_w,
                                 max(8, int(roi.shape[0] * self.refine_w / roi.shape[1]))),
                           interpolation=cv2.INTER_AREA)
        self.keyframes.append(SlamKeyframe(
            pose=self.pose, fp=_fingerprint(roi, self.fp_size),
            roi=small, seq=len(self.keyframes)))

    # ------------------------------------------------------- appearance matching
    def _refine(self, kf: SlamKeyframe, roi: np.ndarray) -> Optional[tuple]:
        """(sub-px content offset current-vs-keyframe in full-res px, pc response):
        current content equals kf content scrolled right by delta. None if unreliable
        (offset out of range, or correlation peak too weak to trust)."""
        cur = cv2.resize(roi, (kf.roi.shape[1], kf.roi.shape[0]),
                         interpolation=cv2.INTER_AREA)
        win = cv2.createHanningWindow((kf.roi.shape[1], kf.roi.shape[0]), cv2.CV_32F)
        (dx, _), resp = cv2.phaseCorrelate(kf.roi, cur, win)
        delta = float(dx) * (self._roi_full_w / kf.roi.shape[1])
        if abs(delta) > self.max_refine_frac * self._roi_full_w or resp < self.min_resp:
            return None
        return delta, float(resp)

    def _best_match(self, roi: np.ndarray, cands: List[SlamKeyframe]):
        if not cands:
            return None, -1.0
        fp = _fingerprint(roi, self.fp_size)
        nccs = [float((fp * kf.fp).sum()) for kf in cands]
        i = int(np.argmax(nccs))
        return cands[i], nccs[i]

    def _try_close(self, roi: np.ndarray) -> None:
        """Collect closure VOTES — far-in-odometry appearance matches, each implying a
        period — and fire when `confirm` votes inside the rolling window agree within
        vote_tol. A single confident match is not evidence in a generative world
        (perceptual aliasing); a stream of independent measurements agreeing on one
        period is."""
        min_gap = self.min_loop_frac * self.ppr_guess
        cands = [kf for kf in self.keyframes if abs(self.pose - kf.pose) >= min_gap]
        kf, ncc = self._best_match(roi, cands)
        if kf is None or ncc < self.tau_close:
            return
        ref = self._refine(kf, roi)
        if ref is None:
            return
        delta, resp = ref
        z = kf.pose + delta                  # measured content pose (previous lap view)
        self._votes.append((self.frame, self.pose - z))
        self._votes = [(f, p) for f, p in self._votes
                       if self.frame - f <= self.vote_window]
        ps = np.array([p for _, p in self._votes])
        med = float(np.median(ps))
        good = ps[np.abs(ps - med) <= self.vote_tol]
        if len(good) >= self.confirm:
            self.period = float(abs(good.mean()))    # px/revolution, drift included
            self.closure = dict(frame=self.frame, period=self.period, ncc=ncc,
                                resp=resp, n_votes=int(len(good)),
                                vote_spread=float(good.std()),
                                kf_seq=kf.seq, kf_pose=kf.pose,
                                calib_err_px=self.period - self.ppr_guess)
            self._votes = []
            self._close_frame = self.frame
            self._kf_len_at_close = len(self.keyframes)
            self._reloc_events, self._corr_sum = [], 0.0

    def _try_reloc(self, roi: np.ndarray) -> None:
        """Post-closure drift control: match within a circular window around the current
        estimate; nudge pose toward the measurement by `gain`."""
        if self.gain <= 0.0:
            return
        W = self.reloc_window_frac * self.period
        # Evidence must come from a PREVIOUS lap: a keyframe recorded near the current
        # pose on the same lap carries the current estimate's own error, so matching it
        # is self-referential feedback that locks drift in rather than correcting it.
        # Linear (continuous-pose) gap >= min_loop_frac * period excludes same-lap keys.
        min_gap = self.min_loop_frac * self.period
        cands = [kf for kf in self.keyframes
                 if abs(self.pose - kf.pose) >= min_gap
                 and abs(wrap(kf.pose - self.pose, self.period)) <= W]
        kf, ncc = self._best_match(roi, cands)
        if kf is None or ncc < self.tau_reloc:
            return
        ref = self._refine(kf, roi)
        if ref is None:
            return
        delta, _resp = ref
        resid = wrap((kf.pose + delta) - self.pose, self.period)
        if abs(resid) > W:                   # refined jump outside the trust window
            return
        self.pose += self.gain * resid
        self._corr_sum += self.gain * resid
        self._reloc_events.append((self.frame, abs(resid)))
        self.n_reloc += 1
        self.reloc_abs_sum += abs(resid)
        self.reloc_abs_max = max(self.reloc_abs_max, abs(resid))

    def _verify(self) -> None:
        """While a closure is young (<= verify_horizon frames), demand that reloc
        evidence keeps flowing and stays consistent; otherwise revoke it. An aliased
        closure looks perfect while the repeated content lasts — the tell is what
        happens after: matches dry up (the copy ran out) or residuals blow up (the
        period is wrong). A closure that survives the horizon is final. Requires the
        reloc feedback to be enabled (gain > 0) — evidence cannot flow without it."""
        if self._close_frame is None or self.gain <= 0.0:
            return
        age = self.frame - self._close_frame
        if age < self.verify_grace:
            return
        if age > self.verify_horizon:
            self._close_frame = None         # verified: closure is now permanent
            return
        recent = [e for e in self._reloc_events
                  if self.frame - e[0] <= self.evidence_window]
        if len(recent) < self.min_evidence:
            self._revoke("no_evidence")
            return
        last = [r for _, r in self._reloc_events[-10:]]
        if len(last) >= 5 and float(np.median(last)) > self.revoke_frac * self.period:
            self._revoke("inconsistent_residuals")

    def _revoke(self, reason: str) -> None:
        """Undo a provisional closure: revert its corrections, prune keyframes noted
        under the bad period (their poses are contaminated), demand a stronger
        consensus next time, and resume searching."""
        self.revoke_log.append(dict(frame=self.frame, period=self.period,
                                    reason=reason, age=self.frame - self._close_frame,
                                    n_events=len(self._reloc_events)))
        self.n_revoked += 1
        self.pose -= self._corr_sum
        self._revoke_event = dict(kf_len=self._kf_len_at_close, reason=reason,
                                  period=self.period)
        del self.keyframes[self._kf_len_at_close:]
        self.period = None
        self.closure = None
        self.confirm += 2 if self.n_revoked <= 3 else 0  # bounded escalation
        self._close_frame, self._kf_len_at_close = None, None
        self._reloc_events, self._corr_sum, self._votes = [], 0.0, []

    def pop_revoke_event(self) -> Optional[dict]:
        """One-shot: the caller (spin_slam) uses this to prune the SAME keyframes from
        its AtlasStore (slam keyframes and store keyframes are inserted 1:1) and to
        reset the store's yaw_period."""
        ev, self._revoke_event = self._revoke_event, None
        return ev

    # ------------------------------------------------------------------ readouts
    def deg(self, pose: Optional[float] = None) -> Optional[float]:
        """Pose as degrees of heading (needs the measured period)."""
        if self.period is None:
            return None
        p = self.pose if pose is None else pose
        return (p / self.period) * 360.0

    def slope(self) -> Optional[float]:
        """Integrated command->content slope (px per command unit) — the calibration
        Phase 6.5 had to borrow from a separate run, now measured in-run."""
        return self.pose / self.cum_cmd if abs(self.cum_cmd) > 1e-9 else None

    def stats(self) -> dict:
        return {
            "pose_px": self.pose,
            "period_px": self.period,
            "deg": self.deg(),
            "n_keyframes": len(self.keyframes),
            "closure": self.closure,
            "n_revoked": self.n_revoked,
            "revoke_log": self.revoke_log,
            "n_reloc": self.n_reloc,
            "reloc_abs_mean": (self.reloc_abs_sum / self.n_reloc) if self.n_reloc else 0.0,
            "reloc_abs_max": self.reloc_abs_max,
            "slope_px_per_cmd": self.slope(),
        }
