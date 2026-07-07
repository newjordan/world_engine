# uv run --dev pytest examples/test_graybox.py -v
"""
Graybox: a scaled-down black-and-white lab for surface-vector guidance (Phase 8 pilot).

The full engine's sampler (WorldEngine._denoise_pass) is a plain Euler integrator over
a velocity field v — the frame is produced by riding v down the sigma schedule, and
nothing external ever enters that integration ("the closed vector"). Phases 3-7 only
shaped the CONTEXT that produces v (KV pinning, restamp, atlas, SLAM); they lobby the
vector, they never touch it. Phase 8 asks: what happens when a geometry-certified
memory term is added to v itself,

    v' = v + lam * M (.) (x0_hat - align(x_mem)) / sigma

where M is the surfacing mask (which cells the memory surface claims), x_mem is the
SLAM-keyed atlas memory warped to the current pose, and align() corrects regime drift
so the guidance pulls on geometry, not on stale tone.

This module is the smallest world where that question is real. Everything is grayscale
(one regime axis: gain/bias, the 1-D stand-in for palette/VAE drift) and model-free
(numpy + cv2, no GPU), mirroring how slam.py was validated on a synthetic panorama
before touching the engine. The surrogate engine is built to REPRODUCE the two failure
modes the guidance must trade off:

  FORGETTING   the engine renders from an internal belief refreshed only by its own
               outputs; columns out of view decay, and the decayed render is written
               back — errors accumulate per lap exactly like the real forgetting curve.
  DYNAMICS     one world cell is a blinking beacon (oscillating intensity). The engine
               tracks its phase from its own outputs; memory guidance that claims the
               beacon's cells pins it to a stale lap-1 phase and the tracker locks —
               the Phase 7 "forgetting/pinning breaks dynamics" finding, in miniature.

The barnacle is the real one: YawSLAM (unmodified) runs on the generated frames for
odometry + loop closure, and AtlasStore (unmodified) holds lap-1 frames keyed by SLAM
pose. Guidance switches on at the moment of loop closure, exactly as it would live.

Tunables under study (run_graybox / graybox_tune.py):
  lam          guidance strength
  guide_sigmas which Euler steps receive guidance (all vs late)
  mask_mode    'all' (undifferentiated surface) vs 'static' (beacon cells excluded —
               the two-class profile split)
  align        'none' (raw memory — corrects regime but seams at the mask boundary)
               vs 'gainbias' (memory aligned into the current regime — geometry-only)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, "examples")
from atlas import AtlasStore          # noqa: E402
from slam import YawSLAM, wrap        # noqa: E402

# --------------------------------------------------------------------------- #
# world constants (mirrors test_slam.py so the SLAM operating point is known-good)
# --------------------------------------------------------------------------- #
P_TRUE = 2600            # px per revolution
H, W_VIEW = 120, 640     # frame size
RATE = 7                 # px scrolled per frame

BEACON_COL = 900         # world column of the dynamic cell
BEACON_ROW = 96          # below the SLAM ROI band (rows 33..76) — odometry stays clean
BEACON_R = 9             # blob radius (px)
BEACON_AMP = 90.0        # peak additive intensity
BEACON_OMEGA = 2 * np.pi / 40.0   # one blink cycle per 40 frames

SIGMAS = np.array([1.0, 0.7, 0.45, 0.25, 0.1, 0.0], dtype=np.float64)


def _blob(h: int, w: int, cy: float, cx: float, r: float) -> np.ndarray:
    """Gaussian blob support on an (h, w) grid, peak 1.0 at (cy, cx)."""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * (r / 2.0) ** 2))


class GrayWorld:
    """Ground truth: periodic static texture + one blinking beacon. Used to seed the
    engine's belief (the 'trained model' analog) and for metrics ONLY — generation
    never reads it."""

    def __init__(self, seed: int = 0):
        import cv2
        rng = np.random.default_rng(seed)
        tex = rng.random((H, P_TRUE)).astype(np.float32)
        tex = cv2.GaussianBlur(tex, (0, 0), 3)
        self.tex = cv2.normalize(tex, None, 40, 215, cv2.NORM_MINMAX)

    def beacon_intensity(self, t: int) -> float:
        return 0.5 + 0.5 * np.sin(BEACON_OMEGA * t)

    def beacon_view_col(self, pose: float) -> Optional[float]:
        """Beacon's column in the view window at `pose`, or None if out of view.
        view(pose)[x] = tex[(x - pose) mod P]  =>  x = (BEACON_COL + pose) mod P."""
        x = (BEACON_COL + pose) % P_TRUE
        return x if 0 <= x < W_VIEW else None

    def static_view(self, pose: float) -> np.ndarray:
        """True static content in view (no beacon, no regime, float)."""
        rolled = np.roll(self.tex, int(round(pose)), axis=1)
        return rolled[:, :W_VIEW].astype(np.float64)


@dataclass
class Regime:
    """Global tone drift — the grayscale stand-in for palette/VAE regime drift.
    Random walk with mild mean reversion; applied at render time only."""
    gain: float = 1.0
    bias: float = 0.0
    walk_gain: float = 0.0015
    walk_bias: float = 0.15
    revert: float = 0.0005

    def step(self, rng: np.random.Generator) -> None:
        self.gain *= float(np.exp(rng.normal(0.0, self.walk_gain)))
        self.gain += self.revert * (1.0 - self.gain)
        self.bias += float(rng.normal(0.0, self.walk_bias))
        self.bias -= self.revert * self.bias

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.gain * x + self.bias

    def invert(self, x: np.ndarray) -> np.ndarray:
        return (x - self.bias) / self.gain


class BeaconTracker:
    """The engine's dynamics module: keeps a phase estimate for the beacon, corrected
    from measurements of its OWN previous output. If guidance pins the rendered beacon
    to a stale constant, the corrections drag the phase to a lock — dynamics break as
    an emergent feedback effect, not a scripted one."""

    def __init__(self, k_corr: float = 0.35):
        self.theta = 0.0
        self.k_corr = k_corr

    def step(self) -> None:
        self.theta += BEACON_OMEGA

    def intensity(self) -> float:
        return 0.5 + 0.5 * np.sin(self.theta)

    def correct(self, z: float) -> None:
        """z: measured intensity in [0, 1]. sin is two-to-one per cycle; take the
        candidate phase nearest the current estimate."""
        s = float(np.clip(2.0 * z - 1.0, -1.0, 1.0))
        base = float(np.arcsin(s))
        cands = [base, np.pi - base]
        best = min(cands, key=lambda c: abs(wrap(c - self.theta, 2 * np.pi)))
        self.theta += self.k_corr * wrap(best - self.theta, 2 * np.pi)


class DriftEngine:
    """Surrogate autoregressive flow-matching world model (the 'closed vector').

    Belief: a private copy of the static texture, refreshed ONLY from the engine's own
    outputs (write_back). Columns not seen for a while render corrupted (staleness
    noise), and the corrupted output is what gets written back — so absence bakes in
    degradation, lap after lap, like the real forgetting curve.

    Rendering composes: belief crop -> staleness corruption -> beacon (at the
    TRACKER's phase) -> regime tone drift. The sampler target ŷ is this render.
    """

    def __init__(self, world: GrayWorld, seed: int = 1,
                 t_forget: float = 600.0, c_max: float = 0.5):
        self.belief = world.tex.astype(np.float64).copy()
        self.stale = np.zeros(P_TRUE, dtype=np.float64)
        self.t_forget = t_forget
        self.c_max = c_max
        self.regime = Regime()
        self.tracker = BeaconTracker()
        self.rng = np.random.default_rng(seed)
        self._last_nobeacon: Optional[np.ndarray] = None
        self._last_blob: Optional[np.ndarray] = None
        self._last_beacon_i: float = 0.0

    def render(self, pose: float, world: GrayWorld) -> np.ndarray:
        """The sampler target ŷ for the current pose (float, regime applied)."""
        p = int(round(pose))
        cols = (np.arange(W_VIEW) - p) % P_TRUE          # view(pose)[x] = tex[(x-p)%P]
        crop = self.belief[:, cols].copy()
        w = self.c_max * np.minimum(1.0, self.stale[cols] / self.t_forget)
        noise = self.rng.normal(128.0, 55.0, crop.shape)
        crop = (1.0 - w)[None, :] * crop + w[None, :] * noise

        self._last_blob = None
        bx = world.beacon_view_col(pose)
        if bx is not None:
            blob = _blob(H, W_VIEW, BEACON_ROW, bx, BEACON_R)
            self._last_nobeacon = crop.copy()
            self._last_beacon_i = self.tracker.intensity()   # frozen for write_back
            crop = crop + self._last_beacon_i * BEACON_AMP * blob
            self._last_blob = blob
        return self.regime.apply(crop)

    def write_back(self, pose: float, out: np.ndarray) -> None:
        """Autoregressive context update: the emitted frame (regime-inverted with the
        engine's own tone state) becomes the belief for the columns in view."""
        p = int(round(pose))
        cols = (np.arange(W_VIEW) - p) % P_TRUE
        content = self.regime.invert(out.astype(np.float64))
        if self._last_blob is not None:   # don't bake the beacon into static belief
            content = content - self._last_beacon_i * BEACON_AMP * self._last_blob
        self.belief[:, cols] = content
        self.stale += 1.0
        self.stale[cols] = 0.0

    def observe_own_output(self, out: np.ndarray) -> None:
        """Measure the beacon in the emitted frame, correct the tracker against the
        phase that RENDERED it (correct-then-step — stepping first drags the phase),
        then advance. This is the feedback path through which pinned outputs freeze
        the dynamics."""
        if self._last_blob is not None:
            blob, base = self._last_blob, self._last_nobeacon
            core = blob > 0.4
            content = self.regime.invert(out.astype(np.float64))
            z = float(((content - base)[core] / BEACON_AMP).mean()
                      / max(blob[core].mean(), 1e-6))
            self.tracker.correct(z)
        self.tracker.step()


# --------------------------------------------------------------------------- #
# the barnacle: SLAM-keyed atlas memory injected into the sampler's vector
# --------------------------------------------------------------------------- #
@dataclass
class GuidanceCfg:
    lam: float = 0.35
    guide_sigmas: tuple = (0.45, 0.25, 0.1)   # which Euler steps receive guidance
    mask_mode: str = "static"                 # 'all' | 'static'
    align: str = "gainbias"                   # 'none' | 'gainbias'
    kf_every: int = 2
    oracle_pose: bool = False                 # ablation: bypass SLAM for retrieval


class Barnacle:
    """YawSLAM + frame atlas + the guidance vector. Rides on the generated frames;
    guidance switches on at loop closure (before that there is no verified geometry)."""

    def __init__(self, cfg: GuidanceCfg):
        self.cfg = cfg
        self.slam = YawSLAM(ppr_guess=P_TRUE * 1.1)
        self.store = AtlasStore(tau_insert=RATE, n_max=512)
        # per-cell |aligned_mem - x0_hat| accumulators: the free classifier signal
        self.cell_err_sum = np.zeros((H, W_VIEW))
        self.cell_err_n = 0
        self.last_mem: Optional[np.ndarray] = None    # diagnostics (post-align)
        self.last_valid: Optional[np.ndarray] = None
        self.last_fit: tuple = (1.0, 0.0)
        self.last_refine: float = 0.0                 # appearance-vs-pose shift (px):
                                                      # the map's registration bias

    @property
    def active(self) -> bool:
        return self.slam.period is not None and len(self.store) > 0

    def after_frame(self, out_u8: np.ndarray, t: int) -> None:
        """Odometry + keyframing on the emitted frame (mirrors the spin_slam wiring:
        store insert and slam.note_keyframe stay 1:1; revocation prunes both)."""
        self.slam.observe(out_u8)
        if t % self.cfg.kf_every == 0:
            if self.store.insert(out_u8.astype(np.float64), yaw=self.slam.pose,
                                 frame_ts=t):
                self.slam.note_keyframe(out_u8)
        if self.slam.period is not None:
            self.store.yaw_period = self.slam.period
        ev = self.slam.pop_revoke_event()
        if ev is not None:
            del self.store.keyframes[ev["kf_len"]:]
            self.store.yaw_period = None

    def memory_view(self, true_pose: float, ref: np.ndarray) -> Optional[tuple]:
        """(mem, valid, refine_px): the nearest stored keyframe REGISTERED to the
        current denoise target, plus a validity mask for the rolled-in strip.

        Registration is two-stage, and the second stage is load-bearing: pose only
        SELECTS the keyframe; the shift applied comes from phase-correlating the
        keyframe against `ref` (the current x0_hat). Keyframe poses carry the map's
        accumulated odometry bias, so a pose-computed shift has a systematic
        translation component — and a guidance vector with a net translation is a
        FORCE on the world's dynamics: it drags rendered content backward, the
        odometry (which measures rendered content) lags further, retrieval regresses,
        and the scroll stalls in a runaway (observed: -11px bias -> -39px in 15
        frames -> run destroyed). Appearance registration zeroes the first moment of
        the guidance: it may correct texture and tone, never position. Position
        errors belong to the SLAM/reloc channel, not to g."""
        if not self.active:
            return None
        q = true_pose if self.cfg.oracle_pose else self.slam.pose
        period = P_TRUE if self.cfg.oracle_pose else self.slam.period
        old = self.store.yaw_period
        self.store.yaw_period = period
        hit = self.store.query(float(q), k=1)[0]
        self.store.yaw_period = old
        coarse = int(round(wrap(q - hit.yaw, period)))

        r0, r1 = int(0.28 * H), int(0.64 * H)
        c0, c1 = int(0.15 * W_VIEW), int(0.85 * W_VIEW)
        kf_roi = np.float32(hit.kv[r0:r1, c0:c1])
        ref_roi = np.float32(ref[r0:r1, c0:c1])
        win = cv2.createHanningWindow((kf_roi.shape[1], kf_roi.shape[0]), cv2.CV_32F)
        (dx, _dy), resp = cv2.phaseCorrelate(kf_roi, ref_roi, win)
        if resp < 0.03 or abs(dx) > 60.0:
            return None                       # no certified registration -> no claim
        delta = int(round(dx))                # current = kf rolled by +delta
        self.last_refine = float(dx - coarse)
        mem = np.roll(hit.kv, delta, axis=1)
        valid = np.ones((H, W_VIEW), dtype=bool)
        if delta > 0:
            valid[:, :delta] = False
        elif delta < 0:
            valid[:, delta:] = False
        return mem, valid, self.last_refine

    def mask(self, valid: np.ndarray, beacon_x: Optional[float]) -> np.ndarray:
        """The surfacing mask M: what the memory surface claims. 'all' claims every
        valid cell (undifferentiated profile); 'static' releases the beacon's cells
        (the two-class split)."""
        m = valid.astype(np.float64)
        if self.cfg.mask_mode == "static" and beacon_x is not None:
            blob = _blob(H, W_VIEW, BEACON_ROW, beacon_x, BEACON_R * 2.0)
            m *= (blob < 0.05)
        return m

    def guide(self, x0_hat: np.ndarray, sigma: float, true_pose: float,
              beacon_x: Optional[float]) -> Optional[np.ndarray]:
        """The guidance term g to add to v (already includes the 1/sigma scale)."""
        mv = self.memory_view(true_pose, x0_hat)
        if mv is None:
            return None
        mem, valid, _refine = mv
        roi = np.zeros((H, W_VIEW), dtype=bool)
        roi[int(0.28 * H):int(0.64 * H), int(0.15 * W_VIEW):int(0.85 * W_VIEW)] = True
        fit_cells = valid & roi                     # SLAM-certified band only
        if self.cfg.align == "gainbias" and fit_cells.sum() > 100:
            mm, xx = mem[fit_cells], x0_hat[fit_cells]
            var = float(((mm - mm.mean()) ** 2).sum())
            a = float(((mm - mm.mean()) * (xx - xx.mean())).sum()) / max(var, 1e-9)
            a = float(np.clip(a, 0.5, 2.0))
            b = float(xx.mean() - a * mm.mean())
            mem = a * mem + b
            self.last_fit = (a, b)
        m = self.mask(valid, beacon_x)
        self.last_mem, self.last_valid = mem, valid
        self.cell_err_sum += m * np.abs(mem - x0_hat)
        self.cell_err_n += 1
        return self.cfg.lam * m * (x0_hat - mem) / max(sigma, 1e-3)


# --------------------------------------------------------------------------- #
# the sampler: same Euler shape as WorldEngine._denoise_pass, plus the barnacle
# --------------------------------------------------------------------------- #
def sample_frame(engine: DriftEngine, world: GrayWorld, pose: float,
                 barnacle: Optional[Barnacle], rng: np.random.Generator,
                 model_noise: float = 3.0) -> np.ndarray:
    """One generated frame: integrate v down the sigma schedule; if the barnacle is
    active, add its guidance to v at the configured steps.

    LIMIT OF THE SURROGATE: the analytic target y_hat does not depend on x, so the
    terminal Euler step (sigma -> 0) lands exactly on the predicted endpoint and
    erases guided history — every schedule containing the last step behaves
    identically, and the final output is exactly (1 - lam*M) * y_hat' + lam*M * mem
    (lam IS the final blend weight; lam > 1 overshoots past the memory, which is the
    E1 cliff). In the real engine the network's implicit target moves with x, so the
    schedule question (guide early vs late) is only answerable there."""
    y_hat = engine.render(pose, world)
    x = rng.normal(128.0, 64.0, y_hat.shape)                 # pure noise at sigma=1
    beacon_x = world.beacon_view_col(pose)
    for sig, dsig in zip(SIGMAS[:-1], np.diff(SIGMAS)):
        v = (x - y_hat) / sig + rng.normal(0.0, model_noise, x.shape)
        if barnacle is not None and barnacle.active and \
                any(abs(sig - gs) < 1e-9 for gs in barnacle.cfg.guide_sigmas):
            x0_hat = x - sig * v
            g = barnacle.guide(x0_hat, sig, pose, beacon_x)
            if g is not None:
                v = v + g
        x = x + dsig * v
    return x


# --------------------------------------------------------------------------- #
# the run loop + metrics
# --------------------------------------------------------------------------- #
def run_graybox(n_laps: float = 4.0, cfg: Optional[GuidanceCfg] = None,
                seed: int = 0) -> dict:
    """Drive the closed loop for n_laps; return per-lap metrics.

    Metrics (per lap):
      static_rmse   RMSE of the emitted frame (regime-inverted with the engine's true
                    tone state) vs the TRUE static view, beacon cells excluded — the
                    forgetting / fidelity curve.
      beacon_corr   Pearson correlation of the emitted beacon intensity series vs the
                    true blink — the dynamics-preservation metric.
      seam          mean |horizontal gradient| excess of the emitted frame over the
                    true view at the beacon mask boundary — tone-mismatch seams.
      regime_gain   fitted output-vs-truth gain error |a-1| on static cells (how much
                    tone drift reaches the emitted frames).
    """
    world = GrayWorld(seed=seed)
    engine = DriftEngine(world, seed=seed + 1)
    barnacle = Barnacle(cfg) if cfg is not None else None
    rng = np.random.default_rng(seed + 2)

    n_frames = int(n_laps * P_TRUE / RATE)
    rec = {k: [[] for _ in range(int(np.ceil(n_laps)))]
           for k in ("static_rmse", "beacon_z", "beacon_true", "seam", "gain_err",
                     "mem_ncc", "refine")}

    pose = 0.0
    for t in range(n_frames):
        out = sample_frame(engine, world, pose, barnacle, rng)
        out_u8 = np.clip(out, 0, 255).astype(np.uint8)
        lap = min(int(pose / P_TRUE), len(rec["static_rmse"]) - 1)

        # ---- metrics (oracle knowledge: lab only) ----
        content = engine.regime.invert(out)
        truth = world.static_view(pose)
        bx = world.beacon_view_col(pose)
        cells = np.ones((H, W_VIEW), dtype=bool)
        if bx is not None:
            blob = _blob(H, W_VIEW, BEACON_ROW, bx, BEACON_R * 2.0)
            cells &= blob < 0.05
        rec["static_rmse"][lap].append(
            float(np.sqrt(((content - truth)[cells] ** 2).mean())))
        p = int(round(pose))
        cols = (np.arange(W_VIEW) - p) % P_TRUE
        fresh = cells & (engine.stale[cols] < 50.0)[None, :]   # tone fit: fresh only
        if fresh.sum() > 500:
            mm, xx = truth[fresh], out[fresh]
            var = float(((mm - mm.mean()) ** 2).sum())
            a = float(((mm - mm.mean()) * (xx - xx.mean())).sum()) / max(var, 1e-9)
            rec["gain_err"][lap].append(abs(a - 1.0))
        if barnacle is not None and barnacle.last_mem is not None:
            mem, vd = barnacle.last_mem, barnacle.last_valid
            sel = vd & cells
            m0, t0 = mem[sel] - mem[sel].mean(), truth[sel] - truth[sel].mean()
            denom = float(np.linalg.norm(m0) * np.linalg.norm(t0))
            if denom > 1e-9:
                rec["mem_ncc"][lap].append(float((m0 * t0).sum()) / denom)
            rec["refine"][lap].append(barnacle.last_refine)
            barnacle.last_mem = None
        if bx is not None:
            blob = _blob(H, W_VIEW, BEACON_ROW, bx, BEACON_R)
            core = blob > 0.4
            z = float(((content - truth)[core] / BEACON_AMP).mean()
                      / max(blob[core].mean(), 1e-6))
            rec["beacon_z"][lap].append(z)
            rec["beacon_true"][lap].append(world.beacon_intensity(t))
            edge = (blob > 0.02) & (blob < 0.3)
            gx_out = np.abs(np.diff(out, axis=1))
            gx_true = np.abs(np.diff(engine.regime.apply(truth), axis=1))
            e = edge[:, :-1]
            rec["seam"][lap].append(float((gx_out[e] - gx_true[e]).mean()))

        # ---- close the loop ----
        engine.observe_own_output(out)
        engine.write_back(pose, out)
        engine.regime.step(engine.rng)
        if barnacle is not None:
            barnacle.after_frame(out_u8, t)
        pose += RATE

    def lap_stat(key):
        return [float(np.mean(v)) if v else float("nan") for v in rec[key]]

    def corr(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if len(a) < 8 or a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    out = {
        "static_rmse": lap_stat("static_rmse"),
        "beacon_corr": [corr(z, s) for z, s in zip(rec["beacon_z"],
                                                   rec["beacon_true"])],
        "seam": lap_stat("seam"),
        "gain_err": lap_stat("gain_err"),
        "mem_ncc": lap_stat("mem_ncc"),
        "refine": lap_stat("refine"),
    }
    if barnacle is not None:
        out["slam"] = barnacle.slam.stats()
        out["atlas"] = barnacle.store.stats()
        if barnacle.cell_err_n > 0:
            per_cell = barnacle.cell_err_sum / barnacle.cell_err_n
            roi = per_cell[int(0.28 * H):int(0.64 * H), :]
            out["cell_err_roi_mean"] = float(roi.mean())
    return out
