# uv run --dev pytest examples/test_rigid.py -v
"""
Rigid latent projection (Phase 8): force the diffusion line through a rigid matrix.

Every prior memory mechanism here (pinning, restamp, atlas: Phases 3-7) shaped the
CONTEXT that produces the sampler's velocity field v — and topped out at ~+1.7 dB,
because the frozen attention under-weights memory (the attention wall). This module
bypasses attention entirely: it constrains the sampler's OUTPUT. After the Euler chain
produces the provisional latent x0, the SLAM-certified band of a registered memory
keyframe replaces (lam=1) or blends into (lam<1) the same band of x0:

    x0' = (1 - lam*M) (.) x0  +  lam*M (.) warp(x0_mem)

and — the durability half — x0' (not x0) is what _cache_pass writes into the KV
context, so the correction re-anchors the autoregressive rollout instead of repainting
one frame. Applying the projection only after the final Euler step is exact for this
purpose: the graybox lab (examples/graybox.py, docs/GRAYBOX_RESULTS.md) showed the
terminal step makes endpoint projection equivalent to per-step guidance at lam == final
blend weight, with a stable plateau through lam=1.

Two safety rules, both established in the graybox and enforced here:

  ZERO NET TRANSLATION   pose (dead-reckoned camera_yaw) only SELECTS the keyframe;
                         the shift actually applied comes from phase-correlating the
                         keyframe against the provisional frame (appearance
                         registration), response-gated. A pose-computed shift carries
                         the map's odometry bias, and a memory force with a net
                         translation drags the world's dynamics into a runaway stall.
  EVIDENCE-GATED CLAIM   weak correlation response or an implausible shift means no
                         certified registration -> no projection this frame. The rigid
                         matrix claims only what it can prove is the same rigid surface.

The mask M is the SLAM-certified band (the roi_of crop, mapped to latent coordinates),
feathered one latent cell. This is the rigid-static profile ONLY — the graybox showed
hard projection outside the certified static class freezes dynamics; other surface
classes get other profiles (or none), later.

Torch-touching code is confined to rigid_step (the engine splice); everything else is
CPU-testable without a model (examples/test_rigid.py), mirroring atlas.py's layout.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, "examples")
from slam import roi_of  # noqa: E402  (same certified band as the SLAM estimator)


# --------------------------------------------------------------------------- #
# pure retrieval / registration geometry (CPU, model-free)
# --------------------------------------------------------------------------- #
@dataclass
class RigidKeyframe:
    """One stored view: pose tag (selection only), latent payload, appearance ROI."""
    yaw: float
    x0: "object"          # torch latent [1, C, h, w] on CPU (opaque here)
    roi: np.ndarray       # full-res gray ROI f32 (registration reference)
    seq: int = 0


class RigidStore:
    """Pose-keyed keyframe store, novelty-gated like AtlasStore; linear yaw distance
    (the hysteresis protocol never wraps a revolution)."""

    def __init__(self, tau_insert: float = 0.05):
        self.tau_insert = tau_insert
        self.keyframes: List[RigidKeyframe] = []

    def __len__(self):
        return len(self.keyframes)

    def insert(self, x0, roi: np.ndarray, yaw: float) -> bool:
        if self.keyframes and min(abs(k.yaw - yaw) for k in self.keyframes) < self.tau_insert:
            return False
        self.keyframes.append(RigidKeyframe(yaw=float(yaw), x0=x0, roi=roi,
                                            seq=len(self.keyframes)))
        return True

    def query(self, yaw: float, mode: str = "nearest") -> Optional[RigidKeyframe]:
        if not self.keyframes:
            return None
        pick = min if mode == "nearest" else max
        return pick(self.keyframes, key=lambda k: abs(k.yaw - yaw))


_WIN_CACHE: dict = {}


def register(kf_roi: np.ndarray, cur_roi: np.ndarray) -> tuple:
    """(dx_px, resp): current content equals keyframe content scrolled right by dx_px
    (slam._refine convention), via windowed phase correlation on the certified band."""
    shape = (kf_roi.shape[1], kf_roi.shape[0])
    if shape not in _WIN_CACHE:
        _WIN_CACHE[shape] = cv2.createHanningWindow(shape, cv2.CV_32F)
    (dx, _dy), resp = cv2.phaseCorrelate(np.float32(kf_roi), np.float32(cur_roi),
                                         _WIN_CACHE[shape])
    return float(dx), float(resp)


def valid_cols(w: int, shift: float) -> np.ndarray:
    """Boolean (w,) mask of columns NOT contaminated by roll wrap-around."""
    v = np.ones(w, dtype=bool)
    n = int(np.ceil(abs(shift)))
    if n == 0:
        return v
    if shift > 0:
        v[:min(n, w)] = False
    else:
        v[-min(n, w):] = False
    return v


# --------------------------------------------------------------------------- #
# torch side: mask, fractional roll, and the engine splice
# --------------------------------------------------------------------------- #
import torch  # noqa: E402  (kept below the pure section, like atlas.py)


def frac_roll(x: torch.Tensor, shift: float) -> torch.Tensor:
    """Roll x along its last dim by a fractional number of columns (lerp of the two
    integer rolls). Sub-cell registration matters: one latent column is ~20+ px."""
    n = int(np.floor(shift))
    f = float(shift - n)
    a = torch.roll(x, n, dims=-1)
    if f == 0.0:
        return a
    return a.lerp(torch.roll(x, n + 1, dims=-1), f)


def band_mask(h: int, w: int, device, r_frac=(0.28, 0.64), c_frac=(0.15, 0.85),
              feather: int = 1) -> torch.Tensor:
    """The rigid-static surfacing mask in latent coordinates: the SLAM-certified band
    (same fractions as slam.roi_of), feathered so the hard projection does not leave a
    step edge for the decoder. Returns float32 [1, 1, 1, h, w] in [0, 1]."""
    m = torch.zeros(1, 1, h, w, dtype=torch.float32, device=device)
    m[..., int(r_frac[0] * h):int(r_frac[1] * h),
         int(c_frac[0] * w):int(c_frac[1] * w)] = 1.0
    for _ in range(max(0, feather)):
        m = torch.nn.functional.avg_pool2d(m, 3, stride=1, padding=1)
    return m.unsqueeze(0)


@torch.inference_mode()
def rigid_step(engine, ctrl, store: Optional[RigidStore], mask: torch.Tensor,
               project: bool = False, capture: bool = False, lam: float = 1.0,
               retrieve: str = "nearest", resp_min: float = 0.05,
               max_shift_frac: float = 0.35, write_back: bool = True,
               wb_dx_max: Optional[float] = None, score_projected: bool = True):
    """One generated frame with optional rigid projection, replicating
    WorldEngine.gen_frame with the projection spliced between the denoise and the
    cache write-back (so the projected latent becomes the context). Returns
    (four_rgb_u8 [4,H,W,3], info dict).

    write_back=False caches the UNprojected latent while still decoding/scoring the
    projected one — a diagnostic that isolates the KV-context channel: if the mid-run
    collapse is a write-back cascade (paste -> poisoned context -> worse provisional ->
    worse paste), this arm keeps the early gain and loses the trough.

    wb_dx_max (px): write back only when the registered shift is at most this —
    the anchor rule. The diagnostic sweep showed writing back a rolled paste stalls
    the world's rotation, the applied shift then grows without bound (10 -> 250 px
    over one pan-back), and quality collapses. A paste written back only when the
    world is already aligned to the map never rolls content through the context,
    so it can re-anchor (negative feedback) but not drag (positive feedback).

    score_projected=False returns the PROVISIONAL decode while still applying the
    write-back rule to the context — isolates the genuine context benefit from the
    cosmetic effect of pasting memory into the scored pixels.

    The VAE decoder is a temporal stream: registering requires decoding the
    provisional latent, so on an accepted projection the stream state is rewound and
    the projected latent is decoded in its place (exactly one effective decode per
    frame). On a rejected registration the provisional decode stands."""
    x = torch.randn(engine.frm_shape, device=engine.device, dtype=engine.dtype)
    inputs = engine.prep_inputs(x=x, ctrl=ctrl)
    x0 = engine._denoise_pass(x, inputs, engine.kv_cache).clone()

    info = {"projected": False, "dx_px": None, "resp": None, "reject": None,
            "kf_yaw": None, "wrote_back": False}
    vst = engine.vae.get_state() if (project and store and len(store)) else None
    four = engine.vae.decode(x0.squeeze(1))

    x0_cache = x0
    if vst is not None:
        cur_gray = cv2.cvtColor(four[-1].detach().cpu().numpy(), cv2.COLOR_RGB2GRAY)
        cur_roi = roi_of(np.float32(cur_gray))
        kf = store.query(float(engine.camera_yaw), mode=retrieve)
        dx, resp = register(kf.roi, cur_roi)
        info.update(dx_px=dx, resp=resp, kf_yaw=kf.yaw)
        img_w = cur_gray.shape[1]
        if resp < resp_min:
            info["reject"] = "resp"
        elif abs(dx) > max_shift_frac * img_w:
            info["reject"] = "shift"
        else:
            lat_w = x0.shape[-1]
            dx_lat = dx * lat_w / img_w
            mem = kf.x0.to(engine.device).float().unsqueeze(0)   # [1,1,C,h,w]
            mem = frac_roll(mem, dx_lat)
            vc = torch.from_numpy(valid_cols(lat_w, dx_lat)).to(engine.device)
            m = (mask * lam) * vc.float().view(1, 1, 1, 1, -1)
            x0_proj = (x0.float() * (1.0 - m) + mem * m).to(x0.dtype)
            info["projected"] = True
            if write_back and (wb_dx_max is None or abs(dx) <= wb_dx_max):
                x0_cache = x0_proj
                info["wrote_back"] = True
            if score_projected:
                engine.vae.load_state(vst)
                four = engine.vae.decode(x0_proj.squeeze(1))
            x0 = x0_proj

    engine._cache_pass(x0_cache, inputs, engine.kv_cache)

    if capture and store is not None:
        gray = cv2.cvtColor(four[-1].detach().cpu().numpy(), cv2.COLOR_RGB2GRAY)
        store.insert(x0.squeeze(0).detach().to("cpu", copy=True),
                     roi_of(np.float32(gray)), float(engine.camera_yaw))
    return four, info
