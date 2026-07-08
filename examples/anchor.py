"""
Pixel-space front-door anchoring helpers for Phase 10.

These functions stay model-free so the append-frame experiment can be tested on CPU:
registration uses the Phase 8/9 rule, reconstruction is an RGB band paste, and the
runner owns the actual engine append.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, "examples")
from rigid import register, valid_cols  # noqa: E402
from slam import roi_of  # noqa: E402


@dataclass
class AnchorKeyframe:
    yaw: float
    frames: np.ndarray
    roi: np.ndarray
    seq: int = 0


class AnchorStore:
    """Pose-keyed RGB keyframe store, novelty-gated like RigidStore."""

    def __init__(self, tau_insert: float = 0.05):
        self.tau_insert = tau_insert
        self.keyframes: List[AnchorKeyframe] = []

    def __len__(self):
        return len(self.keyframes)

    def insert(self, frames: np.ndarray, yaw: float) -> bool:
        if self.keyframes and min(abs(k.yaw - yaw) for k in self.keyframes) < self.tau_insert:
            return False
        last = frames[-1]
        roi = roi_of(np.float32(cv2.cvtColor(last, cv2.COLOR_RGB2GRAY)))
        self.keyframes.append(AnchorKeyframe(
            yaw=float(yaw),
            frames=np.array(frames, copy=True),
            roi=roi,
            seq=len(self.keyframes),
        ))
        return True

    def query(self, yaw: float, mode: str = "nearest") -> Optional[AnchorKeyframe]:
        if not self.keyframes:
            return None
        pick = min if mode == "nearest" else max
        return pick(self.keyframes, key=lambda k: abs(k.yaw - yaw))


def pixel_band_mask(h: int, w: int, r_frac=(0.28, 0.64), c_frac=(0.15, 0.85),
                    feather: int = 8) -> np.ndarray:
    """Float32 [H,W,1] mask for the same certified band used in latent projection."""
    m = np.zeros((h, w), dtype=np.float32)
    m[int(r_frac[0] * h):int(r_frac[1] * h),
      int(c_frac[0] * w):int(c_frac[1] * w)] = 1.0
    if feather > 0:
        k = 2 * int(feather) + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
        mx = float(m.max())
        if mx > 0:
            m /= mx
    return m[..., None]


def shift_rgb_x(img: np.ndarray, dx: float) -> np.ndarray:
    """Subpixel horizontal shift with wrap; callers mask wrapped columns out."""
    h, w = img.shape[:2]
    mat = np.float32([[1.0, 0.0, dx], [0.0, 1.0, 0.0]])
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_WRAP)


def confidence_alpha(resp: Optional[float], resp_min: float = 0.05,
                     resp_full: float = 0.55) -> float:
    """Map phase-correlation response to a blend alpha in [0,1]."""
    if resp is None or resp <= resp_min:
        return 0.0
    if resp_full <= resp_min:
        return 1.0
    return float(np.clip((resp - resp_min) / (resp_full - resp_min), 0.0, 1.0))


def blend_u8(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Return alpha*a + (1-alpha)*b as uint8 RGB."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = a.astype(np.float32) * alpha + b.astype(np.float32) * (1.0 - alpha)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def reconstruct_from_keyframe(kf: AnchorKeyframe, current_rgb: np.ndarray,
                              resp_min: float = 0.05, max_shift_frac: float = 0.35,
                              feather: int = 8) -> tuple[np.ndarray, dict]:
    """Registered RGB atlas reconstruction for one current frame.

    Returns (reconstruction_rgb, info). If registration is rejected, the returned
    image is the current frame and info["projected"] is False.
    """
    cur_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    cur_roi = roi_of(np.float32(cur_gray))
    dx, resp = register(kf.roi, cur_roi)
    info = {"projected": False, "dx_px": dx, "resp": resp, "reject": None,
            "kf_yaw": kf.yaw}
    h, w = current_rgb.shape[:2]
    if resp < resp_min:
        info["reject"] = "resp"
        return np.array(current_rgb, copy=True), info
    if abs(dx) > max_shift_frac * w:
        info["reject"] = "shift"
        return np.array(current_rgb, copy=True), info

    mem = shift_rgb_x(kf.frames[-1], dx).astype(np.float32)
    cur = current_rgb.astype(np.float32)
    vc = valid_cols(w, dx).astype(np.float32)[None, :, None]
    mask = pixel_band_mask(h, w, feather=feather) * vc
    recon = cur * (1.0 - mask) + mem * mask
    info["projected"] = True
    return np.clip(np.round(recon), 0, 255).astype(np.uint8), info


def repeat_as_x4(img: np.ndarray) -> np.ndarray:
    """Waypoint-1.5 append-frame batch from a single RGB reconstruction."""
    return np.repeat(img[None], 4, axis=0).astype(np.uint8, copy=False)
