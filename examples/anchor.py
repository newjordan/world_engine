"""
Pixel-space front-door anchoring helpers for Phase 10.

These functions stay model-free so the append-frame experiment can be tested on CPU:
registration uses the Phase 8/9 rule, reconstruction is an RGB band paste, and the
runner owns the actual engine append.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

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
    micro_yaws: Optional[np.ndarray] = None


class AnchorStore:
    """Pose-keyed RGB keyframe store, novelty-gated like RigidStore."""

    def __init__(self, tau_insert: float = 0.05):
        self.tau_insert = tau_insert
        self.keyframes: List[AnchorKeyframe] = []

    def __len__(self):
        return len(self.keyframes)

    def insert(self, frames: np.ndarray, yaw: float, yaw_delta: float = 0.0,
               micro_yaws: Optional[Sequence[float]] = None) -> bool:
        if self.keyframes and min(abs(k.yaw - yaw) for k in self.keyframes) < self.tau_insert:
            return False
        if micro_yaws is None:
            micro_yaws = micro_yaw_grid(yaw, yaw_delta, n=len(frames))
        micro_yaws = np.asarray(micro_yaws, dtype=np.float32)
        if len(micro_yaws) != len(frames):
            raise ValueError("micro_yaws length must match frames length")
        last = frames[-1]
        roi = roi_of(np.float32(cv2.cvtColor(last, cv2.COLOR_RGB2GRAY)))
        self.keyframes.append(AnchorKeyframe(
            yaw=float(yaw),
            frames=np.array(frames, copy=True),
            roi=roi,
            seq=len(self.keyframes),
            micro_yaws=micro_yaws,
        ))
        return True

    def query(self, yaw: float, mode: str = "nearest") -> Optional[AnchorKeyframe]:
        if not self.keyframes:
            return None
        pick = min if mode == "nearest" else max
        return pick(self.keyframes, key=lambda k: abs(k.yaw - yaw))


def micro_yaw_grid(yaw: float, yaw_delta: float, n: int = 4) -> np.ndarray:
    """Estimated per-frame yaws for a decoded chunk ending at `yaw`.

    WorldEngine dead-reckons one yaw delta per generated/append chunk. The VAE
    chunk contains `n` decoded frames, so the best local estimate is a linear
    interpolation from the previous chunk yaw to the final yaw.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    yaw = float(yaw)
    yaw_delta = float(yaw_delta)
    start = yaw - yaw_delta
    return start + yaw_delta * ((np.arange(n, dtype=np.float32) + 1.0) / float(n))


def nearest_micro_index(kf: AnchorKeyframe, yaw: float) -> int:
    """Return the stored micro-frame index whose estimated yaw is nearest."""
    if kf.micro_yaws is None:
        return len(kf.frames) - 1
    return int(np.argmin(np.abs(kf.micro_yaws.astype(np.float32) - float(yaw))))


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


def full_frame_mask(h: int, w: int, dx: float) -> np.ndarray:
    """Float32 [H,W,1] mask covering every non-wrap column."""
    vc = valid_cols(w, dx).astype(np.float32)[None, :, None]
    return np.ones((h, 1, 1), dtype=np.float32) * vc


def confidence_map_mask(h: int, w: int, dx: float, feather: int = 24,
                        floor: float = 0.28) -> np.ndarray:
    """Soft full-frame mask: strongest in certified band, nonzero elsewhere."""
    floor = float(np.clip(floor, 0.0, 1.0))
    band = pixel_band_mask(h, w, feather=max(1, feather))
    mask = floor + (1.0 - floor) * band
    return mask.astype(np.float32) * full_frame_mask(h, w, dx)


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


def reconstruct_from_source(source_rgb: np.ndarray, source_roi: np.ndarray,
                            current_rgb: np.ndarray, source_yaw: Optional[float] = None,
                            resp_min: float = 0.05, max_shift_frac: float = 0.35,
                            feather: int = 8, mask_mode: str = "band",
                            conf_floor: float = 0.28) -> tuple[np.ndarray, dict]:
    """Registered RGB atlas reconstruction from one source frame.

    Returns (reconstruction_rgb, info). If registration is rejected, the returned
    image is the current frame and info["projected"] is False.
    """
    cur_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    cur_roi = roi_of(np.float32(cur_gray))
    dx, resp = register(source_roi, cur_roi)
    info = {"projected": False, "dx_px": dx, "resp": resp, "reject": None,
            "kf_yaw": source_yaw}
    h, w = current_rgb.shape[:2]
    if resp < resp_min:
        info["reject"] = "resp"
        return np.array(current_rgb, copy=True), info
    if abs(dx) > max_shift_frac * w:
        info["reject"] = "shift"
        return np.array(current_rgb, copy=True), info

    mem = shift_rgb_x(source_rgb, dx).astype(np.float32)
    cur = current_rgb.astype(np.float32)
    if mask_mode == "band":
        mask = pixel_band_mask(h, w, feather=feather) * full_frame_mask(h, w, dx)
    elif mask_mode == "full":
        mask = full_frame_mask(h, w, dx)
    elif mask_mode == "confmap":
        mask = confidence_map_mask(h, w, dx, feather=max(feather, 24),
                                   floor=conf_floor)
    else:
        raise ValueError(f"unknown mask_mode {mask_mode!r}")
    recon = cur * (1.0 - mask) + mem * mask
    info["projected"] = True
    info["mask_mode"] = mask_mode
    return np.clip(np.round(recon), 0, 255).astype(np.uint8), info


def reconstruct_from_keyframe(kf: AnchorKeyframe, current_rgb: np.ndarray,
                              resp_min: float = 0.05, max_shift_frac: float = 0.35,
                              feather: int = 8, mask_mode: str = "band",
                              conf_floor: float = 0.28) -> tuple[np.ndarray, dict]:
    """Registered RGB atlas reconstruction for one current frame."""
    return reconstruct_from_source(
        kf.frames[-1],
        kf.roi,
        current_rgb,
        source_yaw=kf.yaw,
        resp_min=resp_min,
        max_shift_frac=max_shift_frac,
        feather=feather,
        mask_mode=mask_mode,
        conf_floor=conf_floor,
    )


def reconstruct_temporal_batch(store: AnchorStore, current_x4: np.ndarray,
                               target_yaws: Sequence[float],
                               retrieve: str = "nearest",
                               blend: bool = False,
                               resp_min: float = 0.05,
                               resp_full: float = 0.55,
                               max_shift_frac: float = 0.35,
                               feather: int = 8, mask_mode: str = "band",
                               conf_floor: float = 0.28) -> tuple[np.ndarray, list[dict]]:
    """Build a four-frame front-door anchor batch from atlas micro-poses.

    Rejected micro-frames fall back to the current generated micro-frame. The
    returned info list is per micro-frame for diagnostics.
    """
    if len(current_x4) != len(target_yaws):
        raise ValueError("target_yaws length must match current_x4 length")
    out, infos = [], []
    for i, (cur, target_yaw) in enumerate(zip(current_x4, target_yaws)):
        kf = store.query(float(target_yaw), mode=retrieve)
        if kf is None:
            info = {"projected": False, "dx_px": None, "resp": None,
                    "reject": "empty", "anchor": False, "alpha": 0.0,
                    "target_yaw": float(target_yaw), "micro_idx": i}
            out.append(np.array(cur, copy=True))
            infos.append(info)
            continue

        src_idx = nearest_micro_index(kf, float(target_yaw))
        src = kf.frames[src_idx]
        src_roi = kf.roi
        if src_idx != len(kf.frames) - 1:
            src_roi = roi_of(np.float32(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)))
        src_yaw = None if kf.micro_yaws is None else float(kf.micro_yaws[src_idx])
        recon, info = reconstruct_from_source(
            src, src_roi, cur,
            source_yaw=src_yaw if src_yaw is not None else kf.yaw,
            resp_min=resp_min,
            max_shift_frac=max_shift_frac,
            feather=feather,
            mask_mode=mask_mode,
            conf_floor=conf_floor,
        )
        alpha = 1.0 if info["projected"] else 0.0
        if blend and info["projected"]:
            alpha = confidence_alpha(info.get("resp"), resp_min, resp_full)
            recon = blend_u8(recon, cur, alpha)
        info.update({
            "anchor": bool(info["projected"]),
            "alpha": alpha,
            "target_yaw": float(target_yaw),
            "micro_idx": i,
            "kf_seq": kf.seq,
            "kf_frame_idx": src_idx,
            "kf_frame_yaw": src_yaw,
        })
        out.append(recon)
        infos.append(info)
    return np.stack(out, axis=0).astype(np.uint8, copy=False), infos


def repeat_as_x4(img: np.ndarray) -> np.ndarray:
    """Waypoint-1.5 append-frame batch from a single RGB reconstruction."""
    return np.repeat(img[None], 4, axis=0).astype(np.uint8, copy=False)
