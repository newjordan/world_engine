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

    def query_k(self, yaw: float, k: int = 8, mode: str = "min") -> List[AnchorKeyframe]:
        """Up to k keyframes ranked by yaw distance. mode="min" nearest first,
        mode="max" farthest first. Returns the whole store when k exceeds it."""
        if not self.keyframes:
            return []
        ordered = sorted(self.keyframes, key=lambda kf: abs(kf.yaw - yaw),
                         reverse=(mode == "max"))
        return ordered[:max(1, k)]


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


def calibrate_px_per_yaw(store: AnchorStore, resp_min: float = 0.05,
                         max_shift_frac: float = 0.35) -> tuple[Optional[float], int]:
    """Self-calibrated content scale: px of horizontal shift per command-yaw
    unit, from registering adjacent-by-yaw keyframe ROIs at known yaw
    separation. Returns (median dx/dyaw over pairs passing the resp/shift
    gates, n_pairs); (None, 0) when no pair is registerable. Logged diagnostic
    only since Phase 10g.2 — it does not gate acceptance.
    """
    kfs = sorted(store.keyframes, key=lambda kf: kf.yaw)
    ratios = []
    for a, b in zip(kfs, kfs[1:]):
        dyaw = a.yaw - b.yaw
        if dyaw == 0.0:
            continue
        dx, resp = register(a.roi, b.roi)
        if resp < resp_min or abs(dx) > max_shift_frac * a.frames.shape[2]:
            continue
        ratios.append(dx / dyaw)
    if not ratios:
        return None, 0
    return float(np.median(ratios)), len(ratios)


def pose_plausible(dx: float, px_per_yaw: float, yaw_offset: float,
                   bound: float = 1.5) -> bool:
    """Pose-plausibility residual check. Refuted as a gate in Phase 10g.2 (the
    residual cannot separate content drift from wrong-pose aliasing); kept as a
    logged diagnostic only."""
    return abs(dx / px_per_yaw - yaw_offset) <= bound


def pose_window(store: AnchorStore, window_units: float = 3.0,
                spacing_mult: float = 2.5) -> float:
    """Pose-prior candidate window (Phase 10g.2): max of a fixed floor and a
    spacing-aware term so the pool spans several keyframes at any protocol."""
    yaws = sorted(kf.yaw for kf in store.keyframes)
    gaps = [b - a for a, b in zip(yaws, yaws[1:])]
    spacing = float(np.median(gaps)) if gaps else 0.0
    return max(float(window_units), spacing_mult * spacing)


def reacquire(store: AnchorStore, cur_roi: np.ndarray, yaw: float, k: int = 8,
              mode: str = "min", resp_min: float = 0.05,
              max_shift_frac: float = 0.35,
              frame_w: Optional[float] = None,
              window_units: float = 3.0,
              spacing_mult: float = 2.5) -> dict:
    """Content-defined loop-closure re-acquisition (Phase 10g, amended 10g.2).

    Candidates from query_k are first restricted to a pose-prior window
    (|kf.yaw - yaw| <= pose_window(store)), then the current ROI is registered
    against each survivor and the highest-response match passing the resp/shift
    gates wins. px_per_yaw and the winner's plausibility residual are logged
    diagnostics only. reject="empty" when the store is empty, reject="nomatch"
    when the pool is empty or nothing passes the gates — fail closed.
    """
    candidates = store.query_k(yaw, k=k, mode=mode)
    result = {"winner": None, "winner_dx": None, "winner_resp": None,
              "runner_up_resp": None, "n_candidates": len(candidates),
              "n_windowed_out": 0, "window": None, "px_per_yaw": None,
              "calib_pairs": 0, "plaus_residual": None, "reject": None}
    if not candidates:
        result["reject"] = "empty"
        return result
    window = pose_window(store, window_units, spacing_mult)
    pooled = [kf for kf in candidates if abs(kf.yaw - yaw) <= window]
    result.update(window=window, n_windowed_out=len(candidates) - len(pooled))
    px_per_yaw, n_pairs = calibrate_px_per_yaw(store, resp_min=resp_min,
                                               max_shift_frac=max_shift_frac)
    result.update(px_per_yaw=px_per_yaw, calib_pairs=n_pairs)
    if not pooled:
        result["reject"] = "nomatch"
        return result
    if frame_w is None:
        # roi_of crops columns to ~0.70 of the frame width (0.15..0.85).
        frame_w = cur_roi.shape[1] / 0.70
    scored = [(resp, dx, kf) for kf, (dx, resp)
              in ((kf, register(kf.roi, cur_roi)) for kf in pooled)]
    scored.sort(key=lambda t: t[0], reverse=True)
    result["runner_up_resp"] = float(scored[1][0]) if len(scored) > 1 else None
    for resp, dx, kf in scored:
        if resp < resp_min or abs(dx) > max_shift_frac * frame_w:
            continue
        if px_per_yaw:
            result["plaus_residual"] = abs(dx / px_per_yaw - (kf.yaw - yaw))
        result.update(winner=kf, winner_dx=float(dx), winner_resp=float(resp))
        return result
    best_resp, best_dx, _ = scored[0]
    result.update(winner_dx=float(best_dx), winner_resp=float(best_resp),
                  reject="nomatch")
    return result


def reconstruct_temporal_from_keyframe(kf: AnchorKeyframe, current_x4: np.ndarray,
                                       target_yaws: Sequence[float],
                                       blend: bool = False,
                                       resp_min: float = 0.05,
                                       resp_full: float = 0.55,
                                       max_shift_frac: float = 0.35,
                                       feather: int = 8, mask_mode: str = "band",
                                       conf_floor: float = 0.28) -> tuple[np.ndarray, list[dict]]:
    """Temporal batch reconstruction forced against a single keyframe.

    Used by re-acquisition arms: the content-winning keyframe (not per-micro yaw
    retrieval) is the source, and target_yaws are the winner's stored micro_yaws.
    """
    if len(current_x4) != len(target_yaws):
        raise ValueError("target_yaws length must match current_x4 length")
    out, infos = [], []
    for i, (cur, target_yaw) in enumerate(zip(current_x4, target_yaws)):
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


def reacq_reconstruct(store: AnchorStore, current_x4: np.ndarray, yaw: float,
                      k: int = 8, mode: str = "min", resp_min: float = 0.05,
                      resp_full: float = 0.55, max_shift_frac: float = 0.35,
                      feather: int = 8, mask_mode: str = "full",
                      conf_floor: float = 0.28, blend: bool = False):
    """Re-acquire the content-best keyframe and reconstruct the chunk against it.

    Returns (recon_x4_or_None, micro_infos, diag). recon is None when the search
    finds no admissible keyframe (diag["reject"] in {"empty", "nomatch"}); pose is
    reset to the winner's stored micro_yaws before reconstruction.
    """
    cur_roi = roi_of(np.float32(cv2.cvtColor(current_x4[-1], cv2.COLOR_RGB2GRAY)))
    res = reacquire(store, cur_roi, yaw, k=k, mode=mode, resp_min=resp_min,
                    max_shift_frac=max_shift_frac, frame_w=current_x4.shape[2])
    win = res["winner"]
    diag = {
        "n_candidates": res["n_candidates"],
        "winner_seq": None if win is None else int(win.seq),
        "winner_yaw": None if win is None else float(win.yaw),
        "drift": None if win is None else float(win.yaw - yaw),
        "winner_dx": res["winner_dx"],
        "winner_resp": res["winner_resp"],
        "runner_up_resp": res["runner_up_resp"],
        "window": res["window"],
        "n_windowed_out": res["n_windowed_out"],
        "px_per_yaw": res["px_per_yaw"],
        "calib_pairs": res["calib_pairs"],
        "plaus_residual": res["plaus_residual"],
        "reject": res["reject"],
    }
    if win is None:
        return None, [], diag
    if win.micro_yaws is not None and len(win.micro_yaws) == len(current_x4):
        target_yaws = win.micro_yaws
    else:
        target_yaws = [float(win.yaw)] * len(current_x4)
    recon_x4, micro_infos = reconstruct_temporal_from_keyframe(
        win, current_x4, target_yaws, blend=blend, resp_min=resp_min,
        resp_full=resp_full, max_shift_frac=max_shift_frac, feather=feather,
        mask_mode=mask_mode, conf_floor=conf_floor)
    return recon_x4, micro_infos, diag


def repeat_as_x4(img: np.ndarray) -> np.ndarray:
    """Waypoint-1.5 append-frame batch from a single RGB reconstruction."""
    return np.repeat(img[None], 4, axis=0).astype(np.uint8, copy=False)
