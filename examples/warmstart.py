# uv run --dev pytest examples/test_warmstart.py -v
"""
Warmstart / re-dream (Phase 9): launder memory through the model's own sampler.

Phase 8 (examples/rigid.py, docs/RIGID_RESULTS.md) ended on a dilemma. Output-side
memory works — the registered band overlay stacks with the atlas to the +3.40 dB
record — but the durability half is forbidden: ANY hard latent edit written into the
KV context is OOD for the frozen model (anchor_pure −0.77 even when perfectly
aligned), and a rolled paste self-registers into a runaway (|dx| 10→250 px in one
pan-back). The overlay repairs the picture; nothing re-anchors the rollout.

This module restores the durability channel by re-expressing the edit AS A SAMPLE.
The registered memory band is composited into the sampler's PROPOSAL, renoised to a
mid-schedule sigma, and the tail of the Euler chain is re-run under the same KV
context — so the model itself reconciles memory with context before anything touches
the cache:

    x0_prop = full denoise from pure noise           (the model's unaided belief)
    dx,resp = phaseCorrelate(kf.roi, decode(x0_prop))            [register]
    x0_comp = (1 − λ·M) ⊙ x0_prop + λ·M ⊙ warp(x0_mem)           [compose]
    x_s     = (1 − s) · x0_comp + s · ε ,  s ∈ scheduler grid    [renoise]
    x0_re   = Euler chain resumed from s, same ctx + KV          [re-dream]

x0_re is a model output: on-manifold by construction, seam-free because the sampler
smoothed the composite boundary, and context-consistent because the re-dream attended
to the same KV that the next frame will. Writing x0_re back into the cache is
therefore a legal move in exactly the sense Phase 8 proved raw pastes are not. The
decisive question (examples/warm_probe.py): does laundered write-back keep the
overlay's gain while restoring durability (re-anchoring, bounded |dx|), or does any
memory-bearing context edit — even an on-manifold one — still stall the dynamics?

Sigma choice: the schedule is [1.0, 0.9, 0.75, 0.3, 0.0]. Renoising to a GRID sigma
means the re-dream is the exact tail of the training-matched schedule (no off-grid
sigma the model never saw): s=0.3 → one Euler step (light touch, memory mostly
kept), s=0.75 → two steps (deep re-dream, context gets more votes).

Registration safety rules are inherited unchanged from rigid.py: pose only SELECTS
the keyframe (zero net translation — the applied shift is appearance-computed), and
a weak response or implausible shift means no re-dream this frame (evidence-gated).

Torch-touching code is confined to partial_denoise + redream_step; the pure helpers
are CPU-testable without a model (examples/test_warmstart.py).
"""
from __future__ import annotations

import sys
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, "examples")
from slam import roi_of                                            # noqa: E402
from rigid import RigidStore, register, valid_cols                 # noqa: E402


# --------------------------------------------------------------------------- #
# pure schedule / renoise math (CPU, model-free)
# --------------------------------------------------------------------------- #
def resume_index(sigmas, s: float, tol: float = 1e-4) -> int:
    """Index j of the grid sigma the re-dream resumes from. `s` must be a mid-schedule
    grid value: strictly inside (0, sigmas[0]) — resuming from sigmas[0] would discard
    the proposal entirely and from 0.0 there is nothing left to run. Snaps to the
    nearest grid sigma; raises if `s` is not on the grid (the model was only ever
    sampled at grid sigmas — an off-grid renoise level is a silent OOD input)."""
    vals = [float(v) for v in sigmas]
    j = min(range(len(vals)), key=lambda i: abs(vals[i] - s))
    if abs(vals[j] - s) > tol:
        raise ValueError(f"sigma {s} not on scheduler grid {vals}")
    if j == 0 or vals[j] == 0.0:
        raise ValueError(f"resume sigma must be mid-schedule, got {vals[j]}")
    return j


def renoise(x0, eps, s: float):
    """Flow-matching forward map to sigma s: x_s = (1-s)·x0 + s·eps (float math)."""
    return (1.0 - s) * x0 + s * eps


# --------------------------------------------------------------------------- #
# torch side: partial Euler chain + the engine splice
# --------------------------------------------------------------------------- #
import torch  # noqa: E402  (kept below the pure section, like rigid.py)

from rigid import band_mask, frac_roll  # noqa: E402,F401  (band_mask re-exported)


@torch.inference_mode()
def partial_denoise(engine, x, ctx, j: int):
    """Resume WorldEngine's deterministic Euler ODE from scheduler_sigmas[j] down to
    0 under the SAME prepared ctx and frozen KV cache. Eager mirror of
    engine._denoise_pass (which is fullgraph-compiled and always starts at sigma[0])."""
    sig = engine.scheduler_sigmas
    engine.kv_cache.set_frozen(True)
    sigma = x.new_empty((x.size(0), x.size(1)))
    for step_sig, step_dsig in zip(sig[j:], sig[j:].diff()):
        v = engine.model(x, sigma.fill_(step_sig), **ctx, kv_cache=engine.kv_cache)
        x = (x.float() + step_dsig.float() * v.float()).type_as(x)
    return x


@torch.inference_mode()
def redream_step(engine, ctrl, store: Optional[RigidStore], mask: torch.Tensor,
                 project: bool = False, capture: bool = False, sigma_re: float = 0.3,
                 lam: float = 1.0, retrieve: str = "nearest", resp_min: float = 0.05,
                 max_shift_frac: float = 0.35, write_back: bool = True):
    """One generated frame with optional memory re-dream, replicating
    WorldEngine.gen_frame with the propose→register→compose→renoise→re-dream splice.
    Returns (four_rgb_u8 [4,H,W,3], info dict) — same contract as rigid.rigid_step.

    write_back=True  caches x0_re (the laundered latent): the durability claim.
    write_back=False caches x0_prop while still scoring the re-dreamed decode —
    isolates the init/overlay component exactly as rigid_nowb isolated the paste.

    VAE stream discipline matches rigid_step: state is snapshotted before the
    provisional decode and rewound iff the re-dream is accepted, so exactly one
    decode is in effect per frame."""
    x = torch.randn(engine.frm_shape, device=engine.device, dtype=engine.dtype)
    inputs = engine.prep_inputs(x=x, ctrl=ctrl)
    x0_prop = engine._denoise_pass(x, inputs, engine.kv_cache).clone()

    info = {"projected": False, "dx_px": None, "resp": None, "reject": None,
            "kf_yaw": None, "wrote_back": False, "sigma_re": None}
    vst = engine.vae.get_state() if (project and store and len(store)) else None
    four = engine.vae.decode(x0_prop.squeeze(1))

    x0_final = x0_prop
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
            lat_w = x0_prop.shape[-1]
            dx_lat = dx * lat_w / img_w
            mem = kf.x0.to(engine.device).float().unsqueeze(0)   # [1,1,C,h,w]
            mem = frac_roll(mem, dx_lat)
            vc = torch.from_numpy(valid_cols(lat_w, dx_lat)).to(engine.device)
            m = (mask * lam) * vc.float().view(1, 1, 1, 1, -1)
            x0_comp = x0_prop.float() * (1.0 - m) + mem * m

            j = resume_index(engine.scheduler_sigmas, sigma_re)
            s = float(engine.scheduler_sigmas[j])
            eps = torch.randn_like(x0_comp)
            x_s = renoise(x0_comp, eps, s).to(engine.dtype)
            x0_re = partial_denoise(engine, x_s, inputs, j)

            info.update(projected=True, sigma_re=s)
            x0_final = x0_re
            engine.vae.load_state(vst)
            four = engine.vae.decode(x0_re.squeeze(1))

    x0_cache = x0_final if write_back else x0_prop
    info["wrote_back"] = bool(write_back and info["projected"])
    engine._cache_pass(x0_cache, inputs, engine.kv_cache)

    if capture and store is not None:
        gray = cv2.cvtColor(four[-1].detach().cpu().numpy(), cv2.COLOR_RGB2GRAY)
        store.insert(x0_final.squeeze(0).detach().to("cpu", copy=True),
                     roi_of(np.float32(gray)), float(engine.camera_yaw))
    return four, info
