# uv run --dev pytest examples/test_rigid.py -v
"""
CPU tests for the rigid-projection math (examples/rigid.py) — registration,
fractional roll, mask geometry, store policy, and the composite itself. The engine
splice (rigid_step) is exercised by examples/rigid_probe.py on GPU; everything it
composes is validated here without a model.
"""
import sys

sys.path.insert(0, "examples")
import cv2
import numpy as np
import pytest
import torch

from rigid import RigidStore, band_mask, frac_roll, register, valid_cols
from slam import roi_of


def _tex(seed=0, h=720, w=1280):
    rng = np.random.default_rng(seed)
    t = rng.random((h, w)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), 3)
    return cv2.normalize(t, None, 40, 215, cv2.NORM_MINMAX)


# ------------------------------------------------------------------ registration
@pytest.mark.parametrize("shift", [-23, -5, 0, 7, 31])
def test_register_recovers_known_shift(shift):
    """current = keyframe scrolled right by dx: register must return dx."""
    tex = _tex()
    kf_roi = roi_of(tex)
    cur_roi = roi_of(np.roll(tex, shift, axis=1))
    dx, resp = register(kf_roi, cur_roi)
    assert abs(dx - shift) < 0.75, (dx, shift)
    assert resp > 0.2, resp


def test_register_rejects_unrelated_content_via_response():
    """Two unrelated views: the response gate is what says 'no certified
    registration' — it must be far below the matched-content response."""
    a, b = roi_of(_tex(seed=1)), roi_of(_tex(seed=2))
    _, resp_unrelated = register(a, b)
    _, resp_matched = register(a, roi_of(np.roll(_tex(seed=1), 9, axis=1)))
    assert resp_matched > 4 * resp_unrelated, (resp_matched, resp_unrelated)


# ------------------------------------------------------------------ roll geometry
def test_frac_roll_integer_matches_torch_roll():
    x = torch.randn(1, 1, 3, 8, 32)
    assert torch.equal(frac_roll(x, 5.0), torch.roll(x, 5, dims=-1))
    assert torch.equal(frac_roll(x, -3.0), torch.roll(x, -3, dims=-1))


def test_frac_roll_interpolates_between_integer_rolls():
    x = torch.randn(1, 1, 3, 8, 32)
    got = frac_roll(x, 2.25)
    want = torch.roll(x, 2, dims=-1) * 0.75 + torch.roll(x, 3, dims=-1) * 0.25
    assert torch.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("shift,bad", [(2.3, [0, 1, 2]), (-1.5, [30, 31]), (0.0, [])])
def test_valid_cols_masks_wrapped_strip(shift, bad):
    v = valid_cols(32, shift)
    assert sorted(np.where(~v)[0].tolist()) == bad
    assert v.sum() == 32 - len(bad)


# ------------------------------------------------------------------ mask geometry
def test_band_mask_covers_certified_band_and_feathers():
    m = band_mask(20, 56, device="cpu", feather=1)
    assert m.shape == (1, 1, 1, 20, 56)
    assert float(m[..., 10, 28]) == pytest.approx(1.0)   # band interior
    assert float(m[..., 0, 0]) == 0.0                    # far exterior
    edge = m[..., int(0.28 * 20) - 1: int(0.28 * 20) + 1, 28]
    assert 0.0 < float(edge.min()) < 1.0 or float(edge[..., 0]) == 0.0


# ------------------------------------------------------------------ store policy
def test_store_novelty_gate_and_retrieval_modes():
    s = RigidStore(tau_insert=0.1)
    assert s.insert(x0=None, roi=None, yaw=0.0)
    assert not s.insert(x0=None, roi=None, yaw=0.05)   # within tau: rejected
    assert s.insert(x0=None, roi=None, yaw=1.0)
    assert s.query(0.2).yaw == 0.0
    assert s.query(0.2, mode="farthest").yaw == 1.0
    assert RigidStore().query(0.0) is None


# ------------------------------------------------------------------ the composite
def test_hard_projection_replaces_band_only_where_valid():
    """lam=1: masked+valid cells become memory exactly; outside the band and inside
    the wrapped strip the provisional latent is untouched."""
    h, w = 20, 56
    x0 = torch.zeros(1, 1, 3, h, w)
    mem = torch.ones(1, 1, 3, h, w)
    mask = band_mask(h, w, device="cpu", feather=0)
    shift = 3.0
    vc = torch.from_numpy(valid_cols(w, shift)).float().view(1, 1, 1, 1, -1)
    m = mask * 1.0 * vc
    out = x0 * (1 - m) + frac_roll(mem, shift) * m
    r = (int(0.28 * h) + int(0.64 * h)) // 2
    assert float(out[0, 0, 0, r, 28]) == 1.0            # band interior: memory
    assert float(out[0, 0, 0, r, 1]) == 0.0             # wrapped strip: untouched
    assert float(out[0, 0, 0, 0, 28]) == 0.0            # outside band: untouched
