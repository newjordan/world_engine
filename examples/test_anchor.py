"""CPU tests for Phase 10 pixel-space anchor helpers."""
import sys

sys.path.insert(0, "examples")

import cv2
import numpy as np
import pytest

from anchor import (AnchorStore, blend_u8, confidence_alpha, pixel_band_mask,
                    reconstruct_from_keyframe, repeat_as_x4, shift_rgb_x)


def _tex(seed=0, h=120, w=240):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(x, (0, 0), 2)


def test_store_novelty_and_query():
    s = AnchorStore(tau_insert=0.1)
    frames = repeat_as_x4(_tex())
    assert s.insert(frames, yaw=0.0)
    assert not s.insert(frames, yaw=0.05)
    assert s.insert(frames, yaw=1.0)
    assert s.query(0.2).yaw == 0.0
    assert s.query(0.2, mode="farthest").yaw == 1.0


def test_pixel_band_mask_shape_and_band():
    m = pixel_band_mask(100, 200, feather=0)
    assert m.shape == (100, 200, 1)
    assert float(m[45, 100, 0]) == pytest.approx(1.0)
    assert float(m[5, 5, 0]) == 0.0


def test_shift_rgb_x_moves_content_right():
    img = np.zeros((16, 32, 3), np.uint8)
    img[:, 10, :] = 255
    shifted = shift_rgb_x(img, 5.0)
    assert shifted[:, 15, :].max() == 255
    assert shifted[:, 10, :].max() == 0


def test_confidence_alpha_ramp():
    assert confidence_alpha(None) == 0.0
    assert confidence_alpha(0.05, resp_min=0.05, resp_full=0.55) == 0.0
    assert confidence_alpha(0.30, resp_min=0.05, resp_full=0.55) == pytest.approx(0.5)
    assert confidence_alpha(0.80, resp_min=0.05, resp_full=0.55) == 1.0


def test_blend_u8_endpoints_and_midpoint():
    a = np.full((2, 2, 3), 100, np.uint8)
    b = np.full((2, 2, 3), 200, np.uint8)
    assert np.array_equal(blend_u8(a, b, 1.0), a)
    assert np.array_equal(blend_u8(a, b, 0.0), b)
    assert np.array_equal(blend_u8(a, b, 0.25), np.full((2, 2, 3), 175, np.uint8))


def test_repeat_as_x4_contract():
    img = _tex(h=10, w=20)
    got = repeat_as_x4(img)
    assert got.shape == (4, 10, 20, 3)
    assert got.dtype == np.uint8
    assert np.array_equal(got[0], img)
    assert np.array_equal(got[3], img)


def test_reconstruct_from_keyframe_pastes_registered_band():
    base = _tex(seed=2)
    current = shift_rgb_x(base, 9.0)
    s = AnchorStore()
    assert s.insert(repeat_as_x4(base), yaw=0.0)
    recon, info = reconstruct_from_keyframe(s.query(0.0), current, feather=0)
    assert info["projected"], info
    assert abs(info["dx_px"] - 9.0) < 1.0
    band = pixel_band_mask(base.shape[0], base.shape[1], feather=0)[..., 0].astype(bool)
    mem_shifted = shift_rgb_x(base, info["dx_px"])
    # The certified band should move toward the registered memory view.
    before = np.mean(np.abs(current[band].astype(float) - mem_shifted[band].astype(float)))
    after = np.mean(np.abs(recon[band].astype(float) - mem_shifted[band].astype(float)))
    assert after < before * 0.25
