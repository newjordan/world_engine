"""CPU tests for Phase 10 pixel-space anchor helpers."""
import sys

sys.path.insert(0, "examples")

import cv2
import numpy as np
import pytest

from anchor import (AnchorStore, blend_u8, confidence_alpha, confidence_map_mask,
                    full_frame_mask, micro_yaw_grid, nearest_micro_index,
                    pixel_band_mask, reconstruct_from_keyframe,
                    reconstruct_temporal_batch, repeat_as_x4, shift_rgb_x)


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


def test_micro_yaw_grid_tracks_chunk_direction():
    assert np.allclose(micro_yaw_grid(1.0, 0.4, n=4), [0.7, 0.8, 0.9, 1.0])
    assert np.allclose(micro_yaw_grid(1.0, -0.4, n=4), [1.3, 1.2, 1.1, 1.0])


def test_store_records_micro_yaws_and_selects_nearest_frame():
    s = AnchorStore(tau_insert=0.1)
    frames = np.stack([_tex(seed=i, h=30, w=60) for i in range(4)], axis=0)
    assert s.insert(frames, yaw=1.0, yaw_delta=0.4)
    kf = s.query(1.0)
    assert np.allclose(kf.micro_yaws, [0.7, 0.8, 0.9, 1.0])
    assert nearest_micro_index(kf, 0.88) == 2


def test_pixel_band_mask_shape_and_band():
    m = pixel_band_mask(100, 200, feather=0)
    assert m.shape == (100, 200, 1)
    assert float(m[45, 100, 0]) == pytest.approx(1.0)
    assert float(m[5, 5, 0]) == 0.0


def test_full_and_confidence_masks_cover_expected_regions():
    full = full_frame_mask(20, 40, dx=5.0)
    conf = confidence_map_mask(20, 40, dx=5.0, feather=4, floor=0.25)
    band = pixel_band_mask(20, 40, feather=0)
    assert full.shape == (20, 40, 1)
    assert conf.shape == (20, 40, 1)
    assert float(full[10, 20, 0]) == pytest.approx(1.0)
    assert float(conf[10, 20, 0]) > float(conf[1, 1, 0])
    assert float(conf[1, 1, 0]) >= 0.0
    assert float(conf[10, 20, 0]) <= 1.0
    assert float(band[1, 1, 0]) == 0.0


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


def test_reconstruct_from_keyframe_full_mode_replaces_nonwrap_frame():
    base = _tex(seed=4)
    current = shift_rgb_x(base, 7.0)
    band = pixel_band_mask(base.shape[0], base.shape[1], feather=0)[..., 0].astype(bool)
    current_damaged = current.copy()
    current_damaged[~band] = 0
    s = AnchorStore()
    assert s.insert(repeat_as_x4(base), yaw=0.0)

    band_recon, band_info = reconstruct_from_keyframe(
        s.query(0.0), current_damaged, feather=0, mask_mode="band")
    full_recon, full_info = reconstruct_from_keyframe(
        s.query(0.0), current_damaged, feather=0, mask_mode="full")
    assert band_info["projected"] and full_info["projected"]
    mem_shifted = shift_rgb_x(base, full_info["dx_px"])
    outside = ~band
    band_outside_err = np.mean(np.abs(band_recon[outside].astype(float) - mem_shifted[outside].astype(float)))
    full_outside_err = np.mean(np.abs(full_recon[outside].astype(float) - mem_shifted[outside].astype(float)))
    assert full_outside_err < band_outside_err * 0.25


def test_reconstruct_temporal_batch_uses_distinct_micro_frames():
    base = np.stack([_tex(seed=i, h=120, w=240) for i in range(4)], axis=0)
    current = np.stack([shift_rgb_x(frame, 7.0) for frame in base], axis=0)
    s = AnchorStore()
    assert s.insert(base, yaw=1.0, yaw_delta=0.4)
    target_yaws = micro_yaw_grid(1.0, 0.4, n=4)
    recon, infos = reconstruct_temporal_batch(
        s, current, target_yaws, feather=0, resp_min=0.02)

    assert recon.shape == current.shape
    assert [i["kf_frame_idx"] for i in infos] == [0, 1, 2, 3]
    assert all(i["projected"] for i in infos), infos

    band = pixel_band_mask(base.shape[1], base.shape[2], feather=0)[..., 0].astype(bool)
    for j, info in enumerate(infos):
        mem_shifted = shift_rgb_x(base[j], info["dx_px"])
        after = np.mean(np.abs(recon[j][band].astype(float) - mem_shifted[band].astype(float)))
        assert after < 1.0


def test_reconstruct_temporal_batch_full_mode_replaces_outside_band():
    base = np.stack([_tex(seed=i + 10, h=120, w=240) for i in range(4)], axis=0)
    current = np.stack([shift_rgb_x(frame, 7.0) for frame in base], axis=0)
    band = pixel_band_mask(base.shape[1], base.shape[2], feather=0)[..., 0].astype(bool)
    damaged = current.copy()
    damaged[:, ~band] = 0
    s = AnchorStore()
    assert s.insert(base, yaw=1.0, yaw_delta=0.4)
    target_yaws = micro_yaw_grid(1.0, 0.4, n=4)
    recon, infos = reconstruct_temporal_batch(
        s, damaged, target_yaws, feather=0, resp_min=0.02, mask_mode="full")
    assert all(i["projected"] for i in infos), infos
    outside = ~band
    for j, info in enumerate(infos):
        mem_shifted = shift_rgb_x(base[j], info["dx_px"])
        before = np.mean(np.abs(damaged[j][outside].astype(float) - mem_shifted[outside].astype(float)))
        err = np.mean(np.abs(recon[j][outside].astype(float) - mem_shifted[outside].astype(float)))
        assert err < before * 0.1


def test_reconstruct_temporal_batch_honors_farthest_retrieval():
    near = np.stack([_tex(seed=i + 20, h=120, w=240) for i in range(4)], axis=0)
    far = np.stack([_tex(seed=i + 30, h=120, w=240) for i in range(4)], axis=0)
    current = np.stack([shift_rgb_x(frame, 7.0) for frame in near], axis=0)
    s = AnchorStore(tau_insert=0.01)
    assert s.insert(near, yaw=0.0, yaw_delta=0.0)
    assert s.insert(far, yaw=10.0, yaw_delta=0.0)
    target_yaws = [0.0, 0.0, 0.0, 0.0]

    _near_recon, near_infos = reconstruct_temporal_batch(
        s, current, target_yaws, retrieve="nearest", feather=0, resp_min=0.02)
    _far_recon, far_infos = reconstruct_temporal_batch(
        s, current, target_yaws, retrieve="farthest", feather=0, resp_min=0.02)

    assert {i["kf_seq"] for i in near_infos} == {0}
    assert {i["kf_seq"] for i in far_infos} == {1}


def test_reconstruct_temporal_batch_empty_store_falls_back_to_current():
    current = np.stack([_tex(seed=i, h=20, w=40) for i in range(4)], axis=0)
    recon, infos = reconstruct_temporal_batch(AnchorStore(), current, [0.0, 0.0, 0.0, 0.0])
    assert np.array_equal(recon, current)
    assert [i["reject"] for i in infos] == ["empty"] * 4
