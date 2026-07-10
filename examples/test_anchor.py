"""CPU tests for Phase 10 pixel-space anchor helpers."""
import sys

sys.path.insert(0, "examples")

import cv2
import numpy as np
import pytest

from anchor import (AnchorStore, blend_u8, calibrate_px_per_yaw, confidence_alpha,
                    confidence_map_mask, full_frame_mask, micro_yaw_grid,
                    nearest_micro_index, pixel_band_mask, pose_plausible,
                    pose_window, reacq_reconstruct, reacquire,
                    reconstruct_from_keyframe, reconstruct_temporal_batch,
                    repeat_as_x4, shift_rgb_x)
from slam import roi_of


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


def test_anchor4_full_admission_variants_fail_closed():
    from anchor_probe import _admission_failure

    near = [
        {"projected": True, "dx_px": 52.0, "resp": 0.32},
        {"projected": True, "dx_px": 55.0, "resp": 0.31},
        {"projected": True, "dx_px": 53.0, "resp": 0.34},
        {"projected": True, "dx_px": 51.0, "resp": 0.30},
    ]
    far = [dict(i, dx_px=220.0, resp=0.19) for i in near]
    weak = [dict(i, resp=0.19) for i in near]
    partial = [dict(i) for i in near]
    partial[2]["projected"] = False
    pose_near = [
        dict(i, target_yaw=1.00 + 0.05 * n, kf_frame_yaw=1.02 + 0.05 * n)
        for n, i in enumerate(partial)
    ]
    pose_far = [
        dict(i, target_yaw=1.00 + 0.05 * n, kf_frame_yaw=4.00 + 0.05 * n)
        for n, i in enumerate(near)
    ]

    assert _admission_failure("anchor4_full_dx64", near) is None
    assert _admission_failure("anchor4_full_far", partial) is None
    assert _admission_failure("anchor4_full_dx96", far) == "admit_dx96"
    assert _admission_failure("anchor4_full_dx96_far", far) == "admit_dx96"
    assert _admission_failure("anchor4_full_resp25", weak) == "admit_resp25"
    assert _admission_failure("anchor4_full_dx96_resp25", partial) == "admit_partial"
    assert _admission_failure("anchor4_full_pose04", pose_near) is None
    assert _admission_failure("anchor4_full_pose04_far", pose_far) == "admit_pose04"


def test_reconstruct_temporal_batch_empty_store_falls_back_to_current():
    current = np.stack([_tex(seed=i, h=20, w=40) for i in range(4)], axis=0)
    recon, infos = reconstruct_temporal_batch(AnchorStore(), current, [0.0, 0.0, 0.0, 0.0])
    assert np.array_equal(recon, current)
    assert [i["reject"] for i in infos] == ["empty"] * 4


def _roi(rgb):
    return roi_of(np.float32(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))


def test_query_k_ranks_both_modes_and_clamps_k():
    s = AnchorStore(tau_insert=0.1)
    for yaw in (0.0, 1.0, 2.0, 3.0):
        assert s.insert(repeat_as_x4(_tex(seed=int(yaw), h=30, w=60)), yaw=yaw)
    assert [k.yaw for k in s.query_k(0.1, k=2, mode="min")] == [0.0, 1.0]
    assert [k.yaw for k in s.query_k(0.1, k=2, mode="max")] == [3.0, 2.0]
    assert [k.yaw for k in s.query_k(0.1, k=10, mode="min")] == [0.0, 1.0, 2.0, 3.0]
    assert [k.yaw for k in s.query_k(0.1)] == [0.0, 1.0, 2.0, 3.0]


def test_reacquire_picks_content_match_despite_wrong_dead_reckoned_yaw():
    # True content sits at yaw 2.0; dead reckoning drifted to 2.5, nearest a
    # different texture. Re-acquisition must pick the content match from the
    # pose-window pool (10 px/yaw panorama, everything inside the window).
    base = _tex(seed=1)
    s = AnchorStore(tau_insert=0.1)
    for y in (0.0, 1.0, 2.0):                                # seq 0-2, panorama
        assert s.insert(repeat_as_x4(shift_rgb_x(base, -10.0 * y)), yaw=y)
    assert s.insert(repeat_as_x4(_tex(seed=2)), yaw=2.8)     # seq 3, distractor
    current = shift_rgb_x(base, -20.0)
    res = reacquire(s, _roi(current), yaw=2.5, k=8, mode="min")
    assert res["reject"] is None
    assert res["winner"].seq == 2
    assert abs(res["winner_dx"]) < 1.0
    assert res["window"] == pytest.approx(3.0)
    assert res["n_windowed_out"] == 0
    assert res["px_per_yaw"] == pytest.approx(10.0, abs=0.5)
    assert res["plaus_residual"] == pytest.approx(0.5, abs=0.1)
    assert res["winner_resp"] > res["runner_up_resp"]
    assert res["n_candidates"] == 4


def test_reacquire_nomatch_when_no_candidate_passes_gates():
    base = _tex(seed=3)
    s = AnchorStore(tau_insert=0.1)
    assert s.insert(repeat_as_x4(base), yaw=0.0)
    assert s.insert(repeat_as_x4(shift_rgb_x(base, -10.0)), yaw=1.0)
    # Calibration passes (10 px pair) but every candidate's shift exceeds the
    # 0.1 * W = 24 px gate -> nomatch via the content gates.
    current = shift_rgb_x(base, -50.0)
    res = reacquire(s, _roi(current), yaw=1.0, k=8, mode="min", max_shift_frac=0.1)
    assert res["winner"] is None
    assert res["reject"] == "nomatch"
    assert res["n_candidates"] == 2
    assert res["n_windowed_out"] == 0
    assert res["px_per_yaw"] is not None
    assert res["winner_resp"] is not None  # diagnostics still record the best match


def test_reacquire_empty_store_reports_empty():
    current = _tex(seed=5)
    res = reacquire(AnchorStore(), _roi(current), yaw=0.0)
    assert res["winner"] is None
    assert res["reject"] == "empty"
    assert res["n_candidates"] == 0


def test_reacquire_far_pool_rejects_entirely_through_the_window():
    a = _tex(seed=6)
    b = _tex(seed=7)
    s = AnchorStore(tau_insert=0.1)
    for y in (0.0, 0.5, 1.0):                   # seq 0-2, near panorama
        assert s.insert(repeat_as_x4(shift_rgb_x(a, -10.0 * y)), yaw=y)
    assert s.insert(repeat_as_x4(b), yaw=10.0)  # seq 3, far distractor
    current = shift_rgb_x(a, 7.0)
    res = reacquire(s, _roi(current), yaw=0.0, k=1, mode="max")
    assert res["winner"] is None
    assert res["reject"] == "nomatch"
    assert res["n_candidates"] == 1
    assert res["n_windowed_out"] == 1  # the whole far pool is outside the window
    assert res["winner_resp"] is None  # excluded candidates are never scored


def test_reacq_reconstruct_loop_closure_selects_content_keyframe():
    a = np.stack([_tex(seed=i, h=120, w=240) for i in range(4)], axis=0)
    b = np.stack([_tex(seed=i + 40, h=120, w=240) for i in range(4)], axis=0)
    s = AnchorStore(tau_insert=0.01)
    for y in (-1.0, 0.0):                              # seq 0-1, 10 px/yaw panorama
        part = np.stack([shift_rgb_x(f, -10.0 * (y - 1.0)) for f in a], axis=0)
        assert s.insert(part, yaw=y, yaw_delta=0.0)
    assert s.insert(a, yaw=1.0, yaw_delta=0.4)         # seq 2, true content match
    assert s.insert(b, yaw=1.8, yaw_delta=0.4)         # seq 3, nearest to dead yaw
    recon, infos, diag = reacq_reconstruct(
        s, a, yaw=1.5, k=8, mode="min", feather=0, resp_min=0.02)
    assert recon is not None
    assert diag["winner_seq"] == 2
    assert diag["drift"] == pytest.approx(-0.5)
    assert diag["window"] == pytest.approx(3.0)
    assert diag["n_windowed_out"] == 0
    assert diag["px_per_yaw"] == pytest.approx(10.0, abs=0.5)
    assert {i["kf_seq"] for i in infos} == {2}
    assert all(i["projected"] for i in infos), infos
    # pose reset: target yaws are the winner's stored micro-yaws, per-micro aligned.
    assert [i["kf_frame_idx"] for i in infos] == [0, 1, 2, 3]


def test_reacq_consistency_gate_fails_closed_on_dx_spread():
    from anchor_probe import _admission_failure

    tight = [
        dict(projected=True, dx_px=8.0 + n, resp=0.3, target_yaw=1.0 + 0.1 * n,
             kf_frame_yaw=1.0 + 0.1 * n)
        for n in range(4)
    ]
    spread = [dict(i) for i in tight]
    spread[3]["dx_px"] = 60.0
    assert _admission_failure("anchor4_full_reacq_pose02", tight) is None
    assert _admission_failure("anchor4_full_reacq_pose02", spread) == "admit_consist"
    assert _admission_failure("anchor4_full_reacq_pose02_far", spread) == "admit_consist"


def test_calibrate_px_per_yaw_recovers_synthetic_scale():
    base = _tex(seed=9)
    s = AnchorStore(tau_insert=0.1)
    for y in (0.0, 1.0, 2.0):
        assert s.insert(repeat_as_x4(shift_rgb_x(base, -10.0 * y)), yaw=y)
    ppy, n_pairs = calibrate_px_per_yaw(s)
    assert n_pairs == 2
    assert ppy == pytest.approx(10.0, abs=0.5)


def test_reacquire_window_excludes_identical_alias_beyond_window():
    # A pixel-identical alias at a far yaw would win any response ranking; the
    # pose window must drop it from the pool before scoring.
    base = _tex(seed=10)
    s = AnchorStore(tau_insert=0.1)
    for y in (0.0, 1.0, 2.0):                             # seq 0-2, panorama
        assert s.insert(repeat_as_x4(shift_rgb_x(base, -10.0 * y)), yaw=y)
    current = shift_rgb_x(base, -27.0)
    assert s.insert(repeat_as_x4(current), yaw=6.0)       # seq 3, alias of current
    res = reacquire(s, _roi(current), yaw=2.0, k=8, mode="min")
    from rigid import register
    alias_resp = register(s.keyframes[3].roi, _roi(current))[1]
    assert res["reject"] is None
    assert res["winner"].seq == 2
    assert res["window"] == pytest.approx(3.0)
    assert res["n_windowed_out"] == 1
    assert res["winner_resp"] < alias_resp


def test_reacquire_all_candidates_outside_window_rejects_nomatch():
    base = _tex(seed=11)
    s = AnchorStore(tau_insert=0.1)
    assert s.insert(repeat_as_x4(base), yaw=0.0)
    assert s.insert(repeat_as_x4(shift_rgb_x(base, -10.0)), yaw=1.0)
    # Content registers cleanly but dead reckoning is 4+ yaw units off -> the
    # whole pool falls outside the 3.0 window -> fail closed, nothing scored.
    current = shift_rgb_x(base, -10.0)
    res = reacquire(s, _roi(current), yaw=5.0, k=8, mode="min")
    assert res["winner"] is None
    assert res["reject"] == "nomatch"
    assert res["window"] == pytest.approx(3.0)
    assert res["n_windowed_out"] == 2
    assert res["winner_resp"] is None


def test_reacquire_uncalibratable_store_still_matches_in_window():
    base = _tex(seed=12)
    # Single keyframe: no calibration pair, but a window-passing content match
    # must still anchor (10g.2: calibration is diagnostics only, not a gate).
    s1 = AnchorStore()
    assert s1.insert(repeat_as_x4(base), yaw=0.0)
    res = reacquire(s1, _roi(base), yaw=0.0, k=8, mode="min")
    assert res["reject"] is None
    assert res["winner"].seq == 0
    assert res["px_per_yaw"] is None
    assert res["calib_pairs"] == 0
    assert res["plaus_residual"] is None
    # Zero command-yaw separation: still uncalibratable, still matches.
    s2 = AnchorStore(tau_insert=0.0)
    assert s2.insert(repeat_as_x4(base), yaw=0.0)
    assert s2.insert(repeat_as_x4(shift_rgb_x(base, 30.0)), yaw=0.0)
    res = reacquire(s2, _roi(base), yaw=0.0, k=8, mode="min")
    assert res["reject"] is None
    assert res["winner"].seq == 0
    assert res["px_per_yaw"] is None


def test_pose_plausibility_residuals_do_not_separate_pilot2():
    # bench_out/anchor_reacq/pilot2.csv: with the real store calibration
    # (ppy=31.431) the true match (dx 50.02, drift 0.2, residual 1.39) and the
    # far alias (dx 208.559, drift 5.2, residual 1.44) both pass at bound 1.5.
    # The residual has no discriminating power -> diagnostics only (10g.2).
    assert pose_plausible(50.02, 31.431, 0.2)
    assert pose_plausible(208.559, 31.431, 5.2)


def test_pose_window_is_spacing_aware():
    wide = AnchorStore(tau_insert=0.1)
    for y in (0.0, 4.0, 8.0, 12.0):
        assert wide.insert(repeat_as_x4(_tex(seed=13, h=30, w=60)), yaw=y)
    assert pose_window(wide) == pytest.approx(10.0)  # 2.5 x 4.0 spacing > 3.0
    narrow = AnchorStore(tau_insert=0.1)
    for y in (0.0, 0.2, 0.4, 0.6):
        assert narrow.insert(repeat_as_x4(_tex(seed=14, h=30, w=60)), yaw=y)
    assert pose_window(narrow) == pytest.approx(3.0)  # floor wins at 0.2 spacing
    # The 4.0-spacing store admits keyframes +/-8 from the query pose.
    res = reacquire(wide, _roi(_tex(seed=13, h=30, w=60)), yaw=4.0, k=8, mode="min")
    assert res["window"] == pytest.approx(10.0)
    assert res["n_windowed_out"] == 0


def test_reacquire_pilot2_far_candidate_excluded_regardless_of_resp():
    # Pilot2 regression: with the window resolved to 3.0, the far-control alias
    # at |kf.yaw - yaw| = 5.2 (which anchored under 10g.1) is excluded from the
    # pool no matter how strong its response.
    base = _tex(seed=15)
    s = AnchorStore(tau_insert=0.1)
    for y in (0.0, 0.5, 1.0, 1.5, 2.0):                   # seq 0-4, panorama
        assert s.insert(repeat_as_x4(shift_rgb_x(base, -10.0 * y)), yaw=y)
    current = shift_rgb_x(base, -22.0)
    assert s.insert(repeat_as_x4(current), yaw=7.2)       # seq 5, alias at drift 5.2
    res = reacquire(s, _roi(current), yaw=2.0, k=8, mode="min")
    from rigid import register
    alias_resp = register(s.keyframes[5].roi, _roi(current))[1]
    assert res["window"] == pytest.approx(3.0)
    assert res["n_windowed_out"] == 1
    assert res["reject"] is None
    assert res["winner"].yaw <= 2.0 + 3.0
    assert res["winner"].seq != 5
    assert alias_resp > 0.99  # a near-perfect response still cannot enter the pool
