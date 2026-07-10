"""CPU tests for anchor_pan_video's PNG frame spool + streaming MP4 writers."""
import json
import os
import sys

sys.path.insert(0, "examples")

import cv2
import numpy as np

import anchor_pan_video as apv


def _rgb(seed, h=36, w=64):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _flat(color, h=36, w=64):
    return np.full((h, w, 3), color, dtype=np.uint8)


def _res(arm, spool):
    return {"arm": arm, "video": spool, "away_steps": len(spool), "away_deg": 0.0,
            "anchor_info": {}, "post16_psnr": 21.7, "post64_psnr": None}


def _spool(root, n, seed0=0, phase="pan away"):
    s = apv.FrameSpool(root)
    for i in range(n):
        s.append(phase, float(i), _rgb(seed0 + i))
    return s


def _decoded_frame_count(path):
    cap = cv2.VideoCapture(path)
    assert cap.isOpened()
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def test_spool_roundtrip(tmp_path):
    s = apv.FrameSpool(tmp_path / "arm")
    frames = [_rgb(i) for i in range(5)]
    for i, f in enumerate(frames):
        s.append("return", 10.0 * i, f)
    assert len(s) == 5
    for i, f in enumerate(frames):
        entry = s[i]
        assert entry["phase"] == "return"
        assert entry["deg"] == 10.0 * i
        assert np.array_equal(entry["rgb"], f)


def test_spool_keeps_no_arrays_resident(tmp_path):
    s = apv.FrameSpool(tmp_path / "arm")
    for i in range(200):
        s.append("pan away", float(i), _flat(i % 256, h=360, w=640))
    assert len(s) == 200
    # O(1) by construction: metadata is paths + scalars, never pixel arrays.
    for m in s.meta:
        assert not any(isinstance(v, np.ndarray) for v in m.values())
    assert np.array_equal(s[137]["rgb"], _flat(137 % 256, h=360, w=640))


def test_single_mp4_streams(tmp_path):
    s = _spool(tmp_path / "tmp_frames" / "revisit", 12)
    path = str(tmp_path / "revisit.mp4")
    apv.write_single_mp4(path, _res("revisit", s), fps=30)
    assert os.path.getsize(path) > 0
    assert _decoded_frame_count(path) == 12


def test_single_mp4_cv2_fallback_streams(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio", None)  # force cv2 path
    s = _spool(tmp_path / "tmp_frames" / "revisit", 7)
    path = str(tmp_path / "revisit.mp4")
    apv.write_single_mp4(path, _res("revisit", s), fps=30)
    assert os.path.getsize(path) > 0
    assert _decoded_frame_count(path) == 7
    assert not os.path.exists(str(tmp_path / "revisit.tmp.mp4"))


def test_compare_pads_to_4_and_repeats_last(tmp_path, monkeypatch):
    grabbed = {}
    monkeypatch.setattr(apv, "write_mp4",
                        lambda path, make_frames, fps: grabbed.update(frames=list(make_frames())))
    colors = [(200, 30, 30), (30, 200, 30)]
    results = []
    for j, (color, n) in enumerate(zip(colors, [4, 2])):
        s = apv.FrameSpool(tmp_path / "tmp_frames" / f"arm{j}")
        for i in range(n):
            s.append("return", float(i), _flat(np.array(color) - 8 * i))
        results.append(_res(f"arm{j}", s))
    apv.write_compare_mp4(str(tmp_path / "compare_2x2.mp4"), results, fps=30)
    frames = grabbed["frames"]
    assert len(frames) == 4  # longest arm wins
    assert frames[0].shape == (1080, 1920, 3)

    def quad(f, qr, qc):  # panel-center pixel of quadrant (row, col)
        return f[540 * qr + 270, 960 * qc + 480]

    for i, f in enumerate(frames):
        assert np.array_equal(quad(f, 0, 0), np.array(colors[0]) - 8 * i)
        arm1 = np.array(colors[1]) - 8 * min(i, 1)  # short arm repeats last frame
        for qr, qc in ((0, 1), (1, 0), (1, 1)):  # last arm padded into empty cells
            assert np.array_equal(quad(f, qr, qc), arm1)


def test_compare_mp4_streams_unequal_arms(tmp_path):
    results = [_res(f"arm{j}", _spool(tmp_path / "tmp_frames" / f"arm{j}", n, seed0=10 * j))
               for j, n in enumerate([6, 3])]
    path = str(tmp_path / "compare_2x2.mp4")
    apv.write_compare_mp4(path, results, fps=30)
    assert os.path.getsize(path) > 0
    assert _decoded_frame_count(path) == 6


def test_cadence_attempt_entries_are_json_serializable(tmp_path):
    info = {"projected": np.bool_(True), "anchor": np.bool_(True), "reject": None,
            "dx_px": np.float64(12.5), "resp": np.float32(0.41),
            "winner_seq": np.int64(7), "winner_yaw": np.float64(2.5),
            "drift": np.float64(-0.5), "winner_resp": np.float64(0.41),
            "runner_up_resp": None, "window": np.float64(3.0),
            "n_windowed_out": np.int64(0),
            "micro_infos": [{"dx_px": np.float64(1.0)}]}
    entry = apv._cadence_attempt(np.int64(12), np.float64(42.5), info, closure=np.bool_(False))
    dumped = json.loads(json.dumps(entry))
    assert dumped["b"] == 12
    assert dumped["deg"] == 42.5
    assert dumped["closure"] is False
    assert dumped["anchor"] is True
    assert dumped["winner_seq"] == 7
    assert dumped["runner_up_resp"] is None
    assert "micro_infos" not in dumped  # per-micro detail stays out of the manifest rows


def test_metric_text_appends_cadence_count_only_for_cadence_arms(tmp_path):
    res = _res("anchor4_full_reacq_pose02", _spool(tmp_path / "a", 1))
    res["anchor_info"] = {"anchor": True}
    assert apv.metric_text(res) == "p16 21.7dB | anchored"  # non-cadence arms unchanged
    cad = _res("anchor4_full_reacq_k4_pose02", _spool(tmp_path / "b", 1))
    cad["anchor_info"] = {"anchor": False, "reject": "nomatch"}
    cad["cadence_attempts"] = [{"anchor": True}, {"anchor": False, "reject": "nomatch"},
                               {"anchor": True, "closure": True}]
    assert apv.metric_text(cad) == "p16 21.7dB | reject nomatch | cad 2/3"


def test_cleanup_frames_default_and_keep(tmp_path):
    for keep in (False, True):
        spill = tmp_path / "tmp_frames" / "revisit"
        spill.mkdir(parents=True, exist_ok=True)
        (spill / "000000.png").write_bytes(b"x")
        apv.cleanup_frames(str(tmp_path), keep_frames=keep)
        assert (tmp_path / "tmp_frames").exists() == keep
