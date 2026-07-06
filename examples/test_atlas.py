# uv run --dev pytest examples/test_atlas.py -v
"""
CPU-only unit tests for the pose-indexed world atlas (examples/atlas.py, Phase 6).
No model, no GPU: pure retrieval geometry. Locks the insert/query/eviction math before
any GPU time is spent (docs/ATLAS_PLAN.md 6.1).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from atlas import AtlasStore, Keyframe, yaw_dist  # noqa: E402


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


# --------------------------------------------------------------------------- dist

def test_yaw_dist_linear():
    assert approx(yaw_dist(1.0, 4.0), 3.0)
    assert approx(yaw_dist(4.0, 1.0), 3.0)          # symmetric
    assert approx(yaw_dist(-1.0, 1.0), 2.0)
    assert approx(yaw_dist(2.5, 2.5), 0.0)


def test_yaw_dist_circular_wrap():
    P = 10.0
    assert approx(yaw_dist(0.5, P - 0.5, P), 1.0)   # wraps across the boundary
    assert approx(yaw_dist(P - 0.5, 0.5, P), 1.0)   # symmetric
    assert approx(yaw_dist(1.0, 4.0, P), 3.0)       # interior unchanged
    assert approx(yaw_dist(9.9, 0.2, P), 0.3)       # short way round, not 9.7


# ------------------------------------------------------------------------- insert

def test_insert_novelty_gate_dedups():
    s = AtlasStore(tau_insert=1.0)
    assert s.insert("a", yaw=0.0, frame_ts=0) is True
    assert s.insert("b", yaw=0.5, frame_ts=1) is False   # within tau of 0.0
    assert s.insert("c", yaw=0.9, frame_ts=2) is False   # still within tau
    assert s.insert("d", yaw=1.0, frame_ts=3) is True    # exactly tau away -> kept
    assert len(s) == 2
    assert s.n_inserted == 2
    assert s.n_skipped == 2


def test_insert_spaced_all_kept():
    s = AtlasStore(tau_insert=0.1)
    for i in range(10):
        assert s.insert(i, yaw=float(i), frame_ts=i) is True
    assert len(s) == 10
    assert s.n_skipped == 0


# -------------------------------------------------------------------------- query

def test_query_empty_returns_empty():
    assert AtlasStore().query(0.0) == []
    assert AtlasStore().query(0.0, k=4) == []


def test_query_nearest_and_order():
    s = AtlasStore(tau_insert=0.1)
    for y in (0.0, 1.0, 2.0, 3.0):
        s.insert(y, yaw=y, frame_ts=0)
    assert [kf.yaw for kf in s.query(2.4, k=1)] == [2.0]
    assert [kf.yaw for kf in s.query(2.6, k=1)] == [3.0]
    assert [kf.yaw for kf in s.query(2.4, k=2)] == [2.0, 3.0]     # nearest first
    assert [kf.yaw for kf in s.query(2.4, k=3)] == [2.0, 3.0, 1.0]


def test_query_circular_wrap_picks_wrapped_neighbor():
    s = AtlasStore(tau_insert=0.1, yaw_period=10.0)
    for y in (0.2, 5.0, 9.8):
        s.insert(y, yaw=y, frame_ts=0)
    # 9.9 is 0.1 from 9.8 the short way; the linear-nearest would wrongly consider 0.2.
    assert [kf.yaw for kf in s.query(9.9, k=1)] == [9.8]


def test_query_k_larger_than_store():
    s = AtlasStore(tau_insert=0.1)
    s.insert("a", yaw=0.0, frame_ts=0)
    s.insert("b", yaw=1.0, frame_ts=0)
    hits = s.query(0.0, k=10)
    assert len(hits) == 2


# ------------------------------------------------------------------------- payload

def test_kv_payload_is_opaque_and_roundtrips():
    s = AtlasStore(tau_insert=0.1)
    payload = {3: ("K3", "V3"), 7: ("K7", "V7")}   # stand-in for {layer: (K, V)}
    s.insert(payload, yaw=1.23, frame_ts=42)
    hit = s.query(1.23, k=1)[0]
    assert hit.kv is payload                        # identity preserved, not copied
    assert hit.frame_ts == 42
    assert approx(hit.yaw, 1.23)


# ------------------------------------------------------------------------ bounding

def test_bound_preserves_coverage():
    # Uniformly spaced inserts past capacity: the store must thin the interior and keep
    # the explored span (endpoints), not collapse to the most-recent n_max.
    s = AtlasStore(tau_insert=0.5, n_max=4)
    for i in range(10):
        s.insert(i, yaw=float(i), frame_ts=i)
    ys = sorted(kf.yaw for kf in s.keyframes)
    assert len(s) == 4
    assert s.n_inserted == 10
    assert s.n_evicted == 6
    assert approx(ys[0], 0.0)      # span endpoints survive
    assert approx(ys[-1], 9.0)


def test_bound_nmax_one_keeps_newest():
    s = AtlasStore(tau_insert=0.1, n_max=1)
    s.insert("old", yaw=0.0, frame_ts=0)
    s.insert("new", yaw=5.0, frame_ts=1)   # novel -> inserted, then evict back to 1
    assert len(s) == 1
    assert s.keyframes[0].kv == "new"
    assert s.n_evicted == 1


# --------------------------------------------------------------------------- stats

def test_hits_and_stats():
    s = AtlasStore(tau_insert=0.1)
    for y in (0.0, 2.0, 4.0):
        s.insert(y, yaw=y, frame_ts=0)
    s.query(0.0, k=1)
    s.query(0.1, k=1)                       # hits yaw 0.0 again
    hit0 = min(s.keyframes, key=lambda kf: kf.yaw)
    assert hit0.n_hits == 2
    st = s.stats()
    assert st["size"] == 3
    assert st["inserted"] == 3
    assert st["skipped"] == 0
    assert st["evicted"] == 0
    assert approx(st["yaw_span"], 4.0)
