# uv run --dev pytest examples/test_atlas.py -v
"""
Pose-indexed world atlas (Phase 6; see docs/ATLAS_PLAN.md).

A growing (camera pose -> KV keyframe) store that gives Waypoint-1.5 unbounded,
revisitable scene memory. On revisit the engine retrieves the NEAREST stored pose and
restamps only the small residual into the pin slots -> the restamp primitive stays in
the small-delta regime where it is valid (temporal restamp is exact; the spatial
residual -> 0 as retrieval improves). Static pinning of a single far keyframe was a
no-op (Phase 3) because it forced an out-of-distribution position; the atlas structurally
avoids large deltas by always having a nearby keyframe on record.

MVP scope: yaw-only (1-DOF) pose key, matching the engine's dead-reckoned camera_yaw
and the pure-yaw permanence benchmark. Translation (x, y) is Phase 6.6.

This module is deliberately model-agnostic and GPU-free: a keyframe's KV payload is an
opaque object (whatever pin_current copies out of the cache). Everything here is pure
retrieval geometry, fully CPU-testable without a model. The engine wiring lives in
WorldEngine.atlas_capture / atlas_activate (Phase 6.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


def yaw_dist(a: float, b: float, period: Optional[float] = None) -> float:
    """Distance between two yaw values.

    period=None  -> linear |a - b|. The MVP default: camera_yaw is an unwrapped
                    mouse-unit accumulator (sum of mouse-x), and benchmark excursions
                    never span a full revolution, so there is no wrap to model.
    period=P     -> circular distance with wrap-around, once units-per-revolution is
                    calibrated (Phase 6.6). E.g. yaw_dist(0.5, P - 0.5, P) == 1.0.
    """
    d = a - b
    if period is not None:
        d = (d + period / 2.0) % period - period / 2.0
    return abs(d)


@dataclass
class Keyframe:
    """One stored view: a pose tag plus an opaque KV payload."""
    yaw: float
    frame_ts: int
    kv: Any             # opaque per-layer KV payload, e.g. {layer_idx: (K, V)}
    n_hits: int = 0     # times retrieved (stats + eviction tie-break)
    seq: int = 0        # insertion order (provenance + eviction tie-break)


class AtlasStore:
    """Pose-indexed keyframe store, bounded by n_max.

    Write policy -- novelty-gated insert (the SLAM keyframe-insertion rule): skip a
    capture whose pose is within `tau_insert` of an existing keyframe, so the store
    holds distinct views, not near-duplicates.

    Read policy -- k-nearest by yaw distance.

    Bounding -- coverage-preserving eviction: when over n_max, drop the most redundant
    keyframe (the interior member of the densest neighborhood), never a span endpoint,
    so the stored views stay spread across the explored yaw range instead of collapsing
    to a recency window.
    """

    def __init__(self, tau_insert: float = 0.05, n_max: int = 256,
                 yaw_period: Optional[float] = None):
        assert tau_insert >= 0.0, tau_insert
        assert n_max >= 1, n_max
        self.tau_insert = tau_insert
        self.n_max = n_max
        self.yaw_period = yaw_period
        self.keyframes: List[Keyframe] = []
        # lifetime counters (stats / no-silent-truncation)
        self.n_inserted = 0     # accepted inserts
        self.n_skipped = 0      # rejected by the novelty gate
        self.n_evicted = 0      # dropped by bounding
        self._seq = 0

    def __len__(self) -> int:
        return len(self.keyframes)

    def _nearest(self, yaw: float) -> Optional[Keyframe]:
        if not self.keyframes:
            return None
        return min(self.keyframes,
                   key=lambda kf: yaw_dist(kf.yaw, yaw, self.yaw_period))

    def insert(self, kv: Any, yaw: float, frame_ts: int) -> bool:
        """Novelty-gated insert. Returns True iff a new keyframe was stored."""
        near = self._nearest(yaw)
        if near is not None and yaw_dist(near.yaw, yaw, self.yaw_period) < self.tau_insert:
            self.n_skipped += 1
            return False
        self.keyframes.append(Keyframe(yaw=yaw, frame_ts=frame_ts, kv=kv, seq=self._seq))
        self._seq += 1
        self.n_inserted += 1
        if len(self.keyframes) > self.n_max:
            self._evict_densest()
        return True

    def query(self, yaw: float, k: int = 1, mode: str = "nearest") -> List[Keyframe]:
        """Up to k keyframes by yaw distance, closest first. Bumps each hit's n_hits.

        mode="nearest" (default) is the real retrieval. mode="farthest" retrieves the
        most DISTANT keyframes instead — a falsification control: if attending the
        farthest (wrong-pose) memory helps as much as the nearest, the benefit is just
        "extra attended KV", not pose-indexing."""
        if not self.keyframes:
            return []
        ordered = sorted(self.keyframes,
                         key=lambda kf: yaw_dist(kf.yaw, yaw, self.yaw_period),
                         reverse=(mode == "farthest"))
        hits = ordered[:max(1, k)]
        for kf in hits:
            kf.n_hits += 1
        return hits

    def _evict_densest(self) -> None:
        """Remove one keyframe to get back to n_max, preserving yaw coverage.

        We sort by yaw and remove the point in the densest neighborhood (smallest
        left+right neighbor gap). In the linear (period=None) case the two span
        endpoints are excluded so the explored range is never shrunk; in the circular
        case every point is a ring-interior point. Tie-break: fewer hits, then older.
        """
        kfs = self.keyframes
        n = len(kfs)
        order = sorted(range(n), key=lambda i: kfs[i].yaw)  # indices in yaw order
        ys = [kfs[i].yaw for i in order]

        if self.yaw_period is None:
            candidates = range(1, n - 1)                    # interior only
            def gap(pos: int) -> float:
                return (ys[pos] - ys[pos - 1]) + (ys[pos + 1] - ys[pos])
        else:
            candidates = range(n)                           # ring: no endpoints
            def gap(pos: int) -> float:
                left = yaw_dist(ys[pos], ys[(pos - 1) % n], self.yaw_period)
                right = yaw_dist(ys[(pos + 1) % n], ys[pos], self.yaw_period)
                return left + right

        cand = list(candidates)
        if not cand:  # n_max == 1 (linear, n == 2): coverage is moot -> keep the newer
            cand = list(range(n))
            def gap(pos: int) -> float:  # fall back to nearest-neighbor redundancy
                return min(abs(ys[pos] - ys[j]) for j in range(n) if j != pos)

        best = min(cand, key=lambda pos: (gap(pos),
                                          kfs[order[pos]].n_hits,
                                          kfs[order[pos]].seq))
        self.keyframes.pop(order[best])
        self.n_evicted += 1

    def stats(self) -> dict:
        """Snapshot for the /stats overlay and results logging."""
        span = 0.0
        if self.keyframes:
            ys = [kf.yaw for kf in self.keyframes]
            span = max(ys) - min(ys)
        return {
            "size": len(self.keyframes),
            "inserted": self.n_inserted,
            "skipped": self.n_skipped,
            "evicted": self.n_evicted,
            "yaw_span": span,
        }


# ======================= engine glue (Phase 6.2) ============================
# The functions below bridge an AtlasStore to a live WorldEngine's KV cache. They are
# the only torch-touching code in this module; the store above stays torch-free so its
# retrieval math is testable without a model. All three are eager buffer mutations that
# run between compiled generation steps (static shapes -> no recompile), exactly like
# WorldEngine.pin_frame / restamp_memory.
#
# The atlas uses the engine's existing pin slots as a fixed-size *working set*: the
# unbounded store lives in `store` (CPU), and each activate pages the k retrieved
# keyframes into k pin slots with their own (frame_ts, yaw) provenance, so the existing
# LayerKVCache.restamp_pins applies the correct PER-KEYFRAME residual. The store's
# keyframes are re-loaded fresh every activate (restamp composes rotations and is not
# idempotent — always restamp from the pristine captured KV).
import torch  # noqa: E402  (kept below the pure-store section)


def _pin_layers(kv_cache):
    """Yield (layer_idx, layer) for cache layers that own pin slots (global layers by
    default). Layers with n_pin_frames == 0 are skipped."""
    for li, layer in enumerate(kv_cache.layers):
        if layer.n_pin_frames > 0:
            yield li, layer


def atlas_capture(engine, store, yaw: Optional[float] = None) -> bool:
    """Snapshot the current (just-generated) frame's pin-capable-layer KV into `store`,
    tagged with the engine's dead-reckoned camera yaw and frame index. Call right after
    the gen_frame / append_frame whose content should be remembered — same timing as
    engine.pin_frame(). Returns True iff a new keyframe was stored (novelty gate).

    `yaw` overrides the pose tag (default: engine.camera_yaw). Used by the SLAM-keyed
    spin arm (examples/slam.py) to key the store in CONTENT space (visual-odometry px)
    instead of command space — same store geometry, different coordinate system."""
    payload = {}
    for li, layer in _pin_layers(engine.kv_cache):
        # tail slice holds the current frame's post-RoPE KV: [2, B, H, tpf, Dh]
        frame_kv = layer.kv.index_select(3, layer.current_idx)
        payload[li] = frame_kv.detach().to("cpu", copy=True)
    if not payload:
        return False  # engine has no pin slots (n_pin_frames == 0)
    if yaw is None:
        yaw = float(engine.camera_yaw)
    return store.insert(payload, yaw=float(yaw), frame_ts=int(engine.frame_ts) - 1)


def _clear_pins(kv_cache):
    """Mark every pin slot empty (masked out of attention) and drop its provenance, so
    the next activate starts from a clean, deterministic working set."""
    for _, layer in _pin_layers(kv_cache):
        layer.written[layer.pin_start:layer.tail_start] = False
        layer.pin_f.fill_(-1)
        layer.pin_yaw = [None] * max(layer.n_pin_frames, 1)


def atlas_activate(engine, store, offset: int = 8, k: int = 1,
                   align_pose: bool = True, retrieve: str = "nearest",
                   yaw: Optional[float] = None) -> int:
    """Retrieve the k nearest stored keyframes to the current heading, page them (fresh,
    un-rotated) into the pin slots, and restamp each by ITS OWN residual to a recent
    in-distribution offset aligned to the current yaw. Call just before gen_frame during
    a revisit. `retrieve` is passed to AtlasStore.query ('nearest' | 'farthest', the
    latter a control). Returns the number of keyframes activated (0 if the store empty).

    `yaw` overrides the retrieval query pose (default: engine.camera_yaw), for stores
    keyed in a different coordinate system (e.g. SLAM content-space px). The spatial
    restamp target (align_pose=True) still uses engine.camera_yaw — RoPE residuals are
    in command units — so an overridden-key store should use align_pose=False unless
    its keys share the engine's yaw units (retrieval is what does the aligning; see
    ATLAS_RESULTS.md ablation: spatial restamp is a no-op at atlas-small residuals)."""
    kv_cache = engine.kv_cache
    if yaw is None:
        yaw = float(engine.camera_yaw)
    hits = store.query(float(yaw), k=k, mode=retrieve)
    _clear_pins(kv_cache)  # deterministic active set: only this frame's retrieval
    for slot, hit in enumerate(hits):
        for li, layer in _pin_layers(kv_cache):
            if slot >= layer.n_pin_frames:
                break
            src = hit.kv[li].to(layer.kv.device, dtype=layer.kv.dtype)
            dst = layer.frame_offsets + (layer.pin_start + slot * layer.tpf)
            layer.kv.index_copy_(3, dst, src)
            layer.written[dst] = True
            layer.pin_f[slot] = int(hit.frame_ts)
            layer.pin_yaw[slot] = float(hit.yaw)
    if not hits:
        return 0
    target_f = int(engine.frame_ts) - offset
    if align_pose:
        kv_cache.restamp_memory(
            engine._rope_inv_t, engine._n_spatial_pairs, target_f,
            xy=engine._rope_xy, target_yaw=float(engine.camera_yaw),
            width=engine._lat_width, height=engine._lat_height)
    else:
        kv_cache.restamp_memory(engine._rope_inv_t, engine._n_spatial_pairs, target_f)
    return len(hits)
