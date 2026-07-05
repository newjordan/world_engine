# uv run --dev pytest examples/test_kv_pinning.py -v
"""
CPU unit tests for the Phase 3 KV frame-pinning mechanism (src/model/kv_cache.py).

These exercise the ring/pin/tail slot layout, the default-off bit-identity guarantee,
and that a pinned frame survives ring eviction. All CPU-only — no model weights, no
CUDA.

kv_cache.py is loaded standalone (importlib) rather than via `world_engine.model`
because importing the package runs src/__init__.py -> world_engine -> quantize ->
gemlite, which eagerly allocates on CUDA at import time. kv_cache.py itself has no
such dependency.
"""
import importlib.util
import pathlib

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

_KV = pathlib.Path(__file__).resolve().parents[1] / "src" / "model" / "kv_cache.py"
_spec = importlib.util.spec_from_file_location("kv_cache_standalone", _KV)
kvmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kvmod)
LayerKVCache = kvmod.LayerKVCache
StaticKVCache = kvmod.StaticKVCache

TPF = 128  # tokens per frame (must be a multiple of the 128 sparse block size)


def make_layer(window, dilation=1, n_pin_frames=0, B=1, H=2, Dh=4):
    return LayerKVCache(B, H, window * TPF, Dh, torch.float32, TPF,
                        pinned_dilation=dilation, n_pin_frames=n_pin_frames)


def frame_kv(val, B=1, H=2, Dh=4):
    """A frame whose every KV entry equals `val` (so we can identify it later)."""
    return torch.full((2, B, H, TPF, Dh), float(val))


def pos_for(frame_idx, B=1):
    return TensorDict({"f_pos": torch.full((B, TPF), frame_idx, dtype=torch.long)},
                      batch_size=[B, TPF])


def feed(cache, frame_idx, val, is_frozen=False):
    return cache.upsert(frame_kv(val), pos_for(frame_idx), is_frozen)


# --------------------------------------------------------------------------- #
# Layout / bit-identity (the default-off guarantee)
# --------------------------------------------------------------------------- #
def test_pin0_layout_is_original():
    """n_pin_frames=0 must reproduce the original ring+tail layout exactly."""
    L = 4 * TPF
    c = make_layer(window=4, n_pin_frames=0)
    assert c.pin_start == L and c.tail_start == L
    assert c.capacity == L + TPF
    assert torch.equal(c.current_idx, torch.arange(TPF) + L)
    assert c.written[:L].sum() == 0            # ring empty
    assert c.written[L:].all()                 # tail written


def test_pin_layout_regions():
    """With pins, capacity = L + P*tpf + tpf and pins start empty."""
    L, P = 4 * TPF, 3
    c = make_layer(window=4, n_pin_frames=P)
    assert c.pin_start == L
    assert c.tail_start == L + P * TPF
    assert c.capacity == L + (P + 1) * TPF
    assert torch.equal(c.current_idx, torch.arange(TPF) + c.tail_start)
    # ring + pins start unwritten; only tail is written
    assert c.written[: c.tail_start].sum() == 0
    assert c.written[c.tail_start:].all()


def test_unpinned_output_matches_baseline():
    """
    Pins allocated but never written => identical attention to the pin-free cache.
    Proof: identical KV contents at every attended (ring+tail) slot AND identical
    attend-set (pin blocks masked out) => flex_attention output is identical for any q.
    """
    c0 = make_layer(window=4, n_pin_frames=0)
    cP = make_layer(window=4, n_pin_frames=4)
    for f in range(12):                        # exceed the ring to force wraparound
        feed(c0, f, f + 1)
        feed(cP, f, f + 1)

    L = 4 * TPF
    # ring identical
    assert torch.equal(c0.kv[:, :, :, :L], cP.kv[:, :, :, :L])
    assert torch.equal(c0.written[:L], cP.written[:L])
    # tail identical (different offset, same content)
    tail0 = c0.kv[:, :, :, c0.tail_start: c0.tail_start + TPF]
    tailP = cP.kv[:, :, :, cP.tail_start: cP.tail_start + TPF]
    assert torch.equal(tail0, tailP)
    # pin region masked out entirely
    assert cP.written[cP.pin_start: cP.tail_start].sum() == 0


# --------------------------------------------------------------------------- #
# Pinning behavior
# --------------------------------------------------------------------------- #
def test_pinned_frame_survives_eviction():
    """A pinned frame stays present + attendable after the ring has fully cycled."""
    c = make_layer(window=4, dilation=1, n_pin_frames=1)  # horizon 4 latent frames
    feed(c, 0, 99.0)          # distinctive reference frame in the tail
    c.pin_current(0)          # pin it
    for f in range(1, 20):    # cycle the ring many times over -> frame 0 long evicted
        feed(c, f, f + 1.0)

    L = 4 * TPF
    # frame 0 (value 99) is gone from the ring...
    assert not (c.kv[0, 0, 0, :L, 0] == 99.0).any()
    # ...but preserved in the pin slot, and marked written (=> attendable)
    pin = slice(c.pin_start, c.pin_start + TPF)
    assert (c.kv[0, 0, 0, pin, 0] == 99.0).all()
    assert c.written[pin].all()

    # and the block mask fed to attention on the next step includes the pin block
    _, _, _ = feed(c, 20, 21.0)
    assert c._mask_written[pin].all()


def test_pin_roundrobin_and_reset():
    c = make_layer(window=4, n_pin_frames=2)
    feed(c, 0, 10.0); c.pin_current()   # -> slot 0
    feed(c, 1, 11.0); c.pin_current()   # -> slot 1
    feed(c, 2, 12.0); c.pin_current()   # -> slot 0 (wraps, overwrites 10.0)
    s0 = slice(c.pin_start, c.pin_start + TPF)
    s1 = slice(c.pin_start + TPF, c.pin_start + 2 * TPF)
    assert (c.kv[0, 0, 0, s0, 0] == 12.0).all()   # slot 0 overwritten by 3rd pin
    assert (c.kv[0, 0, 0, s1, 0] == 11.0).all()

    c.reset()
    assert c.written[: c.tail_start].sum() == 0    # pins + ring cleared
    assert c.written[c.tail_start:].all()
    assert c.kv.abs().sum() == 0
    assert c._pin_slot == 0


def test_pin0_is_noop():
    c = make_layer(window=4, n_pin_frames=0)
    feed(c, 0, 5.0)
    before = c.kv.clone()
    c.pin_current()               # must do nothing
    assert torch.equal(before, c.kv)


# --------------------------------------------------------------------------- #
# StaticKVCache: global-only pinning by default
# --------------------------------------------------------------------------- #
def _cfg(n_pin_frames=0, pin_all_layers=False):
    # mirrors the real Waypoint-1.5 attention layout (period 4, offset -1)
    return OmegaConf.create({
        "height": 8, "width": 16, "n_kv_heads": 2, "d_model": 64, "n_heads": 2,
        "n_layers": 12, "local_window": 4, "global_window": 8,
        "global_attn_period": 4, "global_attn_offset": -1, "global_pinned_dilation": 2,
        "n_pin_frames": n_pin_frames, "pin_all_layers": pin_all_layers,
    })


def test_static_pins_global_layers_only():
    cfg = _cfg(n_pin_frames=4)
    cache = StaticKVCache(cfg, batch_size=1, dtype=torch.float32)
    period, off = 4, (-1) % 4
    for i, layer in enumerate(cache.layers):
        is_global = (i - off) % period == 0
        assert layer.n_pin_frames == (4 if is_global else 0), f"layer {i}"


def test_static_pin_all_layers():
    cache = StaticKVCache(_cfg(n_pin_frames=4, pin_all_layers=True), 1, torch.float32)
    assert all(layer.n_pin_frames == 4 for layer in cache.layers)


def test_static_default_off_has_no_pins():
    cache = StaticKVCache(_cfg(), 1, torch.float32)
    assert all(layer.n_pin_frames == 0 for layer in cache.layers)
    assert all(layer.capacity == layer.L + layer.tpf for layer in cache.layers)


def test_static_pin_current_writes_only_global():
    cfg = _cfg(n_pin_frames=2)
    cache = StaticKVCache(cfg, 1, torch.float32)
    cache.set_frozen(False)
    tpf = cfg.height * cfg.width
    for i, layer in enumerate(cache.layers):
        kv = torch.full((2, 1, layer.kv.size(2), tpf, layer.kv.size(4)), 7.0)
        layer.upsert(kv, TensorDict({"f_pos": torch.zeros(1, tpf, dtype=torch.long)},
                                    batch_size=[1, tpf]), is_frozen=False)
    cache.pin_current()
    period, off = 4, (-1) % 4
    for i, layer in enumerate(cache.layers):
        pinned = layer.written[layer.pin_start: layer.tail_start].any()
        assert pinned == ((i - off) % period == 0), f"layer {i}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
