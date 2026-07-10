# uv run --dev pytest examples/test_kv_restamp.py -v
"""
CPU unit tests for the temporal keyframe RE-STAMP used by loop-closure permanence
(src/model/kv_cache.py: restamp_temporal_k / LayerKVCache.restamp_pins).

Loads kv_cache.py standalone (importlib) to avoid the package __init__ -> gemlite ->
CUDA import side effect.

The headline test is `test_restamp_equals_reposition`: re-stamping a key that was RoPE'd
at frame p1 must yield *exactly* the key you would have gotten by RoPE'ing the same raw
vector at frame p2 -- i.e. the re-stamp is a real move in time, and only in time.
"""
import importlib.util
import pathlib

import torch

_KV = pathlib.Path(__file__).resolve().parents[1] / "src" / "model" / "kv_cache.py"
_spec = importlib.util.spec_from_file_location("kv_cache_standalone", _KV)
kv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kv)

# Small RoPE geometry mirroring model/attn.py OrthoRoPE (d_head divisible by 8).
D_HEAD = 64
D_XY = D_HEAD // 8           # spatial freqs per axis
N_SPATIAL = 2 * D_XY         # x + y freq-pairs
D_T = D_HEAD // 4            # temporal freqs
HALF = D_HEAD // 2
assert N_SPATIAL + D_T == HALF

torch.manual_seed(0)
INV_T = torch.rand(D_T, dtype=torch.float32) + 0.05   # arbitrary positive temporal freqs
SPATIAL_FREQ = torch.rand(N_SPATIAL, dtype=torch.float32)  # arbitrary fixed x/y angles


def rope_apply(x, freqs):
    """OrthoRoPE forward (model/attn.py:60): de-interleave even/odd, rotate each pair by
    its freq angle, store as two concatenated halves [y0.. | y1..]."""
    cos, sin = freqs.cos(), freqs.sin()
    x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)   # even, odd -> [..., HALF]
    y0 = x0 * cos - x1 * sin
    y1 = x1 * cos + x0 * sin
    return torch.cat((y0, y1), dim=-1).to(x.dtype)


def freqs_at(t: float):
    """Full [HALF] freq vector for temporal position t (spatial part fixed)."""
    return torch.cat((SPATIAL_FREQ, t * INV_T))


def test_restamp_zero_is_identity():
    k = torch.randn(2, 3, D_HEAD)
    out = kv.restamp_temporal_k(k, 0, INV_T, N_SPATIAL)
    assert torch.allclose(out, k, atol=1e-6)


def test_restamp_roundtrip():
    k = torch.randn(4, D_HEAD)
    fwd = kv.restamp_temporal_k(k, 37, INV_T, N_SPATIAL)
    back = kv.restamp_temporal_k(fwd, -37, INV_T, N_SPATIAL)
    assert torch.allclose(back, k, atol=1e-5)
    assert not torch.allclose(fwd, k, atol=1e-3)  # it actually moved


def test_restamp_leaves_spatial_untouched():
    k = torch.randn(5, D_HEAD)
    out = kv.restamp_temporal_k(k, 19, INV_T, N_SPATIAL)
    # spatial freq-pairs live at indices [0, N_SPATIAL) within *each* stored half.
    assert torch.allclose(out[..., :N_SPATIAL], k[..., :N_SPATIAL], atol=1e-6)
    assert torch.allclose(out[..., HALF:HALF + N_SPATIAL], k[..., HALF:HALF + N_SPATIAL], atol=1e-6)
    # temporal part must have changed
    assert not torch.allclose(out[..., N_SPATIAL:HALF], k[..., N_SPATIAL:HALF], atol=1e-3)


def test_restamp_equals_reposition():
    """The crux: restamp(rope@p1, p2-p1) == rope@p2 for the same raw key."""
    raw = torch.randn(3, 7, D_HEAD)
    p1, p2 = 5, 123
    k_at_p1 = rope_apply(raw, freqs_at(p1))
    k_at_p2 = rope_apply(raw, freqs_at(p2))
    k_restamped = kv.restamp_temporal_k(k_at_p1, p2 - p1, INV_T, N_SPATIAL)
    assert torch.allclose(k_restamped, k_at_p2, atol=1e-4), \
        (k_restamped - k_at_p2).abs().max().item()


def test_restamp_preserves_relative_score():
    """Attention reads q.k; restamping the key by the SAME delta as the query's position
    advance must leave the q.k logit invariant (RoPE relativity). This is *why* an
    in-distribution re-stamp lets the frozen model attend to an old keyframe."""
    q_raw = torch.randn(1, D_HEAD)
    k_raw = torch.randn(1, D_HEAD)
    # query at frame 200 vs key roped at frame 100, then key restamped +100 (to 200-0).
    q = rope_apply(q_raw, freqs_at(200))
    k100 = rope_apply(k_raw, freqs_at(100))
    k_restamped = kv.restamp_temporal_k(k100, 100, INV_T, N_SPATIAL)     # now at frame 200
    # score(q@200, k@200) should equal score(q@0-shifted...) -> compare to both roped at same
    k200 = rope_apply(k_raw, freqs_at(200))
    assert torch.allclose((q * k_restamped).sum(), (q * k200).sum(), atol=1e-3)


def _make_layer(n_pin, tpf=128, l_frames=2):
    return kv.LayerKVCache(
        B=1, H=2, L=l_frames * tpf, Dh=D_HEAD, dtype=torch.float32,
        tokens_per_frame=tpf, pinned_dilation=1, n_pin_frames=n_pin)


def test_restamp_pins_noop_when_disabled():
    layer = _make_layer(n_pin=0)
    before = layer.kv.clone()
    layer.restamp_pins(INV_T, N_SPATIAL, target_f=50)   # must be a no-op
    assert torch.equal(layer.kv, before)


def test_layer_pin_then_restamp_moves_key_not_value():
    layer = _make_layer(n_pin=1)
    tpf = layer.tpf
    # Put a known raw frame roped at frame 100 into the tail (current) slot, plus a value.
    raw = torch.randn(1, 2, tpf, D_HEAD)
    k_roped = rope_apply(raw, freqs_at(100))
    val = torch.randn(1, 2, tpf, D_HEAD)
    layer.kv[0, :, :, layer.current_idx, :] = k_roped
    layer.kv[1, :, :, layer.current_idx, :] = val

    layer.pin_current(frame=100)
    lo = layer.pin_start
    hi = lo + tpf
    assert int(layer.pin_f[0]) == 100
    assert torch.allclose(layer.kv[0, :, :, lo:hi, :], k_roped, atol=1e-6)

    layer.restamp_pins(INV_T, N_SPATIAL, target_f=108)   # move memory +8 frames
    assert int(layer.pin_f[0]) == 108
    expected_k = rope_apply(raw, freqs_at(108))
    assert torch.allclose(layer.kv[0, :, :, lo:hi, :], expected_k, atol=1e-4)
    # Value untouched (RoPE / restamp only ever touches keys).
    assert torch.allclose(layer.kv[1, :, :, lo:hi, :], val, atol=1e-6)


def test_reset_clears_pin_timeline():
    layer = _make_layer(n_pin=2)
    layer.pin_current(slot=0, frame=42)
    assert int(layer.pin_f[0]) == 42
    layer.reset()
    assert int(layer.pin_f[0]) == -1


def test_restamp_temporal_pose_equals_reposition():
    """Full (x,y,t) pose restamp through the unified restamp_temporal_k with spatial
    xy supplied: restamp(k@pos1, dt, xy=..., dx_norm=...) == rope@pos2 for the
    same raw key, where pos2 = pos1 + (dx,dt)."""
    D = 128
    N_XY = D // 8          # 16
    N_X, N_Y, N_T = N_XY, N_XY, D // 4
    HALF = D // 2
    assert N_X + N_Y + N_T == HALF

    H, W = 16, 32
    NYQ = 1.0
    max_freq = min(H, W) * NYQ
    n = (N_XY + 1) // 2
    XY = (torch.linspace(1.0, max_freq / 2, n) * torch.pi).repeat_interleave(2)[:N_XY]
    THETA = 10000.0
    INV_T = 1.0 / (THETA ** (torch.arange(0, N_T, 2, dtype=torch.float32) / N_T))
    INV_T = INV_T.repeat_interleave(2)

    def angle_at(x_norm, y_norm, t):
        return torch.cat([x_norm * XY, y_norm * XY, t * INV_T])

    def rope_apply(x, freqs):
        cos, sin = freqs.cos(), freqs.sin()
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).to(x.dtype)

    raw = torch.randn(5, D)
    x1, y1, t1 = -0.5, -0.3, 50
    dyaw = 0.8
    dt = 30
    dx_norm = 2.0 * dyaw / W
    x2, y2, t2 = x1 + dx_norm, y1, t1 + dt

    k1 = rope_apply(raw, angle_at(x1, y1, t1))
    k2 = rope_apply(raw, angle_at(x2, y2, t2))
    k_restamped = kv.restamp_temporal_k(
        k1, dt, INV_T, N_X + N_Y,
        dx_norm=dx_norm, dy_norm=0.0, xy=XY, n_x_pairs=N_X, n_y_pairs=N_Y,
    )
    err = (k_restamped - k2).abs().max().item()
    assert err < 1e-4, err


def test_restamp_temporal_pose_leaves_y_pairs_untouched():
    """When dy_norm=0, the y freq-pairs must be unchanged (unified signature)."""
    D = 128
    N_XY = D // 8
    N_X, N_Y, N_T = N_XY, N_XY, D // 4
    HALF = D // 2
    H, W = 16, 32
    max_freq = min(H, W) * 1.0
    n = (N_XY + 1) // 2
    XY = (torch.linspace(1.0, max_freq / 2, n) * torch.pi).repeat_interleave(2)[:N_XY]
    INV_T = 1.0 / (10000.0 ** (torch.arange(0, N_T, 2, dtype=torch.float32) / N_T)).repeat_interleave(2)

    k = torch.randn(5, D)
    out = kv.restamp_temporal_k(
        k, 10, INV_T, N_X + N_Y,
        dx_norm=0.3, dy_norm=0.0, xy=XY, n_x_pairs=N_X, n_y_pairs=N_Y,
    )
    assert torch.allclose(out[..., N_X:N_X+N_Y], k[..., N_X:N_X+N_Y], atol=1e-6)
    assert torch.allclose(out[..., HALF+N_X:HALF+N_X+N_Y], k[..., HALF+N_X:HALF+N_X+N_Y], atol=1e-6)
    assert not torch.allclose(out[..., :N_X], k[..., :N_X], atol=1e-3)


def test_restamp_pose_wrapper_equals_reposition():
    """restamp_pose_k (thin wrapper over unified restamp_temporal_k) preserves the
    original behavior: restamp(rope@pos1, dyaw, dt) == rope@pos2."""
    D = 128
    N_XY = D // 8
    N_X, N_Y, N_T = N_XY, N_XY, D // 4
    H, W = 16, 32
    max_freq = min(H, W) * 1.0
    n = (N_XY + 1) // 2
    XY = (torch.linspace(1.0, max_freq / 2, n) * torch.pi).repeat_interleave(2)[:N_XY]
    INV_T = 1.0 / (10000.0 ** (torch.arange(0, N_T, 2, dtype=torch.float32) / N_T)).repeat_interleave(2)

    def angle_at(x_norm, y_norm, t):
        return torch.cat([x_norm * XY, y_norm * XY, t * INV_T])

    def rope_apply(x, freqs):
        cos, sin = freqs.cos(), freqs.sin()
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).to(x.dtype)

    raw = torch.randn(5, D)
    x1, y1, t1 = -0.5, -0.3, 50
    dyaw = 0.8
    dt = 30
    dx_norm = 2.0 * dyaw / W
    x2, y2, t2 = x1 + dx_norm, y1, t1 + dt

    k1 = rope_apply(raw, angle_at(x1, y1, t1))
    k2 = rope_apply(raw, angle_at(x2, y2, t2))
    k_restamped = kv.restamp_pose_k(k1, dx_norm, 0.0, dt, XY, INV_T, N_X, N_Y)
    err = (k_restamped - k2).abs().max().item()
    assert err < 1e-4, err


def test_restamp_pose_equals_reposition():
    """Full (x,y,t) pose restamp: restamp(rope@pos1, dyaw, dt) == rope@pos2 for the
    same raw key, where pos2 = pos1 shifted by the camera yaw delta and time delta."""
    # Model-like RoPE geometry (d_head=128, n_x=n_y=16, n_t=32)
    D = 128
    N_XY = D // 8          # 16
    N_X = N_XY
    N_Y = N_XY
    N_T = D // 4           # 32
    HALF = D // 2
    assert N_X + N_Y + N_T == HALF

    H, W = 16, 32
    NYQ = 1.0
    max_freq = min(H, W) * NYQ
    n = (N_XY + 1) // 2
    XY = (torch.linspace(1.0, max_freq / 2, n) * torch.pi).repeat_interleave(2)[:N_XY]
    THETA = 10000.0
    INV_T = 1.0 / (THETA ** (torch.arange(0, N_T, 2, dtype=torch.float32) / N_T))
    INV_T = INV_T.repeat_interleave(2)

    def angle_at(x_norm, y_norm, t):
        return torch.cat([x_norm * XY, y_norm * XY, t * INV_T])

    def rope_apply(x, freqs):
        cos, sin = freqs.cos(), freqs.sin()
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).to(x.dtype)

    raw = torch.randn(5, D)
    x1, y1, t1 = -0.5, -0.3, 50      # pose at pin time
    dyaw = 0.8                       # camera yaw-pixel-shift delta
    dt = 30                          # time delta in frames
    dx_norm = 2.0 * dyaw / W
    x2 = x1 + dx_norm
    y2 = y1  # no vertical shift

    k1 = rope_apply(raw, angle_at(x1, y1, t1))
    k2 = rope_apply(raw, angle_at(x2, y2, t1 + dt))
    k_restamped = kv.restamp_pose_k(k1, dx_norm, 0.0, dt, XY, INV_T, N_X, N_Y)
    err = (k_restamped - k2).abs().max().item()
    assert err < 1e-4, err


def test_restamp_pose_leaves_y_untouched_when_zero():
    """When dy_norm=0, the y freq-pairs must be unchanged."""
    D = 128
    N_XY = D // 8
    N_X, N_Y, N_T = N_XY, N_XY, D // 4
    HALF = D // 2
    H, W = 16, 32
    max_freq = min(H, W) * 1.0
    n = (N_XY + 1) // 2
    XY = (torch.linspace(1.0, max_freq / 2, n) * torch.pi).repeat_interleave(2)[:N_XY]
    THETA = 10000.0
    INV_T = 1.0 / (THETA ** (torch.arange(0, N_T, 2, dtype=torch.float32) / N_T)).repeat_interleave(2)

    k = torch.randn(5, D)
    out = kv.restamp_pose_k(k, 0.3, 0.0, 10, XY, INV_T, N_X, N_Y)
    # y pairs at indices [N_X, N_X+N_Y) in each half
    assert torch.allclose(out[..., N_X:N_X+N_Y], k[..., N_X:N_X+N_Y], atol=1e-6)
    assert torch.allclose(out[..., HALF+N_X:HALF+N_X+N_Y], k[..., HALF+N_X:HALF+N_X+N_Y], atol=1e-6)
    # x pairs must have changed
    assert not torch.allclose(out[..., :N_X], k[..., :N_X], atol=1e-3)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
