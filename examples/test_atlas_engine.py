# uv run --dev pytest examples/test_atlas_engine.py -v
"""
CPU-only mechanism tests for the atlas<->engine glue (examples/atlas.py, Phase 6.2).

No model, no GPU: LayerKVCache is loaded standalone via importlib (like test_ae_state.py)
to avoid the package __init__ -> gemlite -> CUDA import side effect, and the WorldEngine
is replaced by a tiny fake exposing only the attributes the glue reads. The headline
property proved here:

    atlas_capture(frame) -> store -> atlas_activate  ==  native pin_current + restamp

i.e. paging a stored keyframe into a pin slot and restamping it is bit-identical to the
engine's native pin+restamp of that same frame. That equivalence is what lets the atlas
reuse the (already validated) restamp path unchanged.
"""
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from atlas import AtlasStore, atlas_capture, atlas_activate, _clear_pins  # noqa: E402

# Load LayerKVCache standalone (kv_cache.py has no relative imports -> no CUDA/gemlite).
_KV = pathlib.Path(__file__).resolve().parents[1] / "src" / "model" / "kv_cache.py"
_spec = importlib.util.spec_from_file_location("kv_cache_standalone", _KV)
_kv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kv)
LayerKVCache = _kv.LayerKVCache

# --- tiny RoPE geometry (values are arbitrary; equivalence must hold for any) ---
D_HEAD = 64
HALF = D_HEAD // 2          # 32
N_SPATIAL = 16              # 8 x-pairs + 8 y-pairs
N_T = HALF - N_SPATIAL      # 16 temporal pairs
TPF = 4                     # tiny frame (no block-mask path is exercised here)

torch.manual_seed(0)
INV_T = torch.rand(N_T)
XY = torch.rand(N_SPATIAL // 2)   # x and y share the same n_spatial//2 = 8 freqs


def make_layer(n_pin_frames=2):
    return LayerKVCache(B=1, H=1, L=2 * TPF, Dh=D_HEAD, dtype=torch.float32,
                        tokens_per_frame=TPF, pinned_dilation=1,
                        n_pin_frames=n_pin_frames)


def set_tail(layer, kv):
    """Write a known frame into the tail (current-frame) slot."""
    layer.kv.index_copy_(3, layer.current_idx, kv)


def fake_engine(layer, camera_yaw, frame_ts, width=32, height=16):
    """Minimal stand-in for WorldEngine exposing only what the glue reads."""
    def restamp_memory(inv_t, n_sp, target_f, xy=None, target_yaw=None,
                       width=None, height=None):
        layer.restamp_pins(inv_t, n_sp, target_f, xy=xy, target_yaw=target_yaw,
                           width=width, height=height)
    kv_cache = SimpleNamespace(layers=[layer], restamp_memory=restamp_memory)
    return SimpleNamespace(kv_cache=kv_cache, camera_yaw=camera_yaw, frame_ts=frame_ts,
                           _rope_inv_t=INV_T, _n_spatial_pairs=N_SPATIAL, _rope_xy=XY,
                           _lat_width=width, _lat_height=height)


def pin_slot_k(layer, slot=0):
    """The K half of a pin slot's KV."""
    lo = layer.pin_start + slot * layer.tpf
    return layer.kv[0, :, :, lo:lo + layer.tpf, :].clone()


# --------------------------------------------------------------------- capture

def test_capture_stores_tail_kv_with_pose():
    layer = make_layer()
    kv = torch.randn(2, 1, 1, TPF, D_HEAD)
    set_tail(layer, kv)
    store = AtlasStore(tau_insert=0.0)
    eng = fake_engine(layer, camera_yaw=3.0, frame_ts=101)
    assert atlas_capture(eng, store) is True
    assert len(store) == 1
    kf = store.keyframes[0]
    assert kf.yaw == 3.0
    assert kf.frame_ts == 100                      # frame_ts - 1
    assert torch.equal(kf.kv[0], kv)               # layer 0 payload == the tail frame
    assert kf.kv[0].device.type == "cpu"


def test_capture_noop_without_pin_slots():
    layer = make_layer(n_pin_frames=0)
    set_tail(layer, torch.randn(2, 1, 1, TPF, D_HEAD))
    store = AtlasStore()
    eng = fake_engine(layer, camera_yaw=0.0, frame_ts=1)
    assert atlas_capture(eng, store) is False
    assert len(store) == 0


# ------------------------------------------------------------------ equivalence

def test_atlas_inject_restamp_equals_native_pin_restamp():
    kv = torch.randn(2, 1, 1, TPF, D_HEAD)
    pin_f, pin_yaw = 100, 3.0
    target_f, target_yaw = 108, 5.0

    # --- Path A: native pin_current + restamp_pins ---
    la = make_layer()
    set_tail(la, kv)
    la.pin_current(slot=0, frame=pin_f, yaw=pin_yaw)
    la.restamp_pins(INV_T, N_SPATIAL, target_f, xy=XY, target_yaw=target_yaw,
                    width=32, height=16)
    native_k = pin_slot_k(la, 0)

    # --- Path B: atlas_capture -> store -> atlas_activate ---
    lb = make_layer()
    set_tail(lb, kv)
    store = AtlasStore(tau_insert=0.0)
    eng = fake_engine(lb, camera_yaw=pin_yaw, frame_ts=pin_f + 1)   # capture @ yaw 3, ts 100
    atlas_capture(eng, store)
    eng.camera_yaw = target_yaw                                     # now looking at yaw 5
    eng.frame_ts = target_f + 8                                     # target_f = frame_ts - offset
    n = atlas_activate(eng, store, offset=8, k=1, align_pose=True)
    assert n == 1
    atlas_k = pin_slot_k(lb, 0)

    assert torch.equal(native_k, atlas_k), (native_k - atlas_k).abs().max().item()
    # provenance advanced to the restamp target in both paths
    assert int(lb.pin_f[0]) == target_f
    assert lb.pin_yaw[0] == target_yaw


def test_activate_reload_is_idempotent_across_frames():
    # Restamp composes rotations and is NOT idempotent; the glue must re-load pristine KV
    # each activate. Two consecutive activates to the SAME target must give the SAME key.
    kv = torch.randn(2, 1, 1, TPF, D_HEAD)
    layer = make_layer()
    set_tail(layer, kv)
    store = AtlasStore(tau_insert=0.0)
    eng = fake_engine(layer, camera_yaw=3.0, frame_ts=101)
    atlas_capture(eng, store)

    eng.camera_yaw, eng.frame_ts = 5.0, 116
    atlas_activate(eng, store, offset=8, k=1)
    k_first = pin_slot_k(layer, 0)

    atlas_activate(eng, store, offset=8, k=1)   # same target again
    k_second = pin_slot_k(layer, 0)
    assert torch.equal(k_first, k_second)       # no double-rotation drift


# ---------------------------------------------------------------------- inertness

def test_empty_store_activate_leaves_pins_unwritten():
    layer = make_layer()
    set_tail(layer, torch.randn(2, 1, 1, TPF, D_HEAD))
    store = AtlasStore()
    eng = fake_engine(layer, camera_yaw=1.0, frame_ts=50)
    n = atlas_activate(eng, store, offset=8, k=1)
    assert n == 0
    # pin region stays masked out (inert) -> attention is identical to no-atlas
    assert not layer.written[layer.pin_start:layer.tail_start].any()


def test_clear_pins_resets_working_set():
    layer = make_layer()
    kv = torch.randn(2, 1, 1, TPF, D_HEAD)
    set_tail(layer, kv)
    store = AtlasStore(tau_insert=0.0)
    eng = fake_engine(layer, camera_yaw=2.0, frame_ts=41)
    atlas_capture(eng, store)
    eng.camera_yaw, eng.frame_ts = 2.0, 41
    atlas_activate(eng, store, offset=8, k=1)
    assert layer.written[layer.pin_start:layer.pin_start + layer.tpf].all()
    _clear_pins(eng.kv_cache)
    assert not layer.written[layer.pin_start:layer.tail_start].any()
    assert int(layer.pin_f[0]) == -1
