from torch import Tensor
import torch
from torch import nn
from tensordict import TensorDict

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    BlockMask
)


def make_block_mask(T: int, L: int, written: torch.Tensor) -> BlockMask:
    """
    T: Q length for this frame
    L: KV capacity == written.numel()
    written: [L] bool, True where there is valid KV data.
    T and L must be exact multiples of the sparse block size; `written` must be
    block-aligned, i.e. each block is either all True or all False.
    """
    BS = _DEFAULT_SPARSE_BLOCK_SIZE

    if not torch.compiler.is_compiling():
        torch._check(T % BS == 0, f"T ({T}) must be a multiple of block size ({BS})")
        torch._check(L % BS == 0, f"L ({L}) must be a multiple of block size ({BS})")

    Q_blocks = T // BS
    KV_blocks = L // BS

    # [KV_blocks, BS]
    written_blocks = written.view(KV_blocks, BS)

    # For a valid block-aligned mask, each block is either all written or all empty.
    block_any = written_blocks.any(-1)
    if not torch.compiler.is_compiling():
        assert torch.equal(block_any, written_blocks.all(-1)), "written must be block-aligned"

    # Every KV block is a full block
    full_bm = block_any[None, :].expand(Q_blocks, KV_blocks)
    full_kv_num_blocks = full_bm.sum(dim=-1, dtype=torch.int32)[None, None].contiguous()
    full_kv_indices = full_bm.argsort(dim=-1, descending=True, stable=True).to(torch.int32)[None, None].contiguous()

    # No partial blocks at all.
    kv_num_blocks = torch.zeros((1, 1, Q_blocks), dtype=torch.int32, device=written.device)
    kv_indices = torch.zeros((1, 1, Q_blocks, KV_blocks), dtype=torch.int32, device=written.device)

    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=BS,
        mask_mod=None,
        seq_lengths=(T, L),
        compute_q_blocks=False,
    )


class LayerKVCache(nn.Module):
    """
    Ring-buffer KV cache with fixed capacity L (tokens) for history plus, optionally,
    `n_pin_frames` dedicated pin slots (object/scene permanence), plus one extra frame
    (tokens_per_frame) at the tail holding the current frame.

    Slot layout: [ ring (L) | pins (n_pin_frames*tpf) | tail (tpf) ]

    Pin slots are written only by an explicit `pin_current()` call and survive ring
    eviction, so a frame pinned there stays attendable indefinitely. With
    n_pin_frames == 0 the layout, buffers, and behavior are identical to the original
    ring cache (bit-for-bit), so pinning is opt-in and default-off.
    """

    def __init__(self, B, H, L, Dh, dtype, tokens_per_frame: int, pinned_dilation: int = 1,
                 n_pin_frames: int = 0):
        super().__init__()
        self.tpf = tokens_per_frame
        self.L = L
        self.n_pin_frames = n_pin_frames
        # slot regions: ring [0, L) | pins [pin_start, tail_start) | tail [tail_start, capacity)
        self.pin_start = L
        self.tail_start = L + n_pin_frames * self.tpf
        self.capacity = self.tail_start + self.tpf
        self.pinned_dilation = pinned_dilation
        self.num_buckets = (L // self.tpf) // self.pinned_dilation
        assert (L // self.tpf) % pinned_dilation == 0 and L % self.tpf == 0

        # KV buffer: [2, B, H, capacity, Dh]
        self.kv = nn.Buffer(
            torch.zeros(2, B, H, self.capacity, Dh, dtype=dtype),
            persistent=False,
        )

        # which slots have ever been written
        # tail slice [tail_start, capacity) always holds the current frame -> written.
        # ring and pin slots start empty (masked out until first written / pinned).
        written = torch.zeros(self.capacity, dtype=torch.bool)
        written[self.tail_start:] = True
        self.written = nn.Buffer(written, persistent=False)
        self._mask_written = nn.Buffer(torch.empty_like(written), persistent=False)

        # Precompute indices:
        #   frame_offsets: [0, 1, ..., tpf-1] (for ring / pin indexing)
        #   current_idx:   tail slice [tail_start, tail_start+tpf)
        self.frame_offsets = nn.Buffer(torch.arange(self.tpf, dtype=torch.long), persistent=False)
        self.current_idx = nn.Buffer(self.frame_offsets + self.tail_start, persistent=False)
        self._pin_slot = 0  # round-robin write pointer for pin_current()

    def reset(self):
        self.kv.zero_()
        self.written.zero_()
        self.written[self.tail_start:].fill_(True)
        self._pin_slot = 0

    @torch.inference_mode()
    def pin_current(self, slot: int = None):
        """Copy the current tail frame's (post-RoPE) KV into a pin slot so it survives
        ring eviction. No-op when n_pin_frames == 0. Runs eagerly between generation
        steps (a plain buffer mutation with static shapes -> no recompile)."""
        if self.n_pin_frames == 0:
            return
        if slot is None:
            slot = self._pin_slot
            self._pin_slot = (self._pin_slot + 1) % self.n_pin_frames
        dst = self.frame_offsets + (self.pin_start + slot * self.tpf)
        src = self.kv.index_select(3, self.current_idx)
        self.kv.index_copy_(3, dst, src)
        self.written[dst] = True

    def upsert(self, kv: Tensor, pos_ids: TensorDict, is_frozen: bool):
        """
        kv: [2, B, H, T, Dh] for a single frame (T = tokens_per_frame)
        t_pos: [B, T], all equal per frame (ignoring -1)
        """
        T = self.tpf
        f_pos = pos_ids["f_pos"]

        if not torch.compiler.is_compiling():
            torch._check(kv.size(3) == self.tpf, "KV cache expects exactly one frame per upsert")
            torch._check(f_pos.shape == (kv.size(1), T), "t_pos must be [B, T]")
            torch._check(self.tpf <= self.L, "frame longer than KV ring capacity")
            torch._check(self.L % self.tpf == 0, f"L ({self.L}) must be a multiple of tokens_per_frame ({self.tpf})")
            torch._check(self.kv.size(3) == self.capacity, "KV buffer too long (expected L + tokens_per_frame)")
            torch._check((f_pos >= 0).all().item(), "t_pos must be non-negative during inference")
            torch._check(((f_pos == f_pos[:, :1]).all()).item(), "t_pos must be constant within frame")

        frame_idx = f_pos[0, 0]

        # map frame_t to a bucket, each bucket owns T contiguous slots
        bucket = (frame_idx + (self.pinned_dilation - 1)) // self.pinned_dilation
        slot = bucket % self.num_buckets
        base = slot * T

        # indices in the ring for this frame: [T] in [0, L)
        ring_idx = self.frame_offsets + base

        # Always write current frame into the tail slice [L, L+T):
        # this is the "self-attention component" for the current frame.
        self.kv.index_copy_(3, self.current_idx, kv)

        write_step = (frame_idx.remainder(self.pinned_dilation) == 0)
        mask_written = self._mask_written
        mask_written.copy_(self.written)
        mask_written[ring_idx] = mask_written[ring_idx] & ~write_step
        bm = make_block_mask(T, self.capacity, mask_written)

        # Persist current frame into the ring for future queries when unfrozen.
        if not is_frozen:
            # Persist current frame into the ring for future queries.
            dst = torch.where(write_step, ring_idx, self.current_idx)
            self.kv.index_copy_(3, dst, kv)
            self.written[dst] = True

        k, v = self.kv.unbind(0)
        return k, v, bm


class StaticKVCache(nn.Module):
    def __init__(self, config, batch_size, dtype):
        super().__init__()

        self.tpf = config.height * config.width

        local_L = config.local_window * self.tpf
        global_L = config.global_window * self.tpf

        period = config.global_attn_period
        off = config.global_attn_offset % period

        # Object/scene permanence: allocate pin slots on global layers by default
        # (they carry the trained long-horizon attention); pin_all_layers extends to
        # local layers too. n_pin_frames == 0 (default) => no pins => original behavior.
        n_pin = int(config.get("n_pin_frames", 0))
        pin_all = bool(config.get("pin_all_layers", False))

        def is_global(layer_idx):
            return (layer_idx - off) % period == 0

        self.layers = nn.ModuleList([
            LayerKVCache(
                batch_size,
                config.n_kv_heads,
                global_L if is_global(layer_idx) else local_L,
                config.d_model // config.n_heads,
                dtype,
                self.tpf,
                config.global_pinned_dilation if is_global(layer_idx) else 1,
                n_pin_frames=n_pin if (is_global(layer_idx) or pin_all) else 0,
            )
            for layer_idx in range(config.n_layers)
        ])

        self._is_frozen = True

    def reset(self):
        for layer in self.layers:
            layer.reset()
        self._is_frozen = True

    @torch.inference_mode()
    def pin_current(self):
        """Pin the current frame in every layer that has pin slots (global layers by
        default). Layers with n_pin_frames == 0 are no-ops."""
        for layer in self.layers:
            layer.pin_current()

    @torch.inference_mode()
    def get_state(self):
        layers = [(layer.kv.detach().clone(), layer.written.detach().clone()) for layer in self.layers]
        return {"_is_frozen": self._is_frozen, "layers": layers}

    @torch.inference_mode()
    def load_state(self, state):
        self._is_frozen = bool(state.get("_is_frozen", True))
        for layer, (kv, written) in zip(self.layers, state["layers"]):
            layer.kv.copy_(kv)
            layer.written.copy_(written)

    def set_frozen(self, is_frozen: bool):
        self._is_frozen = is_frozen

    def upsert(self, k: Tensor, v: Tensor, pos_ids: TensorDict, layer: int):
        kv = torch.stack([k, v], dim=0)
        return self.layers[layer].upsert(kv, pos_ids, self._is_frozen)
