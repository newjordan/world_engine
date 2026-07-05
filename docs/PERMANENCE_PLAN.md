# Object Permanence in the Waypoint World Engine — Implementation Plan

**Goal:** deliver measured evidence of the model's scene-forgetting behavior, plus an
inference-side mitigation (KV frame pinning), as the foundation for a training-side
scene-memory proposal to the Overworld team.

**Executor notes:** this plan is self-contained; no conversation context needed.
Terminology: use **object/scene permanence** (standard literature term), not
"primitive permanence". This repo is the *inference* library for the pretrained
`Overworld/Waypoint-1.5-1B` checkpoint — the attention layout is baked into the
weights, so novel-architecture work lands in the training repo; everything here is
measurement + inference-side mechanism.

---

## Background: where permanence lives (and fails) in this codebase

- The engine (`src/world_engine.py`) generates one latent frame per `gen_frame()`
  call; each latent frame decodes to `temporal_compression` (4) RGB frames.
  `append_frame()` force-feeds a ground-truth frame into the cache.
- Memory is a **ring-buffer KV cache** (`src/model/kv_cache.py`):
  - Most layers keep the last `local_window` latent frames.
  - Every `global_attn_period`-th layer (offset `global_attn_offset`) keeps
    `global_window` frames thinned by `global_pinned_dilation` (only every
    N-th frame is written — `kv_cache.py:118-140`), i.e. a temporal horizon of
    `global_window × global_pinned_dilation` latent frames.
  - A frame that falls off its ring is overwritten → **forgotten**. That eviction
    is the permanence failure.
- **RoPE constraint (critical for Phase 3):** keys are cached *post-RoPE*
  (`src/model/attn.py:108-111`). A cached key permanently carries the absolute
  temporal phase of its `t_pos`. Relative query–key distances beyond the trained
  global horizon are out-of-distribution.
- Useful existing hooks: `engine.get_state()` / `engine.load_state()` snapshot and
  restore the full KV + timestamp state (`world_engine.py:122-130`).

---

## Phase 0 — Housekeeping (15 min)

1. Commit the pending README fix (`pipeline.append_frame` → `engine.append_frame`)
   as its own commit.
2. Delete `docs/ANGELX_HARNESS_GAUNTLET.md` and `angel_test_output/` — they
   describe a differentiable-rasterizer gauntlet (adjoints, material params)
   unrelated to this codebase, with unfilled placeholder results. **Confirmed for
   deletion by the repo owner via this plan.** Do not carry any of their content
   forward.
3. Create a working branch: `permanence-bench`.

## Phase 1 — Compute the memory horizons (30 min)

1. Load the config without weights:
   ```py
   from world_engine.model import WorldModel
   cfg = WorldModel.load_config("Overworld/Waypoint-1.5-1B")  # needs HF_TOKEN
   ```
2. Record: `local_window`, `global_window`, `global_attn_period`,
   `global_attn_offset`, `global_pinned_dilation`, `height`, `width`
   (`tokens_per_frame = height × width`), `temporal_compression`,
   `inference_fps`, `base_fps`, `n_layers`.
3. Compute and tabulate:
   - Local horizon: `local_window` latent frames → seconds =
     `local_window × temporal_compression / inference_fps`... verify: each latent
     frame spans `temporal_compression` RGB frames at `inference_fps` RGB fps.
   - Global horizon: `global_window × global_pinned_dilation` latent frames.
   - Which layer indices are global.
4. **Deliverable:** horizons table at the top of `docs/PERMANENCE_RESULTS.md`.
   These numbers pick the K-sweep in Phase 2.

## Phase 2 — Revisit-consistency benchmark (the core deliverable)

New file: `examples/permanence_bench.py` (examples/ is the sanctioned home for
non-library tooling; dev deps `opencv-python`, `imageio[pyav]`, `numpy`, `pytest`
are already in the `dev` group — run everything with `uv run --dev`).

### 2a. Pilot: validate the camera-return assumption (do this FIRST)

The whole benchmark rests on: *a mirrored pure-yaw trajectory returns the camera
to (approximately) its starting view.* Verify before building the sweep:

1. Follow `examples/gen_sample.py` for setup (engine init, seed frame). Download
   2–3 seed images from the Biome seeds repo once and cache them under a local
   `bench_assets/` dir (gitignored) — don't hit the network per trial.
2. Seed → `append_frame` → 8 no-op settle frames → 6 frames `mouse=[+0.2, 0]` →
   6 frames `mouse=[-0.2, 0]` → 4 no-op frames. Write video to mp4.
3. Eyeball the video: does the view pan right then return to the original heading?
   If yaw response is asymmetric or drifts, tune magnitude/steps or switch to the
   fallback trajectory (strafe right `button={68}` K/2 then left `button={65}` K/2).
4. **Checkpoint:** do not proceed to 2b until a returning trajectory is confirmed.
   Record the chosen trajectory parameters in the results doc.

Practical gotchas:
- `prep_inputs` mutates `CtrlInput` fields into tensors in-place
  (`world_engine.py:172-174`) — construct a **fresh CtrlInput per step**, never
  reuse an instance.
- First `gen_frame` call triggers torch.compile with max_autotune + cudagraphs:
  budget several minutes of warmup per process. Keep one process per sweep, reset
  the engine between trials with `engine.reset()` (do NOT re-instantiate).
- `gen_frame()` returns `(4, H, W, 3)` uint8 on-device; `.cpu().numpy()` to score.
- Seed `torch.manual_seed(seed)` at each trial start; the denoiser itself is a
  deterministic Euler solve — only the initial noise is stochastic.

### 2b. Three-arm protocol

Per (scene, seed, K) — K in latent frames:

| Arm | Trajectory | What it measures |
|---|---|---|
| **still** | settle → K no-op frames → score | autoregressive drift floor (no forgetting) |
| **revisit** | settle → pan-away K/2 → pan-back K/2 → score | drift + forgetting |
| **oracle** | settle → `get_state()` → run revisit arm → `load_state()` → gen 1 frame → score | sampling-noise floor (perfect memory) |

Common steps:
1. `torch.manual_seed(seed)`; `engine.reset()`; `engine.append_frame(seed_x4)`;
   8 no-op settle frames. Save the last RGB frame of the final 4-pack as
   **reference view A**.
2. Run the arm's trajectory; save the comparable returned frame as **view B**
   (for still/oracle, B is the frame generated at the matching timestep).
3. Score `(A, B)`: **PSNR** and **SSIM** (implement with numpy/cv2 — no new deps).
   Optional `--lpips` flag using the `lpips` package if installed; skip silently
   otherwise. Save an A|B side-by-side PNG per trial.

### 2c. Sweep

- K values: pick from Phase 1 horizons — e.g.
  `{local_window/2, local_window, 2×local_window, global_horizon/2,
  global_horizon, 2×global_horizon}` (round to even; cap total runtime).
- ≥3 scenes × ≥5 seeds per (arm, K). Persist every trial row to CSV incrementally
  (crash-safe): `scene, seed, arm, K, psnr, ssim, lpips, png_path`.
- CLI: `uv run --dev python examples/permanence_bench.py --model Overworld/Waypoint-1.5-1B
  --arms still,revisit,oracle --K ... --seeds 5 --out bench_out/`.

### 2d. Analysis + deliverable

- Plot mean±std PSNR/SSIM vs K per arm (matplotlib if available, else emit the
  CSV and a markdown table).
- **The headline result:** the gap between *revisit* and *still* as K crosses the
  local and global horizons = the measured forgetting curve.
- Write `docs/PERMANENCE_RESULTS.md`: horizons table, protocol description,
  curve, contact-sheet examples, caveats (camera-return approximation,
  single model, opaque metric choice).
- **Acceptance:** results doc exists with real numbers; the revisit-vs-still gap
  is quantified at each K; every claim traceable to a CSV row.

## Phase 3 — KV frame-pinning prototype (only after Phase 2 numbers exist)

**Design — dedicated pin slots** in `LayerKVCache` (`src/model/kv_cache.py`):

1. Add `n_pin_frames: int = 0` to `LayerKVCache.__init__` and thread it through
   `StaticKVCache` from a new config field (default 0 → current behavior, exact
   buffer shapes unchanged). Enable at runtime via
   `WorldEngine(model_config_overrides={"n_pin_frames": 4})` — add the field with
   a default to the config schema wherever `load_config` merges overrides.
2. Layout: capacity becomes `L + n_pin_frames·tpf + tpf` (ring | pins | tail).
   Ring upsert logic untouched; pin region only written by an explicit call;
   tail current-frame slice moves to the end. `written` mask covers pins
   (init False). Block alignment holds because slot granularity stays `tpf`
   (already 128-aligned or the current mask would fail its asserts).
3. New API:
   - `LayerKVCache.pin_current(slot)`: copy the tail-frame KV `[L_tail slice]`
     into pin slot, mark written.
   - `StaticKVCache.pin_current()` → all layers; **default: global layers only**
     (`--pin-all-layers` variant flag) — global layers are where long-horizon
     attention was trained.
   - `WorldEngine.pin_frame()`: call after the `gen_frame`/`append_frame` whose
     content should persist. Must run **outside** the compiled regions (it's a
     plain buffer mutation between steps; shapes are static so no recompile).
4. **Positions:** variant (a), the default — pinned keys keep their original
   post-RoPE phase (honest, zero extra change). Beyond the trained global horizon
   this is OOD — that is exactly what Phase 2's harness measures. Variant (b),
   re-stamped `t_pos` (requires caching pre-RoPE keys for pin slots and rotating
   at read time): **only implement if (a) measurably fails**, it's a much bigger
   change.
5. Evaluate: add `--pin-at-reference` to the Phase 2 harness (pin view A's frame,
   then run the revisit arm). Compare curves: baseline vs pinned vs oracle vs
   still. **Success criterion:** pinned revisit tracks the still-arm floor past
   the global horizon where baseline revisit has degraded; failure (quality
   collapse from OOD positions) is also a publishable result — report either way.
6. Sanity gates before the sweep: with `n_pin_frames=0` the state dict shapes and
   a fixed-seed 10-frame generation must be bit-identical to `main`; with pinning
   enabled and no `pin_frame()` calls, output must match baseline (empty pins are
   masked out by `written`).

## Phase 4 — Report + pitch (half day)

1. Finalize `docs/PERMANENCE_RESULTS.md` with Phase 3 curves.
2. One-page pitch section at the top: measured forgetting curve → inference-side
   pinning result → concrete training-side proposal (train with randomly-pinned
   distant keyframes / learned scene memory so the architecture supports
   permanence natively). The data from Phases 2–3 is the evidence; the training
   proposal is the "big win" ask.
3. PR to `main` with: benchmark script, pinning implementation (flag-gated,
   default-off), results doc. Keep the README fix commit separate.

---

## Environment / execution facts

- GPU: NVIDIA GB10 (unified memory) — bf16 1B model fits comfortably.
- Run everything as `uv run --dev python examples/...`; `HF_TOKEN` must be set.
- For fast iteration on harness *logic* only, `TORCHDYNAMO_DISABLE=1` skips
  compile warmup — never use it for reported numbers.
- `engine.reset()` between trials; one engine instance per process.
- Total estimated GPU time for Phase 2: sweep size ≈ 3 arms × 6 K × 3 scenes ×
  5 seeds ≈ 270 trials; trials are short (≤ 2×K+12 frames) — batch by scene to
  amortize seed-image loading.

## Risks (pinned, with handles)

- **Camera doesn't return** on mirrored yaw → handle: Phase 2a pilot gates
  everything; strafe fallback; report residual misalignment via the oracle arm.
- **Metrics too coarse** (PSNR punishes small view offsets) → handle: SSIM +
  optional LPIPS + side-by-side PNGs for qualitative judgment.
- **Pinning collapses quality** (OOD RoPE distances) → handle: expected possible
  outcome, measured not assumed; variant (b) as follow-up; negative result still
  supports the training-side pitch.
- **compile/cudagraph interference with pin writes** → handle: pin mutation in
  eager between steps; bit-identity sanity gates in Phase 3 step 6.
