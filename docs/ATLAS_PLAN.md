# Phase 6 — Pose-Indexed World Atlas (Implementation Plan)

**Goal:** turn Waypoint-1.5's fixed 128-frame KV ring into an **unbounded, revisitable
world memory** by accumulating a growing `(camera pose → KV keyframe)` store as the
player explores, and on every frame retrieving the *nearest stored pose* and restamping
only the **small residual** into the pin slots. This keeps the restamp primitive inside
the one regime where it is valid (small deltas), which is why it can beat the
single-reference restamp of Phase 5.

**The core insight (why this works when static pinning was a no-op):**
- Temporal repositioning of a revived key is **mathematically exact** (`ts_mult = 1`, so
  RoPE `t_pos` = frame index; proven bit-exact in
  `examples/test_kv_restamp.py::test_restamp_equals_reposition`).
- The **only lossy** part of any restamp is the **spatial (x,y) residual**, which aliases
  and is uncalibrated (`examples/analyze_pose_restamp.py`, three flaws).
- An atlas **never needs a large spatial delta**: retrieve a pose-matched keyframe and
  restamp the residual → the lossy operation is starved of the large deltas that break it,
  and the uniform-shift approximation (`restamp_pose_k`) becomes nearly correct because
  per-token displacement variance collapses as the delta → 0.

**Executor notes:** self-contained; no conversation context needed. This is the
*inference* library for the pretrained `Overworld/Waypoint-1.5-1B` checkpoint — the
attention layout is baked into the weights. The atlas is an **inference-side external
memory + retrieval** mechanism; it does not modify weights. Terminology: **object/scene
permanence**. Success or a measured ceiling are *both* publishable — the residual
atlas→oracle gap is the precise number the training-side pitch needs.

**Prereqs already built (reuse, do not reinvent):**
- Pin slots: `LayerKVCache` layout `[ring | pins | tail]`, `pin_current(frame, yaw)`
  (`src/model/kv_cache.py`), flag-gated by `n_pin_frames` (default 0 = bit-identical).
- Restamp: `restamp_temporal_k` / `restamp_pose_k` (`src/model/kv_cache.py`) and
  `WorldEngine.restamp_memory(offset, align_pose=)` (`src/world_engine.py`).
- Pose signal: `WorldEngine.camera_yaw` dead-reckoned from mouse-x
  (`src/world_engine.py`, accumulated in `prep_inputs`); `pin_frame()` already tags the
  pinned frame with `yaw=camera_yaw`.
- Benchmark: `examples/permanence_bench.py` (arms still/revisit/oracle/restamp,
  PSNR+SSIM, incremental CSV, `--self-test` CPU path); scoring +
  ≥15% verdict in `examples/restamp_gapclosure.py`.
- Live demo: `examples/play_server.py` (`gen_worker` owns the engine; `pacer_worker`
  paces frames).

---

## 6.0 — Design lock & scope (30 min, no GPU)

**Decisions to record at the top of `docs/ATLAS_RESULTS.md` before coding:**

1. **Pose dimensionality — MVP is yaw-only (1-DOF).** The existing dead-reckoning is
   yaw-only, and the Phase-2 benchmark trajectory is pure yaw, so a yaw-keyed atlas is
   directly measurable *today*. Translation (x,y) needs a WASD motion model and is
   **deferred to 6.6**. State this scope explicitly — do not silently ship a yaw atlas as
   if it solved translation permanence.
2. **Storage split — unbounded store + fixed working set.** The atlas store lives
   **outside** the compiled regions (CPU or a plain GPU tensor dict), size `O(scene
   coverage)`. The `n_pin_frames` pin slots are a **fixed-size working set** that a small
   set of retrieved keyframes is paged into each step. This keeps all compiled shapes
   static (no recompiles) while the store grows unbounded.
3. **What is captured per keyframe:** the **post-RoPE global-layer KV** of the target
   frame (identical to what `pin_current` copies), tagged `(yaw, frame_ts_at_capture)`.
   Post-RoPE matches the existing restamp primitive (it composes a *delta* rotation).
4. **Acceptance gate for proceeding to live integration (6.5):** the `atlas` arm's mean
   gap-closure (`restamp_gapclosure.py`) must **exceed the single-frame `restamp` arm's**
   at K ∈ {32, 64, 128}. If it ties/loses, stop at 6.4 and report the ceiling.

---

## 6.1 — `AtlasStore` data structure (CPU-only, fully testable without GPU)

New file: `examples/atlas.py` (examples/ is the sanctioned home for non-library tooling;
promote into `src/` only after 6.3 proves it).

1. `class AtlasStore`:
   - `insert(kv_per_layer, yaw, frame_ts)` with a **novelty gate**: skip if the minimum
     circular-yaw distance to any existing keyframe `< tau_insert` (SLAM keyframe-insertion
     rule). Returns whether it inserted.
   - `query(yaw, k=1)` → the `k` nearest keyframes by **circular** yaw distance
     (`atan2(sin Δ, cos Δ)`), nearest first.
   - **Bounding:** cap at `n_max` keyframes via a coverage policy — bin yaw into buckets
     and keep one-per-bucket (densest-bin eviction), OR least-recently-retrieved eviction.
     Log what was evicted (no silent truncation).
   - `kv_per_layer`: dict `{global_layer_idx: (K, V)}` each `[n_heads_kv, tpf, d_head]`
     (one frame's worth), on CPU by default; `.to(device)` on retrieval.
2. **CPU unit tests** (`examples/test_atlas.py`, pytest, no model load):
   - circular distance correctness across the ±π wrap;
   - novelty gate: N inserts within `tau_insert` of each other → 1 stored;
   - retrieval returns true nearest under wrap-around;
   - bound enforcement: `n_max+extra` inserts spread over yaw → exactly `n_max`, coverage
     preserved (no bucket fully dropped).

**Acceptance:** `uv run --dev pytest examples/test_atlas.py` green; zero GPU, zero model.

---

## 6.2 — Engine integration: capture & activate (needs GPU only for sanity gate)

Add to `WorldEngine` (`src/world_engine.py`), both eager, run **between** compiled steps
(static-shape buffer edits → no recompile), mirroring `pin_frame` / `restamp_memory`:

1. `atlas_capture()`:
   - Pull the just-cached frame's **global-layer** KV out of `kv_cache` (reuse the
     tail-slice read that `pin_current` copies from) and `insert()` it into an
     `AtlasStore` held on the engine, tagged `(self.camera_yaw, int(self.frame_ts)-1)`.
2. `atlas_activate(offset=8, k=1, align_pose=True)`:
   - `hits = store.query(self.camera_yaw, k)`;
   - copy each hit's KV into the pin slots (eager buffer write, like `pin_current`);
   - call the existing `restamp_memory(offset, align_pose=True)` so the loaded keys are
     repositioned to `frame_ts - offset` (in-distribution temporal offset) and rotated by
     the **residual** yaw `self.camera_yaw - hit.yaw` (small by construction).
   - If `k > n_pin_frames`, clamp and log.
3. Flag-gated: no-op unless `n_pin_frames > 0` and an atlas was attached
   (`engine.attach_atlas(AtlasStore(...))`). Zero new surface when unused.

**Sanity gates (full model, bf16, GB10):**
- *Determinism:* atlas detached, `n_pin_frames=0` → **bit-identical** to `main`
  (`max|Δlatent| = max|ΔRGB| = 0`).
- *Inertness:* atlas attached but **no `atlas_capture` calls** → identical to the
  `revisit` arm (empty store ⇒ `query` returns nothing ⇒ no activation).
- *Residual sanity:* after `atlas_activate`, log the residual yaw applied; with a
  keyframe captured at the current heading it must be ≈ 0.

---

## 6.3 — Decisive validation: the `atlas` arm (the whole point) — GPU

Extend `examples/permanence_bench.py` (add to `run_trial` / the arm dispatch, alongside
the existing `restamp` / `restamp_pose` arms):

1. New arm **`atlas`**, per (scene, seed, K):
   - settle → save reference view **A**;
   - **pan-away** K/2: `atlas_capture()` every `M` frames (default `M=4`), tagging each
     with its running `camera_yaw`;
   - **pan-back** K/2: before each `gen_frame`, `atlas_activate(offset, k, align_pose=True)`;
   - score the returned frame **B** vs A (PSNR + SSIM), same as every other arm.
2. Keep `revisit` (floor) and `oracle` (ceiling) as baselines; keep single-frame
   `restamp` for the head-to-head.
3. **Sweep** (start narrow, one scene/seed, then widen): `offset ∈ {4,8,16}`,
   `M ∈ {2,4,8}`, `k ∈ {1,2,4}`, `tau_insert ∈ {small, medium}`. Reuse
   `restamp_pose_probe.py` as the template (one all-layers engine, compile once, sweep in
   a loop, `engine.reset()` between trials).
4. Full sweep once tuned: **3 scenes × 5 seeds × 6 K** for arms
   `still,revisit,oracle,restamp,atlas`; every row to `bench_out/atlas/results.csv`.
5. Score with `restamp_gapclosure.py` (already emits per-K closure + the ≥15% verdict);
   extend it to also print the **`atlas` column and `atlas − restamp` delta**.

**Acceptance:** `atlas` closure **> `restamp` closure** at K ∈ {32,64,128}. A tie/loss is
still a result — it pins the inference ceiling. Report either way. Every claim traces to a
CSV row.

---

## 6.4 — Analysis, ablations & the ceiling number (half day, mostly CPU)

1. Extend `examples/combine_curves.py` to plot **revisit / restamp / atlas / oracle**
   PSNR & SSIM vs K (mean±std), asset under `docs/permanence_assets/`.
2. Report the headline: atlas closure per K, and the **residual atlas→oracle gap** — this
   is the precise "inference ceiling of a perfect pose store" the training pitch cites.
3. Ablations (each a row in the results doc, traced to CSV):
   - retrieval `k=1` vs nearest-few (does blending multiple poses help or smear?);
   - `tau_insert` / `M` sensitivity (coverage density vs cost);
   - `align_pose=True` vs `False` at the atlas's small deltas (isolate the spatial-restamp
     contribution now that the delta is small — this is the clean test the large-delta
     Phase-5 run could not do);
   - store bound `n_max` vs closure (how much memory actually buys permanence).
4. Write `docs/ATLAS_RESULTS.md` (house style of `PERMANENCE_RESULTS.md`): design
   decisions from 6.0, the curve, contact-sheet examples, ablations, caveats, and the
   one-paragraph handoff to the training team (the ceiling number).

---

## 6.5 — Live integration in `play_server.py` (only if 6.3 passes the gate)

1. `gen_worker`: attach an `AtlasStore`; each tick call `atlas_capture()` on a **novelty
   trigger** (pose moved past `tau_insert` since the last capture) and `atlas_activate()`
   before `gen_frame`. Bound the store for long sessions (6.1 policy).
2. Measure the cost: atlas ops are eager CPU↔GPU copies — confirm they do **not** stall
   the gen loop; report the `gen_ms` delta in `/stats`. If retrieval copy is hot, page
   only the `k` hits and keep the store pinned-memory.
3. Add `/atlas` fields to the `/stats` overlay: store size, retrieval hit yaw, residual
   yaw applied, captures this session.
4. **Qualitative acceptance:** pan away for **> 8.5 s** (past the 128-frame global
   horizon), return, and confirm the scene **persists** where baseline hallucinates —
   the vending-machine test from `PERMANENCE_RESULTS.md`, now felt live. Capture a
   before/after clip for the results doc.

---

## 6.6 — Stretch: translation (x,y) permanence (research-grade, after the yaw MVP ships)

1. Dead-reckon **position** from WASD button state (needs a movement/velocity model and a
   mouse/button→world calibration — the `ALFA` gap in `analyze_pose_restamp.py`) → 3-DOF
   pose key `(x, y, yaw)`; retrieval by **frustum overlap**, not scalar distance.
2. Implement the **per-token perspective-correct restamp** sketched in
   `analyze_pose_restamp.py::restamp_pose_per_token` (translation induces parallax the
   uniform shift cannot represent, even at small deltas).
3. Calibrate mouse/button units → latent-pixel shift empirically (optical flow on decoded
   frames under known control input) — closes FLAW 2 and unblocks accurate restamps.
4. Flag as research-grade; the **yaw MVP (6.1–6.5) is the shippable deliverable.**

---

## Risks (pinned, with handles)

- **Model still won't *use* the key even at an in-distribution position** (the Phase-3
  wall may persist regardless of restamp accuracy) → the atlas *measures* the ceiling; a
  low ceiling is the strongest evidence for the training-side ask. Not a failure.
- **Uniform-shift error at moderate residual** → tighten `tau_insert` / raise `M` so
  residuals stay small; only implement per-token restamp (6.6) if the ablation in 6.4
  shows spatial restamp is the bottleneck.
- **Recompiles from shape changes** → `n_pin_frames` stays static; the store lives outside
  compiled regions; all capture/activate edits are eager between steps.
- **VRAM growth on a long session** → store on CPU (pinned memory), page only `k` hits to
  GPU pin slots on demand; bound via 6.1's coverage policy.
- **GPU contention** (a llama-server holding the GB10 blocked the Phase-5 run) → 6.1 and
  the arm *logic* are developed and tested on the CPU `--self-test` path first; GPU is
  reserved only for the decisive 6.3 sweep and 6.2 sanity gates.

## Environment / execution facts

- Run everything as `uv run --dev python examples/...`; `HF_TOKEN` must be set.
- One `WorldEngine` per process; `engine.reset()` between trials (never re-instantiate).
- First `gen_frame` compiles (max_autotune + cudagraphs): budget several minutes warmup.
- `TORCHDYNAMO_DISABLE=1` for harness-logic iteration only — never for reported numbers.
- Fresh `CtrlInput` per step (`prep_inputs` mutates fields in place); pin/atlas mutations
  run eagerly between compiled steps.
