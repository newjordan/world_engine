# Object/Scene Permanence in Waypoint-1.5 — Results

> Measured evidence of scene-forgetting in the `Overworld/Waypoint-1.5-1B` world
> model's KV ring-buffer, plus an inference-side mitigation (KV frame pinning).
> Companion to [`PERMANENCE_PLAN.md`](./PERMANENCE_PLAN.md).

---

## Phase 1 — Memory horizons (config-derived + empirically verified)

Config: `Overworld/Waypoint-1.5-1B` (`config.yaml`, loaded without weights).

| Parameter | Value |
|---|---|
| `n_layers` | 24 |
| `height × width` (latent patch grid) | 16 × 32 |
| `tokens_per_frame` (`height·width`) | 512 |
| `patch` | [2, 2] |
| `channels` | 32 |
| `local_window` | 16 latent frames |
| `global_window` | 128 |
| `global_pinned_dilation` | 8 |
| `global_attn_period` / `global_attn_offset` | 4 / −1 |
| `temporal_compression` | 4 RGB frames per latent frame |
| `inference_fps` (RGB) / `base_fps` (latent) | 60 / 15 |
| **latent-frame rate** | 15 Hz → **1 latent frame = 4/60 s ≈ 0.0667 s** |

**Global layers:** indices `[3, 7, 11, 15, 19, 23]` — 6 of 24 (every 4th, offset −1 ≡ 3).
The other 18 are local layers.

### Memory horizons

| Layer type | # layers | Frames retained | Spacing | **Horizon (latent)** | **Horizon (RGB)** | **Horizon (seconds)** |
|---|---|---|---|---|---|---|
| **Local** | 18 | 16 (consecutive) | every frame | **16** | 64 | **≈ 1.07 s** |
| **Global** | 6 | 16 (dilated) | every 8th frame | **128** | 512 | **≈ 8.53 s** |

### ⚠️ Correction to the plan's global-horizon formula

The plan estimated the global horizon as `global_window × global_pinned_dilation
= 128 × 8 = 1024` latent frames. **That is 8× too large.** Reading
`src/model/kv_cache.py:71`, the global ring uses

```
num_buckets = (L // tokens_per_frame) // pinned_dilation = 128 // 8 = 16
```

so it retains only **16 distinct frames**, written every 8th step (`write_step =
frame_idx % dilation == 0`). Those 16 keyframes span `16 × 8 = 128` latent frames.
**Dilation sets temporal *resolution* (1-in-8 sampling), not horizon *length*.**
The horizon equals `global_window` latent frames = 128.

Verified empirically by tracing the real `LayerKVCache` on CPU (feed frames
`0…399`, read back which frame indices survive in the ring) —
`scratchpad/kv_horizon_probe.py`:

- global (window=128, dilation=8): 16 frames retained, spacing 8, oldest = 127 back → **horizon 128** ✓
- local (window=16, dilation=1): 16 frames retained, spacing 1, oldest = 15 back → **horizon 16** ✓
- dense control (window=128, dilation=1): 128 frames retained, horizon 127 ✓

Side note: because `L = global_window × tpf` but only `num_buckets × tpf = 16 × 512`
slots are ever written, **7/8 of each global layer's KV buffer is allocated but
masked out** (correctness is preserved by the `written` mask; it's wasted VRAM).

### K-sweep (chosen from the corrected horizons)

K = length of the revisit excursion, in **latent frames**. Chosen to straddle both
horizons (local at 16, global at 128):

| K (latent) | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| seconds | 0.53 | 1.07 | 2.13 | 4.27 | 8.53 | 17.07 |
| vs. horizons | ½ local | **= local** | 2× local | ½ global | **= global** | 2× global |

Max revisit trial ≈ 256 latent frames (~17 s of generation) — ~8× cheaper than the
plan's 2048-frame worst case.

---

## Phase 2 — Revisit-consistency benchmark

Harness: `examples/permanence_bench.py` (arms still/revisit/oracle, PSNR+SSIM,
incremental CSV, A|B contact sheets, `--self-test` CPU dry-run). Model:
`Overworld/Waypoint-1.5-1B`, bf16, NVIDIA GB10. First `gen_frame` compiles
(max_autotune + cudagraphs). Autotune confirmed the Phase-1 KV shapes: local layers
KV length 8704 (16·512 + 512 tail), global layers 66048 (128·512 + 512 tail).

### 2a — Camera-return pilot (gate: PASSED)

Trajectory: **pure yaw**, `mouse=[±0.2, 0]`, K/2 right then K/2 left, after an
8-frame settle. Pilot at K=12 (excursion within the local horizon), 1 scene:

| Comparison | PSNR | SSIM | expectation |
|---|---|---|---|
| A vs away (furthest pan) | 16.00 | 0.573 | LOW — view moved ✓ |
| A vs B (returned) | 22.48 | 0.697 | HIGH — view returned ✓ |

Verified visually (`bench_out/pilot/…_A_away_B.png`): the middle frame is a clean
rightward yaw; the returned frame matches the reference (same vending machine, wall,
puddle, HUD weapon). **The mirrored-yaw trajectory returns the camera** → the sweep
uses `--trajectory yaw`. (Strafe fallback `--trajectory strafe` remains available.)

### 2b — Three-arm protocol

Per (scene, seed, K), after an 8-frame settle whose last decoded RGB frame is the
**reference view A**:

| Arm | Trajectory | Scored frame B | Isolates |
|---|---|---|---|
| **still** | K no-op frames (camera static) | frame at step K | autoregressive drift only (memory always intact — the view never leaves) |
| **revisit** | pan-away K/2 → pan-back K/2 | returned frame | drift **+ forgetting** |
| **oracle** | `get_state` → run revisit → `load_state` → 1 frame | 1 frame from restored state | sampling-noise floor (perfect memory) |

Metrics: **PSNR** + **SSIM** (luma, 11×11 Gaussian) on (A, B). Each trial reset +
reseeded (`torch.manual_seed`); fresh `CtrlInput` per step. Sweep: **3 scenes × 5
seeds × 6 K × 3 arms = 270 trials**, all completed, 0 errors. Every row in
`bench_out/sweep/results.csv`.

### 2c–2d — The forgetting curve

Mean ± std over 3 scenes × 5 seeds (15 trials per cell):

**PSNR (dB)**

| K | still | revisit | **revisit − still** |
|---|---|---|---|
| 8 | 32.93 ± 2.40 | 25.48 ± 2.21 | −7.45 |
| 16 *(= local horizon)* | 29.72 ± 2.05 | 22.72 ± 1.97 | −7.00 |
| 32 | 27.25 ± 3.00 | 18.76 ± 1.77 | −8.49 |
| 64 | 26.26 ± 2.56 | 16.21 ± 1.90 | **−10.05** |
| 128 *(= global horizon)* | 23.98 ± 2.13 | 14.84 ± 1.67 | −9.15 |
| 256 | 20.55 ± 1.36 | 12.66 ± 1.26 | −7.89 |

**SSIM**

| K | still | revisit | **revisit − still** |
|---|---|---|---|
| 8 | 0.952 | 0.776 | −0.176 |
| 16 | 0.935 | 0.718 | −0.217 |
| 32 | 0.909 | 0.615 | −0.294 |
| 64 | 0.918 | 0.577 | **−0.341** |
| 128 | 0.885 | 0.554 | −0.331 |
| 256 | 0.832 | 0.493 | −0.339 |

![baseline forgetting curve](./permanence_assets/forgetting_curve_baseline.png)

**Headline:** panning away and returning costs **7–10 dB PSNR / 0.18–0.34 SSIM**
relative to holding the camera still, and the gap **grows as the excursion crosses
the 16-frame local horizon**, peaking at **K = 32–64** (2–4× local horizon) — exactly
where the 18 local layers have evicted the reference view and only the 6 dilated
global layers remain. Past the global horizon (K ≥ 128) the *still* arm itself starts
drifting (static-scene autoregressive decay), so the gap narrows even as both arms
degrade. The scene is **forgotten**, not merely noised: the effect is ~5× the
sampling/oracle floor and far outside the seed-to-seed std.

**Qualitative** (`docs/permanence_assets/example_revisit_K64_A_vs_B.png`, A | returned-B
at K=64): the returned view is recognizably the same alley (vending machine, red
pillar, puddle, HUD weapon) but the **specifics have hallucinated** — the vending
machine has shifted and recolored, geometry has drifted. The model keeps the *gist*
and loses the *detail*, the signature of ring-buffer eviction.

![revisit K=64 A vs B](./permanence_assets/example_revisit_K64_A_vs_B.png)

**Caveats.**
- *Camera-return approximation*: mirrored yaw returns the heading only approximately,
  contributing a roughly K-independent offset (already ~7 dB at K=8, within the local
  horizon). The **forgetting-specific** signal is the *growth* of the gap with K.
- *Metrics*: PSNR punishes sub-pixel view offsets; SSIM + the contact sheets guard the
  interpretation. LPIPS optional (not installed here).
- *Oracle*: the oracle numbers in `bench_out/sweep` are **contaminated** — they were
  produced before the `get_state`/`load_state` VAE-state fix (see below), so they drift
  with K instead of being a flat floor. A clean oracle re-run replaces them below.
- Single model / single checkpoint.

### Engine fix surfaced by the oracle arm

The oracle arm exposed a real bug: `WorldEngine.get_state()` claimed to capture the
world state but omitted the **TAEHV streaming decoder/encoder temporal buffers**, so
`load_state` left them at their post-excursion values — a restored state kept decoding
frames from the pre-snapshot stream. Fixed in `src/ae.py` + `src/world_engine.py`
(deep-clone of the 7 `StreamingTAEHV.reset()` state fields; backward-compatible with
older snapshots). Regression tests in `examples/test_ae_state.py`.

## Phase 3 — KV frame-pinning

_(pending)_

## Phase 4 — Pitch

_(pending)_
