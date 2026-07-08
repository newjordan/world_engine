# Phase 10b plan - Four-frame front-door anchoring

**Status:** planned / runnable, no result yet.
**Branch:** `spin-persistence`
**Condition source:** this file.
**Historical comparator:** Phase 10 `docs/ANCHOR_RESULTS.md`.
**Run label:** `new_experiment`.

## Question

Phase 10 tested the legal front door but fed it a temporally weak batch: one
registered RGB reconstruction repeated four times as `[4,720,1280,3]`. Waypoint-1.5
conditions on four-frame video chunks, so the repeat may be out-of-distribution
even when each image is individually plausible.

This phase asks whether a temporally structured front-door batch can create true
durability: after the last anchor intervention, the model's own no-op rollout
should stay closer to the start view than the stopped-overlay comparator.

## Boundary Change

This is a representation change from Phase 10, not a dose sweep:

- Phase 10 anchor: register one atlas RGB frame to the current frame, then repeat
  the reconstruction 4x.
- Phase 10b anchor4: estimate four micro-yaws for the current return chunk,
  retrieve the nearest atlas micro-frame for each target micro-yaw, register each
  micro-frame against the matching current micro-frame, then append the resulting
  four-frame reconstruction batch.

The intervention still enters only through `engine.append_frame`; there is no
direct KV or latent write-back.

## Resolved Condition

- Model: `Overworld/Waypoint-1.5-1B`
- Device: one CUDA GPU
- Quantization: `None` unless explicitly passed with `--quant`
- Dataset/assets: `bench_assets` seed images loaded by
  `examples/permanence_bench.py`
- Tokenizer/vocab: not used by this image-conditioned benchmark
- Protocol: yaw-only return-to-start hysteresis
- `K=64`, `settle=8`, `yaw_mag=0.2`
- Scenes/seeds: 2 scenes x 3 seeds, seed base 1234 for the decisive run
- Hard regime: heading <= 3.2
- Tuned canary: `atlas_nowb`, `M=1`, restamp offset 4, `atlas_k=1`
- Post-intervention window: `post_frames=64` no-op generated latent frames
- Metric: paired PSNR at post frame 16 against the start view, plus the
  hard-regime `atlas_nowb - revisit` canary
- Source runner: `examples/anchor4_probe.py`, backed by
  `examples/anchor_probe.py` and `examples/anchor.py`

## Arms

| arm | mechanism | purpose |
|---|---|---|
| `revisit` | no memory aid | baseline post-return drift |
| `atlas_nowb` | tuned Phase 9/10 atlas overlay during pan-back only, then overlay off | incumbent canary |
| `anchor4_linear` | one final four-micro-pose RGB anchor batch at loop closure | primary temporal front-door test |
| `anchor4_blend` | same as `anchor4_linear`, but confidence-blends each accepted micro-frame with the current micro-frame | lower-risk temporal anchor |
| `anchor4_k4` | four-micro-pose anchor every 4th return step plus final loop-closure anchor | sparse cadence product/science bridge |

Do not rerun `anchor`, `anchor_blend`, `anchor_k1`, or `anchor_k4` unchanged and
call it new evidence. Those were Phase 10 and failed.

## Reconstruction Rule

For an anchor4 event:

1. Estimate the current generated chunk's four micro-yaws by linearly spreading
   the realized mouse yaw delta across the four decoded frames.
2. For each micro-frame, query the RGB atlas by the target micro-yaw.
3. Select the nearest stored micro-frame inside that atlas keyframe's four-frame
   chunk.
4. Register that stored micro-frame ROI against the corresponding current
   micro-frame ROI using the Phase 8/9 phase-correlation rule.
5. Reject the micro-frame if response is below `resp_min` or if horizontal shift
   exceeds `max_shift_frac * image_width`.
6. Paste only the certified SLAM band and non-wrap columns into the current
   micro-frame. For `anchor4_blend`, map response to alpha with the existing
   confidence ramp.
7. Append the reconstructed `[4,H,W,3]` batch with a no-op control.

If no micro-frame is accepted, do not append an intervention. If only some
micro-frames are accepted, append a mixed batch where rejected micro-frames remain
the current generated frames; the CSV row is marked partial.

## CSV Contract

The runner keeps the Phase 10 CSV schema:

```text
scene,seed,arm,phase,idx,heading,psnr,ssim,projected,dx_px,resp,reject,anchor,alpha,anchor_count
```

For anchor4 intervention rows, `dx_px`, `resp`, and `alpha` are accepted
micro-frame means. Diagnostics count individual micro-frame registrations, not
just anchor events.

## Gates

- **Kill before interpretation:** `atlas_nowb - revisit` in hard-regime pan-back
  rows must replicate within +/- 1 dB of the Phase 10 canary (`+3.47 dB`). If it
  does not, debug the harness before believing anchor4.
- **PASS:** any anchor4 arm beats `atlas_nowb` with overlay stopped by at least
  `+2 dB` at `post_idx=16`, wins 5/6+, and avoids the dx disease band.
- **PARTIAL:** anchor4 improves over raw revisit or over Phase 10 anchors but
  decays before post frame 16. Record the decay curve and carry the best cadence
  into the product compositor.
- **FAIL:** anchor4 does not beat overlay-stopped persistence or reproduces the
  large-dx/stall fingerprint. That strengthens the boundary: frozen-model true
  durability likely needs training, while the product path remains compositor-led.

## Runbook

Pilot:

```bash
uv run --dev python examples/anchor4_probe.py \
  --arms revisit,atlas_nowb,anchor4_linear,anchor4_blend \
  --n-scenes 1 --seeds 1 \
  --run-label scout \
  --csv bench_out/anchor4/pilot.csv
```

Analyze with the existing analyzer:

```bash
uv run --dev python examples/analyze_anchor.py bench_out/anchor4/pilot.csv \
  --out bench_out/anchor4/pilot_report
```

Decisive sweep:

```bash
uv run --dev python examples/anchor4_probe.py \
  --arms revisit,atlas_nowb,anchor4_linear,anchor4_blend,anchor4_k4 \
  --M 1 --restamp-offset 4 \
  --n-scenes 2 --seeds 3 \
  --run-label new_experiment \
  --csv bench_out/anchor4/anchor4.csv
```

Save stdout beside the CSV with `tee`. The runner must print the command,
condition source, source hashes, resolved condition, arms, seed grid, scheduler
grid, and metric at startup.

## CPU Tests

Before any GPU run:

```bash
uv run --dev pytest examples/test_anchor.py examples/test_warmstart.py -q
```

Before a decisive sweep:

```bash
uv run --dev pytest examples/test_rigid.py examples/test_graybox.py -q
```

## Result Language

Do not call the plan or CPU tests a result. Do not compare the pilot as if it were
the n=6 aggregate. A valid Phase 10b result requires the exact condition above,
recorded stdout/log, raw CSV, and a frames-first review packet before claims.
