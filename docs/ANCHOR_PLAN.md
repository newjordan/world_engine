# Phase 10 plan — Front-door re-anchoring with `append_frame`

**Status:** controlled sweep complete; see `docs/ANCHOR_RESULTS.md`.
**Branch:** `spin-persistence`
**Condition source:** this file.
**Historical comparator:** Phase 9 `docs/WARMSTART_RESULTS.md`, especially
`atlas_nowb +3.42 +/- 0.92 dB`, 6/6, and the failed KV write-back basin.

## Question

Phases 8 and 9 killed memory-bearing latent write-back into KV. They did not test
the model's intended conditioning entry: pixel frames passed through the VAE encoder
and cached by `append_frame`.

This phase asks whether a pose-indexed atlas reconstruction, fed through the
front door at loop closure, can repair the model's own continuation after the
intervention ends. That is the durability claim. A one-frame display correction is
not enough.

## Run label

`new_experiment`. This is a different representation channel from the dead KV
latent write-back basin:

- source memory is RGB atlas keyframes;
- reconstruction happens in pixel space;
- the intervention is encoded by the VAE and cached through the same path as a
  normal appended frame;
- no projected latent is written directly into KV.

## Resolved condition

- Model: `Overworld/Waypoint-1.5-1B`
- Device: one CUDA GPU
- Quantization: `None` unless explicitly passed with `--quant`
- Dataset/assets: `bench_assets` seed images loaded by `examples/permanence_bench.py`
- Tokenizer/vocab: not used by this image-conditioned benchmark
- Protocol: yaw-only return-to-start hysteresis
- `K=64`, `settle=8`, `yaw_mag=0.2`
- Scenes/seeds: 2 scenes x 3 seeds, seed base 1234 for the decisive run
- Hard regime: heading <= 3.2
- Tuned atlas/overlay canary: `M=1`, restamp offset 4, `atlas_k=1`
- Anchor reconstruction: repeat one registered RGB reconstruction 4x as the
  Waypoint-1.5 append-frame batch `[4,720,1280,3]`
- Post-intervention durability window: `post_frames=64` no-op generated latent frames
- Metric: PSNR/SSIM against the start view for post frames; paired hard-regime
  PSNR for the `atlas_nowb` canary

## Arms

| arm | mechanism | purpose |
|---|---|---|
| `revisit` | no memory aid | baseline post-return drift |
| `atlas_nowb` | Phase 9 tuned atlas + no-write-back overlay during pan-back only, then overlay off | incumbent canary and cosmetic-persistence collapse curve |
| `anchor` | generate normally through return, synthesize a registered RGB atlas reconstruction at start pose, append it, then run no-op post frames | true durability test |
| `anchor_blend` | same as `anchor`, but append a confidence-weighted blend of reconstruction and current frame | lower-risk front-door anchor |
| `anchor_k1` | anchor every return step plus at the end | cadence stress case |
| `anchor_k4` | anchor every 4th return step plus at the end | sparse cadence case |

Cadence arms insert explicit appended intervention frames after generated return
steps. These appended frames have zero mouse input, so they do not change the
dead-reckoned heading; they do advance model time. Their post curve starts after
the last appended intervention.

## Reconstruction rule

For an anchor event:

1. Query the nearest RGB atlas keyframe by dead-reckoned yaw.
2. Register the keyframe ROI against the current frame ROI with the same
   phase-correlation rule used by Phase 8/9.
3. Reject if response is below `resp_min` or if the shift exceeds
   `max_shift_frac * image_width`.
4. Shift the keyframe RGB by the measured dx, mask to the certified SLAM band,
   and paste only valid non-wrap columns into the current frame.
5. For `anchor_blend`, map response to alpha with a linear ramp from
   `resp_min` to `resp_full` and blend the reconstruction with the current frame.
6. Repeat the final RGB image 4x and call the engine's front-door append path.

The implementation calls `engine.append_frame(recon_x4, ctrl=noop)` for explicit
intervention frames. This intentionally advances one latent timestep; post rows are
scored only after the last intervention.

## CSV contract

`examples/anchor_probe.py` writes:

```text
scene,seed,arm,phase,idx,heading,psnr,ssim,projected,dx_px,resp,reject,anchor,alpha,anchor_count
```

- `phase=back`: return-path rows scored against the matched outbound heading.
- `phase=intervention`: appended anchor frame scored against the start view.
- `phase=post`: no-overlay/no-anchor/no-active-atlas continuation rows scored
  against the start view.
- `projected`: registration accepted for overlay/anchor synthesis.
- `anchor`: this row came from an appended front-door intervention.
- `anchor_count`: cumulative appended interventions in the trial.

## Gates

- **Kill before interpretation:** `atlas_nowb - revisit` in hard-regime pan-back
  rows must replicate within +/- 1 dB of `+3.4 dB`. If it does not, debug the
  harness before believing the anchor arms.
- **PASS:** any anchor arm beats `atlas_nowb` with overlay stopped by at least
  `+2 dB` at `post_idx=16`, wins 5/6+, and avoids the dx disease band
  (`mean accepted |dx|` in the approximate 7-20 px healthy range).
- **PARTIAL:** anchor improves the intervention/early post frames but decays
  before post_idx 16. Record the decay constant and carry the cadence result to
  the live demo path.
- **FAIL:** anchoring triggers large-dx/stall behavior or does not beat
  overlay-stopped persistence. This is evidence that frozen-model durability
  requires training, while the product path continues on compositing.

## Runbook

Pilot first:

```bash
uv run --dev python examples/anchor_probe.py \
  --arms revisit,atlas_nowb,anchor,anchor_blend \
  --n-scenes 1 --seeds 1 \
  --run-label scout \
  --csv bench_out/anchor/pilot.csv
```

Analyze the pilot:

```bash
uv run --dev python examples/analyze_anchor.py bench_out/anchor/pilot.csv \
  --out bench_out/anchor/pilot_report
```

Decisive sweep:

```bash
uv run --dev python examples/anchor_probe.py \
  --arms revisit,atlas_nowb,anchor,anchor_blend,anchor_k1,anchor_k4 \
  --M 1 --restamp-offset 4 \
  --n-scenes 2 --seeds 3 \
  --run-label new_experiment \
  --csv bench_out/anchor/anchor.csv
```

Save stdout beside the CSV with `tee` for any decisive run. The runner prints the
command, source hashes, model, scheduler grid, arms, seeds, and metric at startup.

## CPU tests

Run:

```bash
uv run --dev pytest examples/test_anchor.py examples/test_warmstart.py -q
```

Before a full sweep, also keep the inherited gates green:

```bash
uv run --dev pytest examples/test_rigid.py examples/test_graybox.py -q
```

## Result language

Do not call a pilot a Phase 10 result. Do not compare a pilot to the n=6 Phase 9
table as if it were decisive. Do not claim true durability unless the post rows
after the last intervention were generated with no overlay and no further anchors.
