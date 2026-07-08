# Phase 10c plan - Full-frame and confidence-map front-door anchoring

**Status:** planned / runnable, no result yet.
**Branch:** `spin-persistence`
**Condition source:** this file.
**Historical comparators:** Phase 10 `docs/ANCHOR_RESULTS.md` and Phase 10b
`docs/ANCHOR4_PLAN.md` scout artifacts.
**Run label:** `new_experiment`.

## Question

Phase 10 and the Phase 10b scout both inserted only a band-limited memory
reconstruction into the front door. That may be too weak a representation for the
VAE/context path to move the model's state. This phase tests a stronger
image-channel representation: full-frame registered memory, optionally softened by
a confidence map.

The claim is still true durability: after the final front-door intervention, the
post window must run with no overlay, no active atlas pins, and no further anchors.

## Boundary Change

This is a representation change:

- Phase 10: certified SLAM-band paste repeated 4x.
- Phase 10b: certified SLAM-band paste as a four-frame micro-pose chunk.
- Phase 10c: full-frame registered memory, or a soft full-frame confidence map,
  appended through `engine.append_frame`.

There is still no direct KV or latent write-back.

## Resolved Condition

- Model: `Overworld/Waypoint-1.5-1B`
- Device: one CUDA GPU
- Quantization: `None` unless explicitly passed with `--quant`
- Dataset/assets: `bench_assets` seed images loaded by
  `examples/permanence_bench.py`
- Tokenizer/vocab: not used by this image-conditioned benchmark
- Protocol: yaw-only return-to-start hysteresis
- `K=64`, `settle=8`, `yaw_mag=0.2`
- Scenes/seeds: 2 scenes x 3 seeds, seed base 1234 for decisive runs
- Hard regime: heading <= 3.2
- Tuned canary: `atlas_nowb`, `M=1`, restamp offset 4, `atlas_k=1`
- Post-intervention window: `post_frames=64` no-op generated latent frames
- Metric: paired PSNR at post frame 16 against the start view, plus the
  hard-regime `atlas_nowb - revisit` canary
- Source runner: `examples/anchor_probe.py`

## Arms

| arm | mechanism | purpose |
|---|---|---|
| `revisit` | no memory aid | baseline post-return drift |
| `atlas_nowb` | tuned atlas overlay during pan-back only, then overlay off | incumbent canary |
| `anchor_full` | registered memory pasted over all non-wrap columns, repeated 4x | strongest pixel-memory front-door test |
| `anchor_full_blend` | same as `anchor_full`, confidence-blended with the current frame | softer full-frame test |
| `anchor_confmap` | registered memory weighted by a smooth confidence map: full in the certified band, softer outside it | middle ground between band and full-frame |
| `anchor_full_far` | same as `anchor_full`, but retrieves the farthest pose keyframe | wrong-pose control for texture injection |

Do not call the wrong-pose control a persistence result. It is a falsification
control to check whether full-frame appends are merely injecting plausible texture.

## Reconstruction Rule

For a full-frame anchor event:

1. Query the RGB atlas by current yaw, unless the arm is `anchor_full_far`.
2. Register the retrieved keyframe ROI against the current frame ROI using the
   inherited phase-correlation rule.
3. Reject if response is below `resp_min` or if horizontal shift exceeds
   `max_shift_frac * image_width`.
4. Shift the full retrieved RGB frame by measured `dx`.
5. Apply one of three masks:
   - `full`: all non-wrap columns, across the full image;
   - `full_blend`: full mask, then scalar confidence blend with current RGB;
   - `confmap`: full mask multiplied by a smooth confidence map that is strongest
     in the certified band and tapers outside it.
6. Repeat the reconstructed frame 4x and append it with a no-op control.

## CSV Contract

The runner keeps the Phase 10 CSV schema:

```text
scene,seed,arm,phase,idx,heading,psnr,ssim,projected,dx_px,resp,reject,anchor,alpha,anchor_count
```

## Gates

- **Kill before interpretation:** `atlas_nowb - revisit` in hard-regime pan-back
  rows must replicate within +/- 1 dB of the Phase 10 canary (`+3.47 dB`).
- **PASS:** any valid nearest-pose full/confmap arm beats `atlas_nowb` with
  overlay stopped by at least `+2 dB` at `post_idx=16`, wins 5/6+, and does not
  reproduce the large-dx stall fingerprint.
- **CONTROL CHECK:** `anchor_full_far` should not track or beat nearest-pose
  `anchor_full`. If it does, the representation may be cosmetic texture injection.
- **PARTIAL:** nearest-pose full/confmap arms improve over raw revisit but decay
  before post frame 16. Carry useful cadence or compositor behavior forward.

## Runbook

Scout:

```bash
uv run --dev python examples/anchor_probe.py \
  --condition-source docs/ANCHOR_FULL_PLAN.md \
  --arms revisit,atlas_nowb,anchor_full,anchor_full_blend,anchor_confmap,anchor_full_far \
  --n-scenes 1 --seeds 1 \
  --run-label scout \
  --csv bench_out/anchor_full/scout.csv
```

Decisive sweep only if the scout has a healthy canary and at least one
nearest-pose full/confmap arm clears raw revisit without the wrong-pose control
matching it:

```bash
uv run --dev python examples/anchor_probe.py \
  --condition-source docs/ANCHOR_FULL_PLAN.md \
  --arms revisit,atlas_nowb,anchor_full,anchor_full_blend,anchor_confmap,anchor_full_far \
  --M 1 --restamp-offset 4 \
  --n-scenes 2 --seeds 3 \
  --run-label new_experiment \
  --csv bench_out/anchor_full/anchor_full.csv
```

## Result Language

Do not call a scout decisive. Do not claim true durability from an intervention
frame. Only post-window rows after the last intervention can support a durability
claim.
