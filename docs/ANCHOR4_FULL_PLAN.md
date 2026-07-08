# Phase 10d plan - Full-frame four-frame front-door anchoring

**Status:** controlled n=6 sweep complete; see `docs/ANCHOR4_FULL_RESULTS.md`.
**Branch:** `spin-persistence`
**Condition source:** this file.
**Historical comparators:** `docs/ANCHOR4_PLAN.md` and
`docs/ANCHOR_FULL_RESULTS.md`.
**Run label:** `new_experiment`.

## Question

Phase 10b changed the temporal representation but kept the weak certified-band
paste. Phase 10c changed the spatial representation to full-frame memory and
produced the strongest signal so far. This phase combines those two changes:
full-frame registered memory as a four-frame temporal micro-pose chunk.

The claim remains post-intervention durability: after the final anchor, the model
continues with no overlay, no atlas pins, and no further anchors.

## Boundary Change

This is a representation change:

- `anchor4_*` tested temporal chunks with band masks.
- `anchor_full` tested full-frame memory repeated 4x.
- `anchor4_full` tests full-frame memory as four registered micro-poses.

The intervention still enters only through `engine.append_frame`.

## Resolved Condition

- Model: `Overworld/Waypoint-1.5-1B`
- Device: one CUDA GPU
- Quantization: `None` unless explicitly passed with `--quant`
- Dataset/assets: `bench_assets`
- Protocol: yaw-only return-to-start hysteresis
- `K=64`, `settle=8`, `yaw_mag=0.2`
- Scout: 1 scene x 1 seed, seed base 1234
- Decisive run, if scout earns it: 2 scenes x 3 seeds, seed base 1234
- Tuned canary: `atlas_nowb`, `M=1`, restamp offset 4, `atlas_k=1`
- Metric: paired post16 PSNR against start, plus hard-regime
  `atlas_nowb - revisit` canary

## Arms

| arm | mechanism | purpose |
|---|---|---|
| `revisit` | no memory aid | baseline |
| `atlas_nowb` | tuned overlay during return, stopped for post window | canary |
| `anchor_full` | Phase 10c best arm: one full-frame reconstruction repeated 4x | known positive comparator |
| `anchor4_full` | four full-frame registered micro-poses | primary test |
| `anchor4_full_blend` | four full-frame micro-poses confidence-blended with current frames | softer test |
| `anchor4_full_far` | four full-frame micro-poses using farthest keyframes | wrong-pose control |

## Gates

- Canary must remain within +/- 1 dB of `+3.45 dB`.
- Scout earns an n=6 sweep only if `anchor4_full` beats `anchor_full` or beats
  `atlas_nowb` while `anchor4_full_far` stays below nearest-pose arms.
- A decisive pass would be `+2 dB` over stopped-overlay `atlas_nowb` at post16,
  5/6 wins, with wrong-pose control below nearest-pose full-temporal anchoring.

## Runbook

Scout:

```bash
uv run --dev python examples/anchor_probe.py \
  --condition-source docs/ANCHOR4_FULL_PLAN.md \
  --arms revisit,atlas_nowb,anchor_full,anchor4_full,anchor4_full_blend,anchor4_full_far \
  --n-scenes 1 --seeds 1 \
  --run-label scout \
  --csv bench_out/anchor4_full/scout.csv
```

Decisive sweep only if the scout earns it:

```bash
uv run --dev python examples/anchor_probe.py \
  --condition-source docs/ANCHOR4_FULL_PLAN.md \
  --arms revisit,atlas_nowb,anchor_full,anchor4_full,anchor4_full_blend,anchor4_full_far \
  --M 1 --restamp-offset 4 \
  --n-scenes 2 --seeds 3 \
  --run-label new_experiment \
  --csv bench_out/anchor4_full/anchor4_full.csv
```

## Result Language

Do not call a scout decisive. Do not claim full durability unless post-window rows
after the last intervention meet the stated gate.
