# Phase 10e plan - Admission-gated full-temporal anchoring

**Status:** planned / runnable, no result yet.
**Branch:** `spin-persistence`
**Condition source:** this file.
**Historical comparator:** `docs/ANCHOR4_FULL_RESULTS.md`.
**Run label:** `new_experiment`.

## Question

Phase 10d found the first controlled yaw-only frontier pass: hard full-frame
four-frame anchoring beat the stopped-overlay comparator by more than 2 dB at
post frame 16. The remaining problem is admission: the wrong-pose control still
accepted in half the trials before losing on average.

This phase keeps the useful representation fixed and changes only the admission
policy before the reconstructed temporal chunk is appended.

## Boundary Change

This is an algorithm/admission change, not a new memory channel:

- source memory remains RGB full-frame temporal micro-poses;
- reconstruction still enters only through `engine.append_frame`;
- no direct KV editing;
- no generated or edited frames are used as evidence.

## Resolved Condition

- Model: `Overworld/Waypoint-1.5-1B`
- Device: one CUDA GPU
- Quantization: `None` unless explicitly passed with `--quant`
- Dataset/assets: `bench_assets`
- Protocol: yaw-only return-to-start hysteresis
- `K=64`, `settle=8`, `yaw_mag=0.2`
- Long sweep: 4 scenes x 4 seeds, seed base 1234
- Tuned canary: `atlas_nowb`, `M=1`, restamp offset 4, `atlas_k=1`
- Metric: paired post16 PSNR against start, plus hard-regime
  `atlas_nowb - revisit` canary

## Arms

| arm | mechanism | purpose |
|---|---|---|
| `revisit` | no memory aid | baseline |
| `atlas_nowb` | tuned overlay during return, stopped for post window | canary |
| `anchor_full` | one full-frame reconstruction repeated 4x | Phase 10c comparator |
| `anchor4_full` | full-frame four-frame temporal reconstruction | Phase 10d comparator |
| `anchor4_full_dx64` | `anchor4_full`, reject if mean accepted dx > 64 px | tight dx admission |
| `anchor4_full_dx96` | `anchor4_full`, reject if mean accepted dx > 96 px | moderate dx admission |
| `anchor4_full_resp25` | `anchor4_full`, reject if mean response < 0.25 | response admission |
| `anchor4_full_dx96_resp25` | both dx96 and resp25 gates | combined admission |
| `anchor4_full_far` | farthest-pose full temporal chunk | wrong-pose control |
| `anchor4_full_dx64_far` | farthest-pose chunk with dx64 admission | gated wrong-pose control |
| `anchor4_full_dx96_far` | farthest-pose chunk with dx96 admission | gated wrong-pose control |
| `anchor4_full_resp25_far` | farthest-pose chunk with resp25 admission | gated wrong-pose control |
| `anchor4_full_dx96_resp25_far` | farthest-pose chunk with combined admission | gated wrong-pose control |

All admission-gated arms fail closed if any of the four micro-frames rejects.

## Gates

- Canary must remain within +/- 1 dB of the Phase 10d `+3.39 dB` hard-regime
  result.
- A useful admission improvement must keep `anchor4_full`-level post16 gain while
  lowering wrong-pose acceptance or losing fewer paired comparisons than the
  ungated baseline.
- A promoted arm should beat `atlas_nowb` at post16 by at least `+2 dB`, win at
  least 75% of paired trials, and keep the far control below canary.

## Runbook

Long sweep:

```bash
uv run --dev python examples/anchor_probe.py \
  --condition-source docs/ANCHOR_ADMISSION_PLAN.md \
  --arms revisit,atlas_nowb,anchor_full,anchor4_full,anchor4_full_dx64,anchor4_full_dx96,anchor4_full_resp25,anchor4_full_dx96_resp25,anchor4_full_far,anchor4_full_dx64_far,anchor4_full_dx96_far,anchor4_full_resp25_far,anchor4_full_dx96_resp25_far \
  --M 1 --restamp-offset 4 \
  --n-scenes 4 --seeds 4 \
  --run-label new_experiment \
  --csv bench_out/anchor_admission/anchor_admission.csv
```

Full 180-degree pan MP4 review:

```bash
uv run --dev python examples/anchor_pan_video.py \
  --condition-source docs/ANCHOR_ADMISSION_PLAN.md \
  --arms revisit,atlas_nowb,anchor4_full,anchor4_full_dx96 \
  --target-deg 180 --scene 0 --seed 1234 \
  --out bench_out/anchor_admission/pan180_case0
```

## Result Language

Do not call one video a benchmark. The MP4s are controlled review artifacts for
specific cases. The sweep CSV and analyzer report carry the aggregate result.
