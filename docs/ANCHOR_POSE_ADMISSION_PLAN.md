# Phase 10f plan - Pose-distance admission for full-temporal anchoring

**Status:** planned / runnable, no result yet.
**Branch:** `spin-persistence`
**Condition source:** this file.
**Historical comparators:** `docs/ANCHOR4_FULL_RESULTS.md` and
`docs/ANCHOR_ADMISSION_PLAN.md`.
**Run label:** `new_experiment`.

## Question

Phase 10e tested dx and response admission. It found useful behavior, but also
two failure modes:

- dx gates reject large-shift wrong-pose controls, but miss low-dx pose aliases;
- response gates reject low-response wrong-pose controls, but also reject valid
  low-response anchors.

This phase uses the pose metadata already present in the temporal anchor path:
each projected micro-frame records `target_yaw` and `kf_frame_yaw`. The admission
rule checks their distance directly.

## Boundary Change

This is an admission-only change:

- source memory remains RGB full-frame temporal micro-poses;
- reconstruction still enters only through `engine.append_frame`;
- no direct KV editing;
- partial temporal chunks are allowed if all projected micro-frames are
  pose-consistent.

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
| `anchor4_full_pose02` | allow projected micro-frames only if max pose error <= 0.2 yaw units | strict pose admission |
| `anchor4_full_pose04` | allow projected micro-frames only if max pose error <= 0.4 yaw units | moderate pose admission |
| `anchor4_full_far` | farthest-pose full temporal chunk | wrong-pose control |
| `anchor4_full_pose02_far` | farthest-pose chunk with pose02 admission | gated wrong-pose control |
| `anchor4_full_pose04_far` | farthest-pose chunk with pose04 admission | gated wrong-pose control |

## Gates

- Canary must remain within +/- 1 dB of the Phase 10d `+3.39 dB` hard-regime
  result.
- A useful pose admission arm should preserve the ungated `anchor4_full` post16
  gain and reject far-pose controls more reliably than dx-only admission.
- A promoted arm should beat `atlas_nowb` at post16 by at least `+2 dB`, win at
  least 75% of paired trials, and keep the gated far controls below canary.

## Runbook

Long sweep:

```bash
uv run --dev python examples/anchor_probe.py \
  --condition-source docs/ANCHOR_POSE_ADMISSION_PLAN.md \
  --arms revisit,atlas_nowb,anchor_full,anchor4_full,anchor4_full_pose02,anchor4_full_pose04,anchor4_full_far,anchor4_full_pose02_far,anchor4_full_pose04_far \
  --M 1 --restamp-offset 4 \
  --n-scenes 4 --seeds 4 \
  --run-label new_experiment \
  --csv bench_out/anchor_pose_admission/anchor_pose_admission.csv
```

Full 180-degree pan MP4 review after the sweep:

```bash
uv run --dev python examples/anchor_pan_video.py \
  --condition-source docs/ANCHOR_POSE_ADMISSION_PLAN.md \
  --arms revisit,atlas_nowb,anchor4_full,anchor4_full_pose04 \
  --target-deg 180 --scene 0 --seed 1235 \
  --out bench_out/anchor_pose_admission/pan180_case0
```

## Result Language

Do not call one MP4 a benchmark. Videos are controlled review artifacts for
specific cases; the sweep CSV/report carries the aggregate claim.
