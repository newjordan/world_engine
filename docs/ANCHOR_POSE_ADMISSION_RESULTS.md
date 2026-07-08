# Phase 10f results - Pose-distance admission for full-temporal anchoring

**Status:** complete.
**Branch:** `spin-persistence`.
**Condition source:** `docs/ANCHOR_POSE_ADMISSION_PLAN.md`.
**Run id:** `20260708T143846-0500`.

## Artifacts

- CSV: `bench_out/anchor_pose_admission/anchor_pose_admission_20260708T143846-0500.csv`
- Log: `bench_out/anchor_pose_admission/anchor_pose_admission_20260708T143846-0500.log`
- Report: `bench_out/anchor_pose_admission/anchor_pose_admission_20260708T143846-0500_report/index.html`
- Summary: `bench_out/anchor_pose_admission/anchor_pose_admission_20260708T143846-0500_report/summary.json`
- MP4 review packet: `bench_out/anchor_pose_admission/pan180_scene0_seed1235_20260708T155108-0500/index.html`

Source hashes recorded by the runner:

- `anchor_probe=f7a55aa94cce51d7`
- `anchor=de3aa308191920e2`
- `plan=c65c52a44b6213ff`

## Condition

This was a 4 scene x 4 seed sweep with `Overworld/Waypoint-1.5-1B` on CUDA,
no quantization, `K=64`, `settle=8`, `yaw_mag=0.2`, `M=1`, restamp offset 4,
and post16 PSNR against the start frame. The canary remained valid:
`atlas_nowb - revisit = +2.94 +/- 0.99 dB`, `16/16`.

## Aggregate Result

| arm | post16 delta vs `atlas_nowb` | wins | accept |
|---|---:|---:|---:|
| `anchor4_full` | `+4.35 +/- 2.92 dB` | `15/16` | `94%` |
| `anchor4_full_pose02` | `+4.35 +/- 2.92 dB` | `15/16` | `94%` |
| `anchor4_full_pose04` | `+4.35 +/- 2.92 dB` | `15/16` | `94%` |
| `anchor4_full_far` | `-1.30 +/- 1.34 dB` | `2/16` | `50%` |
| `anchor4_full_pose02_far` | `-1.12 +/- 0.51 dB` | `0/16` | `0%` |
| `anchor4_full_pose04_far` | `-1.12 +/- 0.51 dB` | `0/16` | `0%` |

Pose02 and pose04 preserved the full useful `anchor4_full` gain in every paired
aggregate comparison while rejecting all far-pose control anchors in this sweep.
The promoted algorithmic step is pose-distance admission over projected
micro-frames, with partial chunks allowed when every projected micro-frame is
pose-consistent.

## Video Review

The full-pan review packet records real generated frames as MP4:
settle -> measured 180 degree pan-away -> symmetric return -> anchor attempt ->
post window. The comparison video is 1920x1080 at 30 fps and the per-arm videos
are 960x540 at 30 fps.

Artifacts:

- `compare_2x2.mp4`
- `revisit.mp4`
- `atlas_nowb.mp4`
- `anchor4_full.mp4`
- `anchor4_full_pose02.mp4`
- `manifest.json`

This video packet is a controlled visual review artifact, not the aggregate
benchmark. In the scene0/seed1235 180-degree visual stress case, the final anchor
attempt rejected at loop closure (`reject=empty`, low response), so the MP4 shows
the current long-pan boundary directly. The aggregate claim remains the Phase 10f
yaw return-to-start sweep above.
