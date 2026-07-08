# Phase 10e results - Admission-gated full-temporal anchoring

**Status:** complete.
**Branch:** `spin-persistence`.
**Condition source:** `docs/ANCHOR_ADMISSION_PLAN.md`.
**Run id:** `20260708T130320-0500`.

## Artifacts

- CSV: `bench_out/anchor_admission/anchor_admission_20260708T130320-0500.csv`
- Log: `bench_out/anchor_admission/anchor_admission_20260708T130320-0500.log`
- Report: `bench_out/anchor_admission/anchor_admission_20260708T130320-0500_report/index.html`
- Summary: `bench_out/anchor_admission/anchor_admission_20260708T130320-0500_report/summary.json`

Source hashes recorded by the runner:

- `anchor_probe=f077354a7c3de70d`
- `anchor=de3aa308191920e2`
- `plan=fa1ac0f58909c944`

## Condition

This was a 4 scene x 4 seed sweep with `Overworld/Waypoint-1.5-1B` on CUDA,
no quantization, `K=64`, `settle=8`, `yaw_mag=0.2`, `M=1`, restamp offset 4,
and post16 PSNR against the start frame. The canary remained valid:
`atlas_nowb - revisit = +2.98 +/- 1.00 dB`, `16/16`.

## Result

| arm | post16 delta vs `atlas_nowb` | wins | accept |
|---|---:|---:|---:|
| `anchor4_full` | `+4.32 +/- 3.12 dB` | `14/16` | `94%` |
| `anchor4_full_dx64` | `+3.37 +/- 3.52 dB` | `12/16` | `75%` |
| `anchor4_full_dx96` | `+3.37 +/- 3.52 dB` | `12/16` | `75%` |
| `anchor4_full_resp25` | `+0.86 +/- 2.73 dB` | `6/16` | `38%` |
| `anchor4_full_dx96_resp25` | `+0.86 +/- 2.73 dB` | `6/16` | `38%` |
| `anchor4_full_far` | `-1.24 +/- 1.54 dB` | `2/16` | `62%` |
| `anchor4_full_dx64_far` | `-1.69 +/- 1.21 dB` | `0/16` | `19%` |
| `anchor4_full_dx96_far` | `-1.69 +/- 1.21 dB` | `0/16` | `19%` |
| `anchor4_full_resp25_far` | `-1.11 +/- 0.52 dB` | `0/16` | `0%` |

The dx gates reduced large-shift wrong-pose acceptance but still admitted some
low-dx pose aliases. The response gate rejected all far controls in this sweep,
but also rejected many valid anchors and lost most of the useful gain.

## Decision

Do not promote dx or response admission as the main path. The useful next move is
pose-distance admission using the `target_yaw` and `kf_frame_yaw` metadata already
available in the temporal anchor path.
