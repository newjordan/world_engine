# Phase 10 — Front-door Anchoring Results

- **Branch:** `spin-persistence`
- **Run date:** 2026-07-07
- **Condition source:** `docs/ANCHOR_PLAN.md`
- **Run label:** `new_experiment`
- **Runner:** `examples/anchor_probe.py`
- **Analyzer:** `examples/analyze_anchor.py`
- **Status:** controlled n=6 sweep complete; front-door anchoring did not pass the
  true-durability gate.

## Condition

Yaw-only return-to-start hysteresis, continuing the Phase 8/9 protocol:
`K=64`, settle 8, yaw_mag 0.2, hard regime heading <= 3.2,
2 scenes x 3 seeds, seed base 1234. Tuned comparator:
`atlas_nowb` with `M=1`, restamp offset 4, atlas_k 1. Post-intervention
durability window: 64 no-op generated frames after the final intervention, with
no overlay, no further anchors, and no active atlas pins.

Command:

```bash
uv run --dev python examples/anchor_probe.py \
  --arms revisit,atlas_nowb,anchor,anchor_blend,anchor_k1,anchor_k4 \
  --M 1 --restamp-offset 4 \
  --n-scenes 2 --seeds 3 \
  --run-label new_experiment \
  --csv bench_out/anchor/anchor.csv
```

Resolved by the runner:

- Model: `Overworld/Waypoint-1.5-1B`
- Device: `cuda`
- Quantization: `None`
- Scheduler grid: `[1.0, 0.8984375, 0.75, 0.30078125, 0.0]`
- Source hashes printed in log:
  - `anchor_probe=2e69a1f04104ebeb`
  - `anchor=4d5cd5d33ce72a14`
  - `plan=9e243be0aa3e3eee`

Artifacts:

- Raw CSV: `bench_out/anchor/anchor.csv` (3714 rows)
- Log: `bench_out/anchor/anchor.log`
- Review page: `bench_out/anchor/report/index.html`
- Summary JSON: `bench_out/anchor/report/summary.json`
- Visual review packet: `bench_out/anchor/visual_review/index.html`
- Visual review manifest/log: `bench_out/anchor/visual_review/manifest.json`,
  `bench_out/anchor/visual_review.log`

`bench_out/` is gitignored. Preserve these artifacts before cleanup.

## Visual Review

The first numeric report was hard to inspect because it had no frames. A separate
frames-first packet now reruns two representative trials under the same Phase 10
condition and saves raw frames plus contact sheets:

```bash
uv run --dev python examples/anchor_visual.py \
  --out bench_out/anchor/visual_review \
  --cases seed_01:1235:anchor_k4_good,seed_00:1234:anchor_k4_bad \
  --arms revisit,atlas_nowb,anchor,anchor_k4
```

This packet is a visual explanation of selected cases, not a replacement for the
n=6 aggregate CSV. It shows:

- start frame;
- return / anchor frame;
- post16;
- post64;
- post16 error heatmap against the start frame;
- small per-case metrics for the same visual rerun.

Use `bench_out/anchor/visual_review/index.html` for human inspection before the
numeric-only report.

## Headline

The canary replicated:

| paired, hard regime (n=6) | delta PSNR | wins |
|---|---:|---:|
| `atlas_nowb - revisit` | `+3.47 +/- 1.00 dB` | `6/6` |

The true-durability gate did not pass at post-intervention frame 16:

| paired at post16 (n=6) | delta PSNR | wins |
|---|---:|---:|
| `anchor - atlas_nowb` | `-0.78 +/- 1.11 dB` | `1/6` |
| `anchor_blend - atlas_nowb` | `-1.10 +/- 0.70 dB` | `0/6` |
| `anchor_k1 - atlas_nowb` | `-0.75 +/- 0.40 dB` | `0/6` |
| `anchor_k4 - atlas_nowb` | `-0.51 +/- 1.46 dB` | `3/6` |

Against raw revisit, anchors do help somewhat:

| paired at post16 (n=6) | delta PSNR | wins |
|---|---:|---:|
| `anchor - revisit` | `+0.57 +/- 0.72 dB` | `6/6` |
| `anchor_blend - revisit` | `+0.25 +/- 0.14 dB` | `6/6` |
| `anchor_k1 - revisit` | `+0.60 +/- 0.50 dB` | `5/6` |
| `anchor_k4 - revisit` | `+0.84 +/- 1.35 dB` | `5/6` |

Mean post16 PSNR:

| arm | post16 PSNR |
|---|---:|
| `revisit` | `15.72` |
| `atlas_nowb` | `17.07` |
| `anchor` | `16.29` |
| `anchor_blend` | `15.97` |
| `anchor_k1` | `16.32` |
| `anchor_k4` | `16.56` |

## Diagnostics

| arm | accept | mean accepted \|dx\| | mean response | anchors/trial |
|---|---:|---:|---:|---:|
| `atlas_nowb` | 100% | `7.6 px` | `0.564` | 0 |
| `anchor` | 100% | `56.4 px` | `0.297` | 1 |
| `anchor_blend` | 100% | `56.4 px` | `0.297` | 1 |
| `anchor_k1` | 100% | `38.4 px` | `0.595` | 32 |
| `anchor_k4` | 100% | `13.3 px` | `0.544` | 9 |

Single front-door anchors (`anchor`, `anchor_blend`) accepted registration but did
so with the large-dx disease fingerprint. Dense cadence (`anchor_k1`) improved
over revisit but still had large dx. Sparse cadence (`anchor_k4`) had the healthy
dx band and was the best anchor variant, but it still lost to the overlay-stopped
comparator on average and only won 3/6 at post16.

## Verdict

Front-door anchoring, as tested here, did **not** produce true durability. No
variant beat `atlas_nowb` with overlay/active atlas stopped by the required
`+2 dB` at post16, and no variant reached the 5/6 win gate.

The result is not a total no-signal: anchoring generally improves over raw revisit,
and `anchor_k4` is a plausible product-side cadence aid because it keeps dx healthy.
But scientifically it is a **FAIL for true durability** under the Phase 10 gate.
The model's own continuation is not repaired enough by appending the RGB atlas
reconstruction through the VAE front door.

## Boundary

This result covers Waypoint-1.5-1B, yaw-only return-to-start, repeated-one-frame
RGB reconstructions as `[4,720,1280,3]` append batches, the SLAM-band
reconstruction used by `examples/anchor.py`, and the six arms above. It does not
test full-frame appends without a band mask, multi-micro-pose 4-frame
reconstructions, translation, learned persistence, or a trained model.

## Pilot Caveats

The first pilot artifacts `bench_out/anchor/pilot.csv` and
`bench_out/anchor/pilot_report/` are invalid for post-window interpretation:
`atlas_nowb` left active atlas pins installed during the post window. The runner
was fixed before `pilot_fixed.csv`, `cadence_pilot.csv`, and the decisive
`anchor.csv` sweep. The pilot logs before the run-label patch also print
`Run label: new_experiment`; they should be treated as scout artifacts only.
