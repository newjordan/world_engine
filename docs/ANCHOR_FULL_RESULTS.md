# Phase 10c - Full-frame front-door anchoring results

**Status:** controlled n=6 sweep complete.
**Run date:** 2026-07-08.
**Condition source:** `docs/ANCHOR_FULL_PLAN.md`.
**Run label:** `new_experiment`.
**Runner:** `examples/anchor_probe.py`.
**Analyzer:** `examples/analyze_anchor.py`.

## Condition

Yaw-only return-to-start hysteresis, continuing the Phase 10 protocol:
`K=64`, settle 8, yaw_mag 0.2, hard regime heading <= 3.2,
2 scenes x 3 seeds, seed base 1234. Tuned canary:
`atlas_nowb` with `M=1`, restamp offset 4, `atlas_k=1`.
Post-intervention durability window: 64 no-op generated frames after the final
intervention, with no overlay, no further anchors, and no active atlas pins.

Command:

```bash
uv run --dev python examples/anchor_probe.py \
  --condition-source docs/ANCHOR_FULL_PLAN.md \
  --arms revisit,atlas_nowb,anchor_full,anchor_full_blend,anchor_confmap,anchor_full_far \
  --M 1 --restamp-offset 4 \
  --n-scenes 2 --seeds 3 \
  --run-label new_experiment \
  --csv bench_out/anchor_full/anchor_full_20260708T103626-0500.csv
```

Resolved by the runner:

- Model: `Overworld/Waypoint-1.5-1B`
- Device: `cuda`
- Quantization: `None`
- Scheduler grid: `[1.0, 0.8984375, 0.75, 0.30078125, 0.0]`
- Source hashes printed in log:
  - `anchor_probe=60a78ae480c51b6c`
  - `anchor=de3aa308191920e2`
  - `plan=3ab6b40b4c79170b`

Artifacts:

- Raw CSV: `bench_out/anchor_full/anchor_full_20260708T103626-0500.csv`
  (3477 rows)
- Log: `bench_out/anchor_full/anchor_full_20260708T103626-0500.log`
- Metric report:
  `bench_out/anchor_full/anchor_full_20260708T103626-0500_report/index.html`
- Summary JSON:
  `bench_out/anchor_full/anchor_full_20260708T103626-0500_report/summary.json`
- Frames-first visual review:
  `bench_out/anchor_full/anchor_full_20260708T103626-0500_visual/index.html`
- Visual manifest/log:
  `bench_out/anchor_full/anchor_full_20260708T103626-0500_visual/manifest.json`,
  `bench_out/anchor_full/anchor_full_20260708T103626-0500_visual.log`

`bench_out/` is gitignored. Preserve these artifacts before cleanup.

## Headline

The canary replicated:

| paired, hard regime (n=6) | delta PSNR | wins |
|---|---:|---:|
| `atlas_nowb - revisit` | `+3.45 +/- 0.94 dB` | `6/6` |

Full-frame front-door anchoring produced the first nearest-pose anchor signal that
beats the stopped-overlay comparator on average at post frame 16:

| paired at post16 (n=6) | delta PSNR | wins |
|---|---:|---:|
| `anchor_full - atlas_nowb` | `+1.20 +/- 1.21 dB` | `5/6` |
| `anchor_full - revisit` | `+2.61 +/- 0.76 dB` | `6/6` |
| `anchor_full_blend - atlas_nowb` | `+0.09 +/- 0.64 dB` | `4/6` |
| `anchor_full_blend - revisit` | `+1.50 +/- 0.50 dB` | `6/6` |
| `anchor_confmap - atlas_nowb` | `-0.41 +/- 1.08 dB` | `3/6` |
| `anchor_confmap - revisit` | `+1.00 +/- 0.70 dB` | `6/6` |
| `anchor_full_far - atlas_nowb` | `-0.97 +/- 0.29 dB` | `0/6` |
| `anchor_full_far - revisit` | `+0.45 +/- 0.52 dB` | `3/6` |

Mean post16 PSNR:

| arm | post16 PSNR |
|---|---:|
| `revisit` | `15.71` |
| `atlas_nowb` | `17.13` |
| `anchor_full` | `18.33` |
| `anchor_full_blend` | `17.21` |
| `anchor_confmap` | `16.71` |
| `anchor_full_far` | `16.16` |

## Diagnostics

| arm | accept | mean accepted \|dx\| | mean response | anchors/trial |
|---|---:|---:|---:|---:|
| `atlas_nowb` | 100% | `7.5 px` | `0.567` | 0 |
| `anchor_full` | 100% | `55.7 px` | `0.336` | 1 |
| `anchor_full_blend` | 100% | `55.7 px` | `0.336` | 1 |
| `anchor_confmap` | 100% | `55.7 px` | `0.336` | 1 |
| `anchor_full_far` | 50% | `~240.8 px` accepted cases | `0.198` accepted cases | 0.5 |

The wrong-pose control matters: `anchor_full_far` lost to `atlas_nowb` in 6/6
paired comparisons, while nearest-pose `anchor_full` beat it in 5/6. That argues
the full-frame gain is at least pose-specific, not just generic texture injection.

## Verdict

This is the strongest inference-side durability signal so far. Full-frame
front-door anchoring is a real representation improvement over the band-limited
front-door anchors: it beats raw revisit in 6/6 and beats the stopped-overlay
canary on average with 5/6 wins at post16.

It is not a formal pass under the original Phase 10 gate because the margin is
`+1.20 dB`, below the `+2 dB` target, and the accepted registration dx remains in
the large-shift band. The result should be promoted as a frontier/partial signal,
not as solved true durability.

## Boundary

This result covers Waypoint-1.5-1B, yaw-only return-to-start, one final full-frame
RGB reconstruction repeated 4x through `append_frame`, and the arms above. It does
not test:

- full-frame temporal four-frame micro-pose chunks;
- black/white primitive or surreal synthetic scenes;
- translation/out-and-back trajectories;
- trained persistence;
- live demo integration.

## Next Moves

1. Combine the two positive representation changes: full-frame plus four-frame
   temporal micro-poses.
2. Add a black/white primitive-scene lane for clearer visual diagnosis, while
   labeling it as potentially out-of-distribution for the current model.
3. Build a live compositor demo around `atlas_nowb` plus `anchor_full` as a
   manual/automatic loop-closure intervention.
