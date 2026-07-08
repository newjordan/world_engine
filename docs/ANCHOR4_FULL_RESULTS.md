# Phase 10d - Full-frame four-frame anchoring results

**Status:** controlled n=6 sweep complete.
**Run date:** 2026-07-08.
**Condition source:** `docs/ANCHOR4_FULL_PLAN.md`.
**Run label:** `new_experiment`.
**Runner:** `examples/anchor_probe.py`.
**Analyzer:** `examples/analyze_anchor.py`.
**Visual review:** `examples/anchor_visual.py`.

## Condition

Yaw-only return-to-start hysteresis, continuing the Phase 10 protocol:
`K=64`, settle 8, yaw_mag 0.2, hard regime heading <= 3.2,
2 scenes x 3 seeds, seed base 1234. Tuned canary:
`atlas_nowb` with `M=1`, restamp offset 4, `atlas_k=1`.
Post-intervention durability window: 64 no-op generated frames after the final
intervention, with no overlay, no further anchors, and no active atlas pins.

This phase combines the two prior positive representation changes:
full-frame registered memory plus four-frame temporal micro-pose chunks.

Command:

```bash
uv run --dev python examples/anchor_probe.py \
  --condition-source docs/ANCHOR4_FULL_PLAN.md \
  --arms revisit,atlas_nowb,anchor_full,anchor4_full,anchor4_full_blend,anchor4_full_far \
  --M 1 --restamp-offset 4 \
  --n-scenes 2 --seeds 3 \
  --run-label new_experiment \
  --csv bench_out/anchor4_full/anchor4_full_20260708T111752-0500.csv
```

Resolved by the runner:

- Model: `Overworld/Waypoint-1.5-1B`
- Device: `cuda`
- Quantization: `None`
- Scheduler grid: `[1.0, 0.8984375, 0.75, 0.30078125, 0.0]`
- Source hashes printed in log:
  - `anchor_probe=72ca8e432943f94a`
  - `anchor=de3aa308191920e2`
  - `plan=47e2951da7d8b44e`

Artifacts:

- Raw CSV: `bench_out/anchor4_full/anchor4_full_20260708T111752-0500.csv`
  (3477 rows)
- Log: `bench_out/anchor4_full/anchor4_full_20260708T111752-0500.log`
- Metric report:
  `bench_out/anchor4_full/anchor4_full_20260708T111752-0500_report/index.html`
- Summary JSON:
  `bench_out/anchor4_full/anchor4_full_20260708T111752-0500_report/summary.json`
- Frames-first visual review:
  `bench_out/anchor4_full/anchor4_full_20260708T111752-0500_visual/index.html`
- Visual manifest/log:
  `bench_out/anchor4_full/anchor4_full_20260708T111752-0500_visual/manifest.json`,
  `bench_out/anchor4_full/anchor4_full_20260708T111752-0500_visual.log`

`bench_out/` is gitignored. Preserve these artifacts before cleanup.

## Headline

The canary replicated:

| paired, hard regime (n=6) | delta PSNR | wins |
|---|---:|---:|
| `atlas_nowb - revisit` | `+3.39 +/- 0.97 dB` | `6/6` |

Full-frame four-frame front-door anchoring crossed the Phase 10d pass gate over
the stopped-overlay comparator at post frame 16:

| paired at post16 (n=6) | delta PSNR | wins |
|---|---:|---:|
| `anchor_full - atlas_nowb` | `+1.52 +/- 1.20 dB` | `6/6` |
| `anchor_full - revisit` | `+2.88 +/- 0.96 dB` | `6/6` |
| `anchor4_full - atlas_nowb` | `+3.05 +/- 2.05 dB` | `6/6` |
| `anchor4_full - revisit` | `+4.41 +/- 2.02 dB` | `6/6` |
| `anchor4_full_blend - atlas_nowb` | `+0.91 +/- 0.72 dB` | `5/6` |
| `anchor4_full_blend - revisit` | `+2.27 +/- 1.13 dB` | `6/6` |
| `anchor4_full_far - atlas_nowb` | `-0.94 +/- 0.36 dB` | `0/6` |
| `anchor4_full_far - revisit` | `+0.42 +/- 0.60 dB` | `3/6` |
| `anchor4_full - anchor_full` | `+1.53 +/- 1.77 dB` | `6/6` |

Mean post16 PSNR:

| arm | post16 PSNR |
|---|---:|
| `revisit` | `15.73` |
| `atlas_nowb` | `17.09` |
| `anchor_full` | `18.61` |
| `anchor4_full` | `20.14` |
| `anchor4_full_blend` | `18.00` |
| `anchor4_full_far` | `16.15` |

## Diagnostics

| arm | accept | mean accepted \|dx\| | mean response | anchors/trial |
|---|---:|---:|---:|---:|
| `atlas_nowb` | 100% | `7.65 px` | `0.541` | 0 |
| `anchor_full` | 100% | `55.64 px` | `0.309` | 1 |
| `anchor4_full` | 100% | `53.17 px` | `0.314` | 1 |
| `anchor4_full_blend` | 100% | `53.17 px` | `0.314` | 1 |
| `anchor4_full_far` | 50% | `232.78 px` accepted cases | `0.194` accepted cases | 0.5 |

The wrong-pose control is separated: `anchor4_full_far` lost to `atlas_nowb` in
6/6 paired comparisons, while nearest-pose `anchor4_full` beat `atlas_nowb` in
6/6 and beat the single-frame `anchor_full` comparator in 6/6 paired comparisons.

## Verdict

This is the first Phase 10 result that clears the stated `+2 dB` post16 gate over
the stopped-overlay comparator while also keeping a wrong-pose control below the
nearest-pose arm. The useful representation is hard full-frame temporal
reconstruction; the confidence-blended temporal variant underperformed it.

This is still an inference-side, yaw-only loop-closure result. It should be
reported as a controlled frontier pass for this condition, not as solved general
permanence. The result does not prove translation robustness, trained
persistence, live control integration, or behavior on the planned black/white
primitive surreal lane.

## Review URLs

When the local server is running from `bench_out/anchor4_full` on port 8783:

- Metric report:
  `http://100.124.153.1:8783/anchor4_full_20260708T111752-0500_report/index.html`
- Visual review:
  `http://100.124.153.1:8783/anchor4_full_20260708T111752-0500_visual/index.html`

Both pages returned `200 OK` over tailnet and were inspected with Chromium on
2026-07-08.

## Next Moves

1. Add an admission policy for hard full-frame temporal anchoring: keep the
   nearest-pose reconstruction, but reject low-response or implausibly large-dx
   cases before appending.
2. Port the condition to the black/white primitive surreal lane as
   `mechanics_proxy` or `ood_primitives`, not as a direct benchmark replacement.
3. Build a live loop-closure demo that can trigger `anchor4_full` on manual or
   automatic return-to-known-pose events.
