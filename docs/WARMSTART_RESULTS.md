# Phase 9 — Warmstart / Re-dream Results

- **Branch:** `spin-persistence`
- **Run date:** 2026-07-07
- **Run code:** `3338c8e` (`examples/warmstart.py`, `examples/warm_probe.py`)
- **Review generator:** `examples/analyze_warm.py`
- **Condition source:** `docs/WARMSTART_PLAN.md`
- **Status:** controlled sweep complete; laundered write-back failed.

## Condition

Return-to-start hysteresis, same protocol as Phase 8: K=64, settle 8, yaw_mag 0.2,
hard regime heading <= 3.2, 2 scenes x 3 seeds, paired per-trial stats. Tuned atlas
density: `--M 1 --restamp-offset 4`.

Command:

```bash
uv run --dev python examples/warm_probe.py \
  --arms revisit,atlas,atlas_nowb,redream,redream_nowb,redream_far,atlas_redream \
  --M 1 --restamp-offset 4 \
  --n-scenes 2 --seeds 3 \
  --csv bench_out/warm/warm.csv
```

The live engine resolved the scheduler grid as `[1.0, 0.8984375, 0.75, 0.30078125,
0.0]` because the grid is stored in model dtype. `sigma_re=0.3` snapped to the live
grid point `0.30078125`, so the redream tail is still the 1-step grid tail.

Artifacts:

- Review page: `bench_out/warm/index.html`
- Raw CSV: `bench_out/warm/warm.csv` (1344 rows)
- Log: `bench_out/warm/warm.log`
- Summary JSON: `bench_out/warm/summary.json`
- Pilot CSV: `bench_out/warm/pilot.csv`
- Invalid first pilot log: `bench_out/warm/pilot_failed_offgrid_20260707T0012.log`

The first pilot attempt is invalid as a result: it completed `revisit`, then failed
before redream rows because `resume_index` rejected `0.3` against the live bf16 grid
value `0.30078125`. Commit `3338c8e` fixed that by snapping within dtype-grid tolerance
while still rejecting genuinely off-grid sigmas.

## Headline

| paired, hard regime (n=6) | delta PSNR | wins |
|---|---:|---:|
| atlas - revisit | +2.63 +/- 0.75 dB | 6/6 |
| atlas_nowb - revisit | **+3.42 +/- 0.92 dB** | 6/6 |
| redream - revisit | **-2.90 +/- 1.34 dB** | 0/6 |
| redream_nowb - revisit | +1.37 +/- 0.85 dB | 6/6 |
| redream_far - revisit | -2.79 +/- 1.66 dB | 0/6 |
| atlas_redream - revisit | -1.10 +/- 1.80 dB | 2/6 |
| **redream - redream_nowb** | **-4.28 +/- 1.63 dB** | **0/6** |
| redream - redream_far | -0.11 +/- 0.80 dB | 3/6 |
| atlas_nowb - atlas | +0.80 +/- 0.21 dB | 6/6 |
| atlas_redream - atlas | **-3.72 +/- 1.19 dB** | 0/6 |

Projection diagnostics, mean over per-trial stats:

| arm | accept | write-back | mean accepted \|dx\| | mean response |
|---|---:|---:|---:|---:|
| atlas_nowb | 100% | 0% | 7.5 px | 0.538 |
| redream | 87% | 87% | **130.3 px** | 0.268 |
| redream_nowb | 99% | 0% | **17.4 px** | 0.487 |
| redream_far | 43% | 43% | 119.8 px | 0.106 |
| atlas_redream | 98% | 98% | 61.7 px | 0.387 |

## Verdict

Laundered write-back does **not** restore durability. The exact channel-splitter says
the re-dreamed output is useful when it is not written back (`redream_nowb +1.37 dB`,
6/6), but writing the re-dreamed latent into KV makes the rollout worse than revisit
(`redream -2.90 dB`, 0/6) and far worse than the no-write-back isolation arm
(`redream - redream_nowb = -4.28 dB`, 0/6).

The dx fingerprint matches the Phase 8 disease in a new coat. With write-back,
mean accepted |dx| is 130 px; without write-back it is 17 px. The sampler laundering
made the edited latent a model output, but it did not make memory-bearing context edits
stable. The model still stalls its own dynamics once memory-bearing latents enter KV.

The incumbent inference recipe replicated cleanly: tuned atlas + output overlay
(`atlas_nowb`) is +3.42 dB over revisit and +0.80 dB over atlas, both 6/6. This matches
Phase 8's +3.40 dB record within run noise.

## Stop Rule

Do not keep launching sigma-0.3 laundered write-back runs or simple scheduler-dose
variants as if they are new evidence. The pilot and the full sweep are two verified
negative runs in the same representation class with the same failure basin:
memory-bearing write-back causes large dx and rollout degradation. A `redream75` run is
a dose tweak, not a new representation, unless explicitly requested as a diagnostic.

Valid next work must change the representation/search space or change the claim:

- write up the training-side result: persistence cannot be bolted onto this frozen
  model by writing memory-bearing latents into context, even after sampler laundering;
- pursue no-write-back image-channel improvements only, clearly scoped as overlay/init
  quality rather than durability;
- design a genuinely different mechanism such as trained persistence, context training,
  or a non-KV durability channel.

## Boundary

This result is controlled for Waypoint-1.5B, yaw-only return-to-start, K=64, sigma≈0.3,
and the Phase 9 runner. It does not test adaptive dosing, warm_fast, translation,
training-time persistence, or a model trained with explicit persistent memory targets.
