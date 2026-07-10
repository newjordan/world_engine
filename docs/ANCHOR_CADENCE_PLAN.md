# Phase 10h plan - Mid-pan cadence anchoring (bounded drift before closure)

**Status:** planned; harness prerequisites in flight, no result yet.
**Branch:** `spin-persistence`
**Condition source:** this file.
**Historical comparators:** `docs/ANCHOR_REACQ_RESULTS.md` (Phase 10g) and the
pan packet `bench_out/anchor_reacq/pan180_scene0_seed1235/` (locked 217 away
steps, measured 180.16 deg).
**Run label:** `new_experiment` (aggregate sweep); pilots are `scout`.

## Amendment 10h.1 - Far-cadence pool cardinality (2026-07-10)

The first 1x1 Protocol A wiring scout refuted the original far-control
construction, not the pose window. Raw artifacts:
`bench_out/anchor_cadence/pilot.csv` and `pilot.log`, source commit `28fb15e`.
At return headings 3.8, 3.0, and 2.2, the `mode="max", k=8` pool had only
5/4/7 of its eight candidates outside the 3.0-unit pose window. The remaining
in-window tail produced three real accepts (67% attempt accept rate overall),
with winner drifts -2.6, -2.6, and +2.8 yaw units. Thus the window was applied
correctly; the assumption that eight farthest candidates would all remain
outside it near the midpoint of a roughly 6.2-unit store span was false.

For the Phase 10h cadence `_far` falsification arm only, candidate cardinality
is amended to `k=1` before the unchanged pose window. The single farthest
keyframe remains outside the 3.0-unit window over this locked return path, so
the arm again tests the stated no-teleport invariant. Non-far cadence arms keep
`--reacq-k 8`, and the already-reported Phase 10g closure-only far condition is
unchanged. This is a control-construction repair, not matcher tuning: response,
shift, pose window, admission gates, cadence, and model condition are frozen.
The scout must be rerun from committed code and show every cadence-far attempt
as `nomatch` before the pilot may launch.

## Question

Phase 10g proved the closure-time re-acquisition machinery is correct and
honest: at 180-degree loop closure the returned content shares nothing with the
start scene (best in-window resp 0.031 < 0.05 floor) and the system says
`nomatch`. The wall is **content death during the pan**, not retrieval. The
stop rule in `docs/ANCHOR_REACQ_RESULTS.md` forbids tuning the matcher against
it.

The question: **does anchoring DURING the return pan — cadence re-acquisition
against pose-windowed keyframes captured on the away leg — bound content drift
so that the world still resembles memory at closure, letting the promoted 10g
machinery close the 180-degree loop?**

## Representation Change (stop-rule test)

This changes **when memory enters the rollout** (during-trajectory cadence vs
closure-only), not how memory is found:

- the 10g retrieval pipeline is frozen: pose-prior window
  (`max(3.0, 2.5 x median keyframe spacing)`), content ranking, resp/shift
  gates (`resp_min=0.05` unchanged), cross-micro `admit_consist`, pose
  admission final gate, explicit `nomatch` fail-closed;
- the only new parameter is the cadence period `C` — inherited from the Phase
  10c-d cadence lineage (`anchor4_k4` healthy dx; dense `anchor_k1` showed the
  large-dx disease — carried forward as evidence, not re-tuned);
- the new claim is **bounded drift under continuous ego-motion**: each accepted
  mid-pan anchor re-registers the rollout against memory while the temporal gap
  to the matching away-leg keyframe is still short, so drift cannot compound
  into content death by closure time.

Boundary unchanged from 10f/10g: RGB full-frame temporal micro-poses in
`AnchorStore`; reconstruction enters only through `engine.append_frame`; no KV
editing (standing stop rule); `camera_yaw` never mutated; every rejected
attempt is a no-op (fail closed, no silent fallback).

## Algorithm

For cadence-reacq arms, on the **return leg only**: every `C` return steps
(plus the final closure step, same `needs_final_anchor` semantics as
`anchor_probe.py`), run the full 10g reacq pipeline against the store. Away
leg unchanged (keyframe inserts every `M` steps, no anchoring).

Why return-leg only: away-leg content at the frontier is novel by
construction — there is nothing at that pose to re-acquire, and reconstruction
would fight generation. The return leg has same-pose keyframes from the away
pass; the temporal gap grows from ~0 at turnaround to ~2N steps at closure.
Each accepted cadence anchor re-registers content while the gap is short.
Hypothesis: closure-time in-window response stays >= the 0.05 floor instead of
collapsing to 0.031.

A cadence attempt that rejects (`nomatch`, `admit_consist`, `resp`, `shift`,
`admit_pose*`) appends nothing and the rollout continues — the arm must degrade
gracefully to the incumbent's behavior, never worse by construction.

## Resolved Condition

Identical to Phase 10g (`docs/ANCHOR_REACQ_PLAN.md`) except the arms:

- Model: `Overworld/Waypoint-1.5-1B`, one CUDA GPU, no quant unless `--quant`
- Assets: `bench_assets`
- Protocol A: yaw-only return-to-start hysteresis, `K=64`, `settle=8`,
  `yaw_mag=0.2`, 4 scenes x 4 seeds, seed base 1234
- Protocol B: content-measured 180-degree pan, `yaw_mag=4.0`, `cap=800`,
  locked away-steps replayed across arms
- Canary: `atlas_nowb`, `M=1`, restamp offset 4, `atlas_k=1`
- Metric: paired post16 PSNR vs start; accept/reject accounting; dx healthy
  band (~7-20 px) for accepted anchors

## Arms

| arm | mechanism | purpose |
|---|---|---|
| `revisit` | no memory aid | baseline |
| `atlas_nowb` | tuned overlay during return, stopped for post window | canary |
| `anchor4_full_reacq_pose02` | 10g promoted arm (closure-only reacq) | incumbent |
| `anchor4_full_reacq_k4_pose02` | cadence C=4 return-leg reacq + closure | candidate |
| `anchor4_full_reacq_k4_pose02_far` | same cadence, farthest candidate pool before the window | falsification control |

The far control must reject every cadence attempt via the pose window
(`nomatch`) — dozens of attempts per trajectory instead of one, so it is a
*stronger* no-teleport test than 10g's. Zero appends, post-window delta vs
revisit exactly +0.00.

Naming: `_far` stays terminal (runner suffix checks); `k4` token extends
`_cadence()` parsing. Pilot may compare `k8` as the single alternate — two
cadence values max, this is not a dose sweep (NORTHSTAR rule).

## Gates

Protocol A (regression guard — the mild protocol's return leg is already
pose-consistent, so cadence reacq should select the same keyframes):

- Canary `atlas_nowb - revisit` within +/- 1 dB of +2.94 dB, else stop and
  debug the harness.
- `anchor4_full_reacq_k4_pose02` must not regress vs the 10g incumbent:
  paired post16 delta within noise or better; accepted-anchor dx stays in the
  healthy band (no `anchor_k1`-style dense-cadence disease).
- Far control: ~0% accept, explicit `nomatch`, post-window delta exactly +0.00.

Protocol B (the headline — 3 valid pan cases: scene 0 / seed 1235 at locked
217 steps; scene 1 / seed 1236 after the streamed-MP4 fix; a validated
replacement third case):

- Replicate the known boundary: incumbent `nomatch` at closure on scene 0
  (best in-window resp ~0.031).
- **PASS:** the cadence arm accepts mid-pan anchors with healthy dx AND the
  closure anchor is accepted (resp >= 0.05, dx healthy) in at least 2 of 3
  cases, with post-window PSNR beating the incumbent's rejected-anchor
  outcome, and the far control never appending.
- **PARTIAL:** mid-pan anchors are accepted with healthy dx but closure still
  ends `nomatch` — report the resp-vs-return-step curve and where
  re-registration stops holding (diagnosis, not a threshold license). Also
  PARTIAL: closure resp improves materially over 0.031 but stays sub-floor
  (report the measured value; do not lower the floor).
- **FAIL:** cadence triggers the dx-runaway/stall fingerprint, or the far
  control appends, or the post window degrades vs the incumbent. Document as
  the negative and stop this lane.

## New Diagnostics To Log

Per cadence attempt (CSV `intervention` rows and pan-video manifest): return
step index, accept/reject token, winner `kf_seq`, winner yaw vs dead-reckoned
yaw, win resp, runner-up resp, dx. The money plot is **closure-time best
in-window resp per arm** (incumbent baseline 0.031) and the
**resp-vs-return-step curve** showing drift bounded (or not) as the temporal
gap grows.

## Prerequisites (harness, before any Protocol B case)

1. **Streamed MP4 writing** in `anchor_pan_video.py` — in-flight. The scene 1
   case needs 455 away steps; the old in-memory path OOMs at 54 GB RSS, and
   the GPU currently co-tenants a 58 GB vLLM server (leanstral), so the fix is
   mandatory, not optional.
2. **Cadence path in `anchor_pan_video.py`** — the runner currently anchors
   once at closure only; add return-leg cadence mirroring `anchor_probe.py`'s
   `b % cadence == 0` plumbing plus `needs_final_anchor`.
3. **Validated third pan case:** measured content rotation must reach ~170+
   deg at cap 800 before the case is citable (scene 3 or a new seed; scene 2
   is pathological — 30.5 deg at cap 800, twice replicated).

## Pilot (before the sweep, run label `scout`)

1x1 on scene 0 / seed 1235 at the locked 217 steps: wiring; C=4 vs C=8 single
comparison (pick one, record why); far-control behavior over many attempts.
If a gate is refuted in pilot, amend this file with the evidence per 10g
practice (amendments 10g.1/10g.2 are the template) before any sweep.

## Runbook

Aggregate regression sweep (after pilot):

```bash
uv run --dev python examples/anchor_probe.py \
  --condition-source docs/ANCHOR_CADENCE_PLAN.md \
  --arms revisit,atlas_nowb,anchor4_full_reacq_pose02,anchor4_full_reacq_k4_pose02,anchor4_full_reacq_k4_pose02_far \
  --M 1 --restamp-offset 4 \
  --n-scenes 4 --seeds 4 \
  --run-label new_experiment \
  --csv bench_out/anchor_cadence/anchor_cadence.csv
```

Long-pan MP4 packets (one per valid case, locked away-steps):

```bash
uv run --dev python examples/anchor_pan_video.py \
  --condition-source docs/ANCHOR_CADENCE_PLAN.md \
  --arms revisit,atlas_nowb,anchor4_full_reacq_pose02,anchor4_full_reacq_k4_pose02 \
  --target-deg 180 --yaw-mag 4.0 --cap 800 \
  --scene 0 --seed 1235 \
  --out bench_out/anchor_cadence/pan180_scene0_seed1235
# repeat for scene 1 / seed 1236 and the validated third case
```

MP4 packet requirement unchanged from 10g: `compare_2x2.mp4`, per-arm MP4s,
`manifest.json` with source hashes, locked away-steps, per-attempt anchor
status, winner identities, post-window PSNR. Runners print source hashes and
the resolved condition at start.

## Result Language

Videos are controlled review artifacts for specific cases; the sweep CSV
carries the aggregate regression claim and the three-case pan protocol carries
the loop-closure claim. Protocol A and Protocol B post16 values are different
conditions and must not be pooled. One MP4 is never a benchmark.
