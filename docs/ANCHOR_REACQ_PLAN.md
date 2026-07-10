# Phase 10g plan - Content-defined loop-closure re-acquisition

**Status:** planned / runnable, no result yet.
**Branch:** `spin-persistence`
**Condition source:** this file.
**Historical comparators:** `docs/ANCHOR_POSE_ADMISSION_RESULTS.md` (Phase 10f)
and the pan180 review packet
`bench_out/anchor_pose_admission/pan180_scene0_seed1235_20260708T155108-0500/`.
**Run label:** `new_experiment` (aggregate sweep); `mechanics_proxy` does not
apply — assets remain `bench_assets`.

## Question

Phase 10f proved pose-distance admission preserves the full `anchor4_full` gain
(+4.35 dB vs `atlas_nowb`, 15/16) while rejecting far-pose controls. Its boundary
is long-pan loop closure: in the 180-degree pan case the final anchor attempt
rejects with `reject=empty` — the yaw-nearest keyframe is retrieved, but after
~430 dead-reckoned steps the pose key is wrong enough that all four micro-frames
fail phase correlation (`resp < resp_min`), and the summary collapses to `empty`
(`_summarize_micro_infos`).

Retrieval today is `AnchorStore.query`: single hypothesis, linear `abs(yaw)`
distance, keyed by dead-reckoned yaw. The question: **does content-defined
multi-hypothesis retrieval re-acquire the correct keyframe at long-pan loop
closure, admit it through the existing pose gate, and produce a real post-window
gain where the incumbent rejects?**

## Representation Change (stop-rule test)

This changes the retrieval/search space, not a threshold:

- retrieval goes from single yaw-nearest keyframe to a ranked candidate search
  over the whole store (k-nearest yaw bins, scored by registration content
  response);
- the dead-reckoned pose estimate is corrected from the winning content match
  before reconstruction (pose reset to the matched keyframe's stored micro-yaws);
- a new cross-micro consistency gate requires the four micro-frame registrations
  to agree (dx spread bound) before append;
- an explicit `nomatch` state distinguishes "searched, nothing correlates" from
  "store empty"; both fail closed with no append.

Boundary unchanged from Phase 10f:

- source memory remains RGB full-frame temporal micro-poses in `AnchorStore`;
- reconstruction enters only through `engine.append_frame`;
- no direct KV editing (standing stop rule);
- pose admission (`pose02`) remains the final gate before append;
- the engine's internal `camera_yaw` is never mutated; the corrected yaw is used
  only for reconstruction and logging.

## Algorithm (to implement in `examples/anchor.py` + runners)

At anchor time, for `reacq` arms:

1. **Candidate search:** `AnchorStore.query_k(yaw, k)` returns up to `k=8`
   nearest-yaw keyframes (whole store allowed if smaller). For the `_far`
   control, the candidate pool is the `k` farthest instead.
2. **Content scoring:** register the current ROI against each candidate's ROI
   (`rigid.register` phase correlation). Rank by response. Optional cheap
   prefilter via `slam._fingerprint` NCC is allowed but must be recorded in the
   run provenance if used.
3. **Winner + pose correction:** best candidate must have `resp >= resp_min`
   (unchanged 0.05) and `|dx| <= max_shift_frac * W` (unchanged). Target yaws
   for the temporal chunk are reset to the winner's stored `micro_yaws`
   (content-defined pose, replacing the dead-reckoned grid).
4. **Temporal reconstruction:** existing `reconstruct_temporal_batch` machinery
   against the winning keyframe (dx-shifted masked band paste per micro-frame,
   as in Phase 10d-f).
5. **Consistency gate (new):** the four per-micro registrations must agree:
   `max_j |dx_j - median(dx)| <= 24 px` (healthy accepted band is ~7-20 px).
   Violation rejects with `admit_consist`.
6. **Pose admission (final gate, unchanged):** `pose02` distance check over the
   projected micro-frames vs the winner's stored micro-yaws.
7. **No-match state:** if no candidate passes step 3, reject with `nomatch`
   (new token). `empty` now means the store itself is empty, nothing else.
   Both return the current frame unmodified — fail closed, no silent fallback.

## Resolved Condition

- Model: `Overworld/Waypoint-1.5-1B`
- Device: one CUDA GPU
- Quantization: `None` unless explicitly passed with `--quant`
- Dataset/assets: `bench_assets`
- Protocol A (aggregate regression sweep): yaw-only return-to-start hysteresis,
  `K=64`, `settle=8`, `yaw_mag=0.2`, 4 scenes x 4 seeds, seed base 1234
- Protocol B (long-pan criterion): content-measured 180-degree pan
  (`anchor_pan_video.py`), `yaw_mag=4.0`, `cap=800`, locked away-steps replayed
  across arms
- Tuned canary: `atlas_nowb`, `M=1`, restamp offset 4, `atlas_k=1`
- Metric: paired post16 PSNR against start; per-arm accept/reject accounting;
  dx fingerprint must stay in the healthy band for accepted anchors

## Arms

| arm | mechanism | purpose |
|---|---|---|
| `revisit` | no memory aid | baseline |
| `atlas_nowb` | tuned overlay during return, stopped for post window | canary |
| `anchor4_full_pose02` | Phase 10f promoted arm (single-hypothesis retrieval) | incumbent |
| `anchor4_full_reacq_pose02` | content-defined multi-hypothesis re-acquisition + consistency gate + pose admission | candidate |
| `anchor4_full_reacq_pose02_far` | same pipeline, candidate pool restricted to farthest-yaw bins | wrong-pose control |

The far control is the falsification arm: a looser search must NOT buy its gain
by admitting wrong-pose content. It should reject via low response, the
consistency gate, or pose admission.

## Gates

Aggregate sweep (Protocol A — regression guard):

- Canary `atlas_nowb - revisit` within +/- 1 dB of +2.94 dB (Phase 10f value);
  otherwise stop and debug the harness.
- `anchor4_full_reacq_pose02` must not regress vs `anchor4_full_pose02`: paired
  post16 delta within noise or better (the mild protocol is already
  pose-consistent, so re-acquisition should select the same keyframes).
- `anchor4_full_reacq_pose02_far` must stay at or below canary with ~0% accept.

Long-pan criterion (Protocol B — the headline):

- Replicate the known failure: `anchor4_full_pose02` rejects (`reject=empty`
  under the old vocabulary) at loop closure on scene 0 / seed 1235.
- **PASS:** `anchor4_full_reacq_pose02` anchors at loop closure (accepted, dx in
  healthy band) in at least 2 of 3 pan cases, with post-window PSNR vs start
  beating the incumbent's rejected-anchor outcome, and the far control never
  anchoring.
- **PARTIAL:** re-acquisition finds the correct keyframe (logged winner matches
  the settle-time keyframe) but admission rejects it; report the blocking gate
  and its measured values. This is a diagnosis result, not a threshold-tweak
  license.
- **FAIL:** re-acquisition admits wrong-pose anchors (far control anchors, or
  the winner is visibly wrong in the MP4) or degrades the post window. Document
  as the negative and stop this lane.

Pan cases: scene 0 / seed 1235 (known-failure replication), scene 1 / seed 1236,
scene 2 / seed 1237. Note: the prior scene 2 attempt at cap 500 did not reach
the 180-degree content target; cap is 800 here for all cases.

## Runbook

Aggregate regression sweep:

```bash
uv run --dev python examples/anchor_probe.py \
  --condition-source docs/ANCHOR_REACQ_PLAN.md \
  --arms revisit,atlas_nowb,anchor4_full_pose02,anchor4_full_reacq_pose02,anchor4_full_reacq_pose02_far \
  --M 1 --restamp-offset 4 \
  --n-scenes 4 --seeds 4 \
  --run-label new_experiment \
  --csv bench_out/anchor_reacq/anchor_reacq.csv
```

Long-pan MP4 packets (one per case):

```bash
uv run --dev python examples/anchor_pan_video.py \
  --condition-source docs/ANCHOR_REACQ_PLAN.md \
  --arms revisit,atlas_nowb,anchor4_full_pose02,anchor4_full_reacq_pose02 \
  --target-deg 180 --yaw-mag 4.0 --cap 800 \
  --scene 0 --seed 1235 \
  --out bench_out/anchor_reacq/pan180_scene0_seed1235
# repeat with --scene 1 --seed 1236 and --scene 2 --seed 1237
```

MP4 packet requirement: every serious Protocol B case ships `compare_2x2.mp4`
plus per-arm MP4s and `manifest.json` carrying source hashes, the locked
away-steps, per-arm anchor status, winner keyframe identity, and post-window
PSNR. Runners must print source hashes and the resolved condition at start, as
the current runners do.

## New Diagnostics To Log

Per anchor attempt (CSV `intervention` rows and pan-video manifest):

- candidate count searched, winner `kf_seq`, winner yaw vs dead-reckoned yaw
  (the measured drift), winner response, runner-up response;
- reject vocabulary: `nomatch` (searched, none passed), `empty` (store empty),
  `admit_consist` (dx spread), existing `resp`/`shift`/`admit_pose*` unchanged.

## Pilot Amendment (10g.1)

The 1x1 scout (`bench_out/anchor_reacq/pilot.csv`, run label `scout`, hashes
`anchor_probe=d59972b0606a80c3 anchor=2c43cb7b58575357`) validated wiring and
the no-regression gate (`reacq +2.45` vs incumbent `+2.64` dB over
`atlas_nowb`), but the far control **anchored** (win_resp 0.237, dx 207.9 px,
cross-micro consistent), violating the ~0% accept gate. Root cause: with pose
reset, pose admission is vacuous and cross-micro consistency passes for any
coherent wrong-pose match; response ranking cannot separate a true match from
a symmetric-content alias.

Amendment — the reacq algorithm gains a **pose-plausibility admission** step
between steps 2 and 3, part of the algorithm (not an arm variant):

- calibrate `px_per_yaw` from the store itself: register adjacent keyframe
  ROIs at known command-yaw separation; robust median over pairs that pass
  resp/shift gates;
- a candidate is plausible iff `|dx/px_per_yaw - (kf_yaw - camera_yaw)| <=
  1.5` yaw units; implausible candidates are skipped during the ranked search
  (diagnostics count them);
- if calibration is impossible (no registerable adjacent pairs) or no
  plausible candidate passes the content gates, reject `nomatch` — fail
  closed, never fall back to trusting raw response.

Pilot evidence this separates truth from alias: true match `dx 69.9 px` at
`drift 1.0` (69.9 px/yaw, residual 0.0); alias `dx 207.8 px` at `drift 5.2`
(residual 2.2 yaw units, rejected at 1.5).

## Pilot Amendment 2 (10g.2) — plausibility gate refuted, pose window adopted

The 10g.1 re-pilot (`bench_out/anchor_reacq/pilot2.csv`) refuted the
plausibility gate. With the store-measured calibration (`ppy=31.431`,
capture-time adjacent pairs), the true match's residual (1.39 yaw units,
`drift 0.2, dx 50.0`) and the far alias's residual (1.44, `drift 5.2,
dx 208.6`) are indistinguishable; both passed at bound 1.5 and the far control
anchored again. 10g.1's supporting arithmetic used the winner's own dx/drift
ratio as the calibration — circular. Two structural confounds: px-per-yaw gain
is context-unstable (~31 px/yaw between adjacent capture pairs vs ~70 px/yaw
against a drifted return view), and legitimate content drift enters the
residual exactly like wrong-pose aliasing. The residual has no discriminating
power. Negative finding, documented, gate withdrawn.

Positive pilot2 finding: content-ranked selection inside the near pool beat
the incumbent (`reacq +4.16` vs incumbent `+2.70` dB over `atlas_nowb`, one
trial) — the search itself has value once the pool is honest.

Amendment — replace the plausibility hard gate with a **pose-prior window on
the candidate pool** (this is the original handoff spec: "multi-hypothesis
retrieval over yaw bins *near* loop closure"):

- candidate pool = keyframes with `|kf.yaw - camera_yaw| <= window`, where
  `window = max(3.0 cmd-yaw units, 2.5 x median adjacent keyframe spacing)`
  (spacing-aware so the pool spans several keyframes under both the 0.2-unit
  mild protocol and the 4.0-unit pan protocol);
- within the pool: rank by response, resp/shift gates, cross-micro
  `admit_consist`, pose admission unchanged;
- the plausibility residual and `px_per_yaw` remain **logged diagnostics
  only**;
- empty pool or no passing candidate → `nomatch`, fail closed;
- the `_far` control keeps its farthest-pool selection *before* the window is
  applied, so it must reject via the window (`nomatch`) — the memory system
  refuses to teleport regardless of content response.

## Result Language

Do not call one MP4 a benchmark. Videos are controlled review artifacts for
specific cases; the sweep CSV/report carries the aggregate regression claim and
the three-case pan protocol carries the loop-closure claim. Post16 PSNR from
Protocol A and Protocol B are different conditions and must not be pooled.
