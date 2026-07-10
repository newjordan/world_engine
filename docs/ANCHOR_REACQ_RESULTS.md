# Phase 10g results - Content-defined loop-closure re-acquisition

**Status:** complete (Protocol A full; Protocol B one of three cases, see
boundary).
**Branch:** `spin-persistence`.
**Condition source:** `docs/ANCHOR_REACQ_PLAN.md` (including Pilot Amendments
10g.1 and 10g.2 — read them; two candidate gates were refuted in pilots before
the promoted algorithm emerged).
**Final algorithm (10g.2):** content-ranked multi-hypothesis retrieval inside a
spacing-aware pose-prior window (`max(3.0, 2.5 x median keyframe spacing)`),
resp/shift gates, cross-micro `admit_consist`, pose admission final gate,
explicit `nomatch` fail-closed state. Pose-plausibility residual is a logged
diagnostic only (refuted as a gate in pilot 2).

## Artifacts

- Pilots: `bench_out/anchor_reacq/pilot.csv` / `pilot2.csv` / `pilot3.csv`
  (+ logs). Pilot 1: wiring + far-control leak. Pilot 2: plausibility-gate
  refutation. Pilot 3: all scout gates green.
- Sweep CSV: `bench_out/anchor_reacq/anchor_reacq.csv` (7711 rows)
- Sweep log: `bench_out/anchor_reacq/sweep.log`
- Pan packet (scene 0): `bench_out/anchor_reacq/pan180_scene0_seed1235/`
  (`compare_2x2.mp4`, per-arm MP4s, `manifest.json`)
- Evidence stills: `docs/permanence_assets/reacq_pan180_scene0_start.png`,
  `docs/permanence_assets/reacq_pan180_scene0_return_collapse.png`
- Sweep source hashes: `anchor_probe=6ccd09c949481551`
  `anchor=cdd28604b439fb40` `plan=f5cd88e904552bf8`

## Protocol A — aggregate regression sweep (4 scenes x 4 seeds): PASS

Canary: `atlas_nowb - revisit = +2.94 +/- 1.07 dB, 16/16` (Phase 10f value
replicated exactly).

| arm | post16 vs `atlas_nowb` | wins | accept |
|---|---:|---:|---:|
| `anchor4_full_pose02` (incumbent) | `+3.93 +/- 3.34 dB` | `14/16` | `92%` |
| `anchor4_full_reacq_pose02` | `+3.75 +/- 2.66 dB` | `16/16` | `97%` |
| `anchor4_full_reacq_pose02_far` | `-1.08 +/- 0.55 dB` | `0/16` | `0%` |

All three gates pass: canary within +/- 1 dB; no regression vs incumbent (same
mean within noise, tighter spread, 16/16 wins); far control 0% accept with
16/16 explicit `nomatch` and post-window delta vs revisit exactly `+0.00`
(fail-closed = baseline, no side effects). The pose window is a structural
no-teleport guarantee: wrong-pose keyframes are never scored, no matter their
content response.

## Protocol B — 180-degree pan loop closure: boundary identified

Completed case (scene 0 / seed 1235, locked 217 away steps, measured 180.16
degrees):

| arm | anchor outcome | post16 | post64 |
|---|---|---:|---:|
| `revisit` | - | 11.60 | 11.74 |
| `atlas_nowb` | overlay could not register | 10.99 | 11.34 |
| `anchor4_full_pose02` | `reject=empty` (Phase 10f failure replicated) | 11.60 | 11.74 |
| `anchor4_full_reacq_pose02` | **explicit `nomatch`**: window 10.0 admitted 3 candidates, best in-window resp `0.031` < floor `0.05` | 11.60 | 11.74 |

**The verdict on the mechanism:** the MP4 frames show the returned view shares
*nothing* with the start scene — the neon-alley content has fully collapsed
into abstract smear during the 434-step round trip (see the two evidence
stills). The 180-degree boundary is **not a retrieval failure and not a weak
matcher: the remembered place no longer exists in the rollout at closure
time.** No closure-time matcher can reacquire content that is gone.
`nomatch` is the correct, honest answer, and the re-acquisition system now
says it explicitly instead of silently degrading (`reject=empty` overload is
resolved: `empty` = store empty; `nomatch` = searched and refused).

Incomplete cases, documented not hidden:

- scene 1 / seed 1236: reached 455 away steps (180.6 deg), then the runner was
  **killed by the kernel OOM killer** (54 GB RSS) during the third arm —
  in-memory frame accumulation for MP4 writing does not survive 455-step
  trajectories. Harness boundary, not a science result. Fix: stream frames to
  disk in `anchor_pan_video.py`.
- scene 2 / seed 1237: content rotation pathology replicated — only 30.5 deg
  of measured content rotation at the 800-step cap (prior attempt: same at
  cap 500). This scene/seed cannot express the 180-degree condition; a
  replacement case is needed.

## Verdict (per plan taxonomy): PARTIAL

The re-acquisition machinery is correct and promoted (Protocol A: full gain,
16/16, structural wrong-pose rejection; explicit no-match state delivered —
wishlist item 4). The full-180 claim is **re-scoped, not achieved**: at this
pan length the failure mode is total content drift, which no closure-time
mechanism can repair. The measured blocking values are recorded above
(best in-window resp 0.031 vs 0.05 floor).

## Stop Rule / Next Representation Change

- Do not tune `resp_min`, window size, or matcher strength to force a
  closure-time anchor on scene 0: the frames show there is nothing to match.
  That would be a threshold tweak against evidence.
- The next representation change is **mid-pan cadence anchoring**: anchor
  against fresh keyframes DURING the pan (content still registers over short
  horizons) so drift is bounded before closure, then loop-close on a world
  that still resembles memory. This changes when memory enters, not how it is
  found.
- Harness prerequisites for the next Protocol B round: streamed MP4 writing
  (fixes OOM at 455+ steps) and a validated third pan case to replace
  scene 2.
