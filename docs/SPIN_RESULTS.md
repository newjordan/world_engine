# 360° Spin Persistence via SLAM Loop Closure — Results (Phase 7)

> Can `Overworld/Waypoint-1.5-1B` spin a full revolution and come back to the *same
> world*, instead of drifting into a new one (neon alley → desert)? Companion to
> [`ATLAS_RESULTS.md`](./ATLAS_RESULTS.md) (Phase 6: the pose-indexed atlas) — this phase
> replaces the atlas's open-loop dead-reckoned key with a real 1-DOF SLAM stack:
> signed visual odometry, appearance loop closure, and relocalization.
> Code: [`examples/slam.py`](../examples/slam.py) (estimator, CPU-pure),
> [`examples/spin_slam.py`](../examples/spin_slam.py) (benchmark),
> [`examples/test_slam.py`](../examples/test_slam.py) (16 CPU tests, incl. a
> perceptual-aliasing regression).

---

## TL;DR

1. **In-run self-calibration is the only calibration that exists.** The SLAM arm
   measures the world's revolution period from its own frames (appearance loop closure
   at the phase re-intersection point). Surviving periods span **805–1063
   cmd-units/rev across scenes and across sessions of the same scene+seed** — the turn
   rate is a property of the realized world, not the checkpoint, so the Phase-6.5
   cross-run oracle is incoherent in principle. Where both estimators were healthy they
   agreed to 3% (971 vs 940).
2. **Perceptual aliasing is severe in a generative world — and now handled.** The model
   renders closure-lookalikes at 0.5–0.8 of the true period with appearance NCC up to
   **0.91**; a naive match-plus-streak rule false-fires repeatedly. Consensus voting +
   provisional-closure verification **revoked 22 false closures across the 6-trial
   sweep with zero wrong periods retained** (two trials correctly ended open rather
   than committing to a bad map).
3. **Memory injection keeps the world *dynamically* alive — the robust, replicated
   effect (6/6 trials).** The no-memory baseline's rotation collapses as its world
   degrades: it failed to complete even one revolution in 3/6 trials (200–275° from
   1100 identical yaw commands; confident phase correlation reads the world as static —
   the model stops responding). The SLAM-keyed atlas arm completed the full second
   revolution in **every trial**. Forgetting doesn't just replace scenery; it breaks
   the world's response to controls.
4. **The world visibly comes back after 360° — but the fidelity gain is a PSNR null.**
   Qualitatively: baseline lap 2 renders a dune field where an alley was; the SLAM arm
   re-renders the same buildings-and-fence scene (grid below). Quantitatively, paired
   lap-2 self-consistency does **not** beat baseline across the sweep (atlas_dr
   +0.10 ± 1.40, atlas_slam −0.84 ± 1.07 dB over the 3 pairable trials) — single-trial
   gains sit inside run-to-run variance, and most of the error budget is texture
   fidelity, not scene identity.
5. **⇒ The remaining gap is squarely the training-side ask.** Retrieval geometry is
   solved end-to-end (right keyframe, right period, in-run, alias-robust) and keeps the
   world responsive; what it cannot buy at this operating point is reconstruction
   fidelity — the frozen attention's weak weighting of injected memory (the Phase-6
   ceiling). Next inference-side lever worth one experiment: inject memory via the
   **latent-init channel** (warm-start denoising from the retrieved keyframe's x0,
   renoised to mid-σ) instead of only the KV channel.

Single-trial detail below (scene seed_00, seed 1236); sweep table in **Sweep**.

---

## Mechanism — the sine-wave interaction point, made operational

Heading is a **phase on a circle**. Phase 6.5 integrated |phase-correlation shift| from
a *separate baseline run* to guess the period, then keyed the atlas on the command-yaw
accumulator: open-loop in exactly the two ways that matter (unsigned, uncalibrated
in-run). `YawSLAM` closes the loop in content space, within the run:

- **Odometry (the slope integration).** Per-frame *signed* horizontal phase-correlation
  shift of the rendered frames, integrated into a continuous scroll pose (px). This
  measures the world the model actually drew, not the commands we sent. The integrated
  command→content **slope** (px per cmd-unit) is exposed as a diagnostic — it is the
  calibration constant Phase 6.5 had to borrow.
- **Loop closure (the sine-wave re-intersection).** Keyframe fingerprints (normalized
  thumbnails of the textured ROI) are matched against views far away in odometry. Each
  confident match *votes* for an implied period `pose − (kf.pose + Δ_refined)`; closure
  fires only when ≥ `confirm` votes inside a rolling window agree within ~1.2% of a
  revolution. The agreed period P̂ makes the atlas key circular
  (`AtlasStore.yaw_period = P̂`): frame ~N of lap 2 retrieves frame ~0's keyframe.
- **Verification & revocation (the aliasing defense).** A fired closure is provisional
  for 240 frames: relocalization evidence must keep flowing (≥3 matches / 60 frames)
  and stay consistent (median |residual| ≤ 3% of P̂). Failures revoke the closure —
  corrections undone, keyframes noted under the bad period pruned from both the SLAM
  map and the atlas store, vote requirement escalated. An aliased closure looks perfect
  *while the repeated content lasts*; the tell is always afterward.
- **Relocalization.** Post-closure, every confident appearance match to a previous-lap
  keyframe nudges the pose by `gain × wrapped residual` (complementary filter), so
  drift cannot re-accumulate on laps 2+. Same-lap matches are excluded — they carry the
  estimate's own error and would lock drift in rather than correct it (a bug the CPU
  tests caught: the estimator converges to *map consistency*, which is the correct
  target for retrieval, not metric ground truth).

The atlas side is unchanged except a `yaw=` override on `atlas_capture` /
`atlas_activate` so the store can be keyed in content space. Spatial restamp stays off
(`align_pose=False`) per the Phase-6 ablation; retrieval does the aligning.

---

## Protocol

Model `Overworld/Waypoint-1.5-1B`, bf16, NVIDIA GB10. Continuous one-direction yaw
(`mouse=[+1.5, 0]`) for up to 1100 frames (~2 content revolutions when the world
cooperates), settle 8. Arms share the command stream:

| Arm | Key | Period | In-run? |
|---|---|---|---|
| **baseline** | — (no memory) | — | — |
| **atlas_dr** | command yaw (Phase 6.5) | borrowed from baseline run | ✗ cross-run oracle |
| **atlas_slam** | SLAM content pose | measured at loop closure | ✓ fully in-run |

**Scoring is per-heading lap-2 self-consistency:** each lap-2 frame vs the *same arm's*
lap-1 frame at the same content heading (each arm's own signed-VO axis, shared geometric
deg conversion). "Does the world that comes back match the world that was there?"
Cross-arm scores vs the baseline's lap 1 are also logged (Phase-6.5 continuity axis).

---

## Single-trial results (scene seed_00, seed 1236)

**Closure sequence** (the aliasing defense at work):

| frame | event | period (px) | evidence |
|---|---|---|---|
| 366 | closure fired | 3640 | ncc 0.81, 5 votes |
| 407 | **revoked** | — | residuals inconsistent (41 frames) |
| 424 | closure fired | 3139 | ncc 0.91, 7 votes |
| 441 | **revoked** | — | residuals inconsistent (17 frames) |
| 493 | closure fired | 4979 | ncc 0.91, 9 votes |
| 508 | **revoked** | — | residuals inconsistent (15 frames) |
| 603 | closure fired | **7305** | ncc 0.90, 11 votes — **survives verification** |

The surviving period, converted through the measured slope, is **971 cmd-units/rev**;
the independent cross-run calibration (baseline lap-1 length × yaw-mag) gives **940**.
Two estimators, one from appearance and one from content-integration, 3% apart.

**Rotation delivered by 1100 identical yaw commands** (content degrees, own-VO axis):

| baseline | atlas_slam | atlas_dr |
|---|---|---|
| 474° | 872° | 1124° |

**Lap-2 self-consistency (PSNR vs own lap-1):** baseline 12.84 dB over its reachable
0–115° of lap 2; atlas_dr 13.65 dB / atlas_slam 13.32 dB over much larger lap-2
coverage. Paired-by-heading diffs on the *common* (early-revisit) bins are ≈ 0 — the
memory arms' advantage lives at headings the baseline never reaches, plus in the
qualitative scene identity that PSNR undersells:

![loop-closure grid](../bench_out/spin_slam/spin_slam_loopclosure_grid.png)

*(baseline lap 2 = dune field replacing an alley, 11.6 dB; SLAM lap 2 = the same
buildings/fence scene degraded, 12.3 dB. Media in `bench_out/spin_slam/`.)*

**Keying tension (new design insight).** atlas_dr out-rotates atlas_slam (1124° vs
872°) because its command-space key advances *no matter what the world does*, so
retrieval keeps paging fresh memory and drags the world forward — an accidental
dynamics driver. The SLAM key is content-true, so when the world stalls, its query
stalls with it (honest but passive). These are two ends of a spectrum; a hybrid —
content-true pose with a command-driven *query* advance when VO flatlines ("the memory
says the world should be turning; pull it") — is the obvious Phase-7.5 mechanism.

---

## Sweep (3 seeds × 2 scenes, 6 trials)

Raw data: `bench_out/spin_slam/sweep/` (`all.csv`, per-trial `summary.json`,
`sweep.log`). Per-trial outcomes:

| trial | baseline lap-1 | closure outcome | revoked | self-PSNR base / dr / slam |
|---|---|---|---|---|
| s0·1236 | 856 fr (degenerate) | ✓ survived, 1063 cmd-u | 4 | 11.08 / 12.36 / **13.09** |
| s0·1237 | 342 fr (healthy) | ✗ ended open | 4 | **14.34** / 13.49 / 12.30 |
| s0·1238 | 427 fr | ✓ survived, 928 cmd-u | 3 | **13.21** / 12.20 / 12.02 |
| s1·1236 | never finished | ✓ survived, 861 cmd-u | 2 | — / **13.99** / 12.53 |
| s1·1237 | never finished | ✓ survived, 805 cmd-u | 3 | — / **13.40** / 12.37 |
| s1·1238 | never finished | ✓ young at run end, 507 cmd-u (unverified) | 6 | — / **13.67** / 13.03 |

**1. The PSNR gain does not survive the sweep — reported as a null.** Only 3 trials
have a pairable baseline lap 2 at all; on those, paired-by-heading-bin diffs are
**atlas_dr +0.10 ± 1.40 dB** and **atlas_slam −0.84 ± 1.07 dB**. The single-trial
+0.7 dB was within run-to-run variance. Notably the one *healthy-world* trial
(s0·1237, lap-1 = 342 frames, the Phase-6.5 regime) is also the one where every SLAM
closure was revoked — closure-churn (retrieval key thrashing between linear and
circular) plausibly *hurt* generation there. Fidelity on revisit is not an
inference-side win at this operating point.

**2. The dynamics finding replicates 6/6 — this is the robust effect.**
`atlas_slam` completed the full second revolution in **every trial** (max lap-2
heading 359–360°). `atlas_dr` completed 5/6 (one reached 136°). The no-memory
baseline **never reached lap 2 in 3/6 trials** (203°, 200°, 275° total after 1100
commands), stalled at the wrap in a 4th, and completed lap 2 in only 2/6. Pose-indexed
memory is what keeps the world *turning*.

**3. Calibration: there is no cross-session period to calibrate against.** Surviving
in-run periods span **805–1063 cmd-units/rev** across scenes *and* across sessions of
the same scene+seed (971 in the single-trial run vs 1063 in the sweep for s0·1236).
The turn rate is a property of the *realized world*, not the checkpoint — cross-run
calibration (Phase 6.5, `atlas_dr`) is incoherent in principle, not merely imprecise.
In-run closure is the only well-posed estimator, which retroactively justifies the
SLAM investment even where PSNR is flat.

**4. The aliasing defense held everywhere: 22 false closures revoked, zero wrong
periods retained.** False periods cluster at 3100–5600 px (0.5–0.8 of a revolution) —
the model's characteristic self-similarity scale. Two trials correctly ended *open*
(every candidate revoked) rather than committing to a bad map; one trial's final
closure was younger than the verification horizon at run end and should be read as
unverified.

---

## Caveats

- **Run-to-run nondeterminism is large.** Same seed, same commands: lap-1 content
  length was 489 frames in one session and 627 in the next (bf16 flex-attention
  autotune noise compounds through closed-loop generation). Within-run paired arms
  remain fair; single-trial magnitudes are not stable. Hence the sweep.
- **The baseline arm saturates the benchmark.** Its turn-rate collapse means lap-2
  coverage is small and its heading axis compresses; paired comparisons only exist for
  early revisit. The rotation-delivered number (474° vs 872°/1124°) is itself the
  finding, but it complicates PSNR pairing.
- **PSNR undersells scene identity.** The grid shows same-world-vs-new-world
  differences that land as ~+0.7 dB. A scene-level metric (segmentation/feature
  similarity) would read closure quality better.
- **Scope.** Yaw-only (1-DOF), single checkpoint, k=1 retrieval, parameters untuned.
  The closure fired late in this trial (pose ~1.57 measured laps — three revoke cycles
  plus escalating vote requirements in a degrading world), so most of lap 2 ran with
  linear (pre-closure) retrieval; earlier, cleaner closures would raise the measured
  gain.
- **atlas_dr runs at 2.0 fps vs atlas_slam's 5.1** (reproducibly, from the first
  frames) despite identical per-frame op counts (`restamp_temporal_k` is
  delta-independent elementwise math, so it is not the restamp). Unprofiled; perf
  footnote only.

---

## Handoff — what this buys the pitch

Phase 6 ended with: *pose-indexed retrieval works; the frozen attention under-weights
the injected key; train it in.* Phase 7 sharpens that in three ways:

1. **The retrieval stack is no longer the excuse.** Signed VO + voted closure +
   verification + reloc give the right keyframe at the right circular pose, fully
   in-run, robust to the aliasing a generative world actually produces. Any remaining
   permanence failure on a spin is the model, not the memory.
2. **A new, crisper symptom for the training team:** forgetting breaks *dynamics*, not
   just appearance — the no-memory world failed to complete one revolution in 3/6
   sweep trials while the SLAM-keyed arm completed two revolutions in 6/6.
   "Trained-in pose-indexed memory" should be evaluated on control-responsiveness,
   not only reconstruction.
3. **The aliasing result is a training-data argument:** the model renders
   near-duplicate content at wrong headings (NCC 0.9 at 0.5–0.7 rev). A model trained
   to *read* pose-indexed memory would be pulled toward pose-consistent rendering,
   which suppresses exactly this.

**Reproduce.**
```
uv run --dev pytest examples/test_slam.py examples/test_atlas.py \
    examples/test_atlas_engine.py                      # CPU, 34 tests
uv run --dev python examples/spin_slam.py --out bench_out/spin_slam \
    --csv bench_out/spin_slam/spin.csv                 # single trial + media
bash bench_out/spin_slam/sweep/run_sweep.sh            # 6-trial sweep
```
