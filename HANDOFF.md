# Handoff — spin-persistence (2026-07-06 ~23:45)

## State
- **Branch `spin-persistence`.** Phase 8 committed at `2b97dbc` (graybox + rigid: write-back
  cascade diagnosed and killed; **tuned atlas + output overlay = +3.40 dB, 6/6 — inference-side
  record, 2× Phase 6**). Phase 9 runner/tests are committed at `95ab61d` on top.
  Not pushed.
- CPU tests green through Phase 8: `uv run --dev pytest examples/test_rigid.py examples/test_graybox.py`
  (14 + graybox), plus the Phase ≤7 suites.
- Shareable results page (Phase 8 drill-down, charts from raw CSVs):
  https://claude.ai/code/artifact/487bf14a-5d05-4cc8-a1ac-8a9ebe6adddf

## Your job: Phase 9 — warmstart / re-dream (laundered memory write-back)
**Read `docs/WARMSTART_PLAN.md` first. It is the spec.** Short version:

- Phase 8 proved raw latent edits in the KV context are fatal (OOD + self-registration
  runaway) — the overlay repairs the picture but nothing re-anchors the rollout.
- Phase 9 idea: composite the registered memory band into the sampler's PROPOSAL,
  renoise to a grid sigma (0.3 → 1 Euler step, 0.75 → 2; grid is `[1.0,.9,.75,.3,0]`),
  re-run the chain tail under the same ctx/KV, and write THAT back — the edit
  re-expressed as a model sample. Durability channel, made legal.
- `examples/warmstart.py` EXISTS (resume_index / renoise / partial_denoise /
  redream_step, same contract as rigid_step). Import-checked only — **never run on GPU**.
- `examples/test_warmstart.py` and `examples/warm_probe.py` now exist. CPU checks passed:
  `uv run --dev pytest examples/test_warmstart.py -v` and
  `uv run --dev pytest examples/test_rigid.py examples/test_graybox.py -q`.
- GPU pilot is still NOT run. Last checked GPU had an active `VLLM::EngineCore`
  process using ~58 GB, so the safe resume point is the pilot:
  `uv run --dev python examples/warm_probe.py --arms revisit,redream,redream_nowb --n-scenes 1 --seeds 1 --csv bench_out/warm/pilot.csv`.
  The runner prints condition/source hashes and writes dx/provenance rows by default.
  Then the 7-arm tuned sweep (`--M 1 --restamp-offset 4`), 2 scenes × 3 seeds,
  CSVs → `bench_out/warm/`.
- **The fingerprint is the per-frame dx curve**: bounded ⇒ laundering restored
  durability (then go for the triple-stack record); monotone growth like
  bench_out/rigid/diag.csv ⇒ even on-manifold edits stall dynamics ⇒ write it up as
  the strongest training-side evidence yet. Either outcome is a publishable result.

## What Phase 8 established (details: docs/RIGID_RESULTS.md)
1. Write-back is the whole disease: rigid −1.34 (0/6) vs rigid_nowb +0.45 (6/6); |dx|
   runs away 10→250 px only with write-back on; gate strictness irrelevant; harm ∝ λ.
2. Alignment can't save it: anchor arms negative — hard in-context edits are
   intrinsically OOD for the frozen model. Terminal context editing is dead, not mistuned.
3. The two clean channels stack additively: attention (atlas +0.93) + output overlay
   (+0.46) = +1.42; at tuned density +2.62 / **+3.40**, overlay contribution GROWS
   with keyframe density (+0.48 → +0.78).
4. Honesty: ~half the raw paste gain is cosmetic (pose control +0.25, 5/6); paired
   within-trial stats only.

## After Phase 9 (in order, from the SPIN handoff — still valid)
1. warm_fast real-time path (1–2 steps from previous-frame registration — FASTER than
   baseline; play_server payoff). Only after the decisive probe.
2. Phase-7.5 hybrid keying: content-true pose + cmd-driven query advance when VO flatlines.
3. Closure-churn cost investigation (s0_1237 trial).

## Gotchas
- Run-to-run nondeterminism is large (same seed ⇒ different worlds); only within-trial
  paired comparisons are valid.
- Never call `engine.prep_inputs` twice per frame (double-advances frame_ts + camera_yaw);
  reuse `inputs` for propose and re-dream chains (redream_step already does).
- VAE decode is a temporal stream: snapshot state before the provisional decode, rewind
  + re-decode on accept (redream_step already does; pattern from rigid_step).
- `partial_denoise` must run with `kv_cache.set_frozen(True)` (it does); `_cache_pass`
  unfreezes itself.
- Off-grid renoise sigmas are rejected on purpose — the model never saw them.
- `bench_out/` is gitignored; keep new CSVs/logs in `bench_out/warm/`.
- The LAN demo server from the Phase 7 handoff is no longer expected to be running.
