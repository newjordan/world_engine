# Handoff — spin-persistence (2026-07-06 ~17:10)

## State
- **Branch `spin-persistence`, commit `a57a23a`** (Phase 7: SLAM loop closure). Not pushed.
- 34 CPU tests green: `uv run --dev pytest examples/test_slam.py examples/test_atlas.py examples/test_atlas_engine.py`

## Running right now
- **LAN demo server**: http://192.168.1.176:8850/web/ (bg task `b6n8qywu7`, serves `bench_out/spin_slam/`). Kill when done.
- Sweep DONE (6/6 trials, ~18:35). Results compiled into docs/SPIN_RESULTS.md.

## What Phase 7 established (details: docs/SPIN_RESULTS.md — sweep-final)
1. **Dynamics finding, replicated 6/6 (the headline):** no-memory baseline failed to finish ONE revolution in 3/6 trials (200–275°/1100 cmds); SLAM-keyed atlas completed TWO revolutions in 6/6. Forgetting breaks control-responsiveness, not just appearance.
2. **PSNR fidelity gain is a NULL at sweep level** (dr +0.10±1.40, slam −0.84±1.07 dB, 3 pairable trials) — single-trial gains are run-to-run noise. Reported honestly in the doc. Qualitative scene-identity return is real (grid png).
3. **Cross-run calibration is incoherent in principle**: surviving in-run periods 805–1063 cmd-u/rev vary across sessions of the SAME scene+seed. In-run closure is the only well-posed estimator.
4. **Aliasing defense held**: 22 false closures revoked across sweep, 0 wrong periods retained, 2 trials correctly ended open (NCC ≤0.91 lookalikes at 0.5–0.8 rev).

## Next steps (in order)
1. **atlas_warmstart arm** (user-endorsed direction, not started): store latent x0 in atlas alongside KV (`gen_frame(return_img=False)` path), on revisit renoise retrieved x0 to σ∈{0.3, 0.75} (sigmas are `[1.0,.9,.75,.3,0]`, `world_engine._denoise_pass`) and run remaining steps only. Injects memory via latent-init channel → may dodge the attention wall (the PSNR null above is the motivation) AND is 2-4× faster. Gate on retrieval residual.
2. Phase-7.5 hybrid keying: content-true pose + cmd-driven *query* advance when VO flatlines (dr arm shows dragging works).
3. Closure-churn cost: the one healthy-world trial (s0_1237) had all closures revoked and memory arms UNDERPERFORMED baseline — investigate whether revoke/re-close thrash hurts generation; maybe freeze retrieval mode during provisional closures.
4. Unprofiled: atlas_dr arm runs 2.0 fps vs slam 5.1, op counts identical (not the restamp — it's delta-independent).

## Gotchas
- Run-to-run nondeterminism is large (same seed ⇒ different worlds); only within-run paired comparisons are valid.
- Reloc evidence must come from a PREVIOUS lap (same-lap = self-referential lock-in).
- SLAM converges to map-consistency, not metric truth — correct for retrieval; don't "fix" it.
- Turn-rate stalls are REAL model behavior (phase-corr reads static with high confidence) — don't add cmd-fallback odometry.
- `bench_out/` is gitignored; regenerate media via `examples/spin_slam.py` (single trial ~15 min on GB10).
