# Phase 9 plan — Warmstart / Re-dream: laundered memory write-back

**Status: runner ready, GPU untested.** `examples/warmstart.py` is written and
import-checked. `examples/test_warmstart.py` and `examples/warm_probe.py` are committed
at `95ab61d`; CPU tests pass, but **no GPU run has happened**. This doc is the handoff
spec for the agent that runs it.

**Branch:** `spin-persistence` · **Prereqs:** docs/RIGID_RESULTS.md (Phase 8),
docs/SPIN_RESULTS.md §next-steps (warmstart endorsement), examples/rigid.py.

## The question

Phase 8 ended on a dilemma. Output-side memory works — tuned atlas + registered band
overlay = **+3.40 dB, 6/6** — but the durability half is forbidden: any hard latent
edit written into the KV context is OOD for the frozen model (`anchor_pure` −0.77 even
perfectly aligned), and a rolled paste self-registers into a runaway (|dx| 10→250 px in
one pan-back). The overlay repairs the *picture*; nothing re-anchors the *rollout*.

Phase 9 tests one idea: **re-express the edit as a sample.** Composite the registered
memory band into the sampler's proposal, renoise to a mid-schedule sigma, and re-run
the tail of the Euler chain under the same KV context — then write *that* back:

    x0_prop = full denoise from pure noise            (the model's unaided belief)
    dx,resp = phaseCorrelate(kf.roi, decode(x0_prop))             [register]
    x0_comp = (1 − λ·M) ⊙ x0_prop + λ·M ⊙ warp(x0_mem)            [compose]
    x_s     = (1 − s) · x0_comp + s · ε ,   s ∈ scheduler grid    [renoise]
    x0_re   = Euler chain resumed from s, same ctx + KV           [re-dream]
    cache   ← x0_re                                               [commit]

x0_re is a model output: on-manifold by construction, seam-free (the sampler smooths
the composite boundary), and context-consistent (the re-dream attended to the same KV
the next frame will). Writing it back is legal in exactly the sense Phase 8 proved raw
pastes are not. **The decisive question: does laundered write-back restore durability
(re-anchoring, bounded |dx|), or does ANY memory-bearing context edit — even an
on-manifold one — stall the world's dynamics?** Either answer is a result: the first
is a new record path; the second is the strongest training-side evidence yet that
persistence cannot be bolted on at inference.

## Engine facts the implementation relies on (verified this session)

- Config scheduler grid is `[1.0, 0.9, 0.75, 0.3, 0.0]`; the live engine stores it
  in model dtype, so on bf16 it resolves to `[1.0, 0.8984375, 0.75, 0.30078125, 0.0]`.
  The doc-endorsed renoise levels σ∈{0.3, 0.75} are still **grid points** after dtype
  snapping: s≈0.3 → 1 Euler step, s=0.75 → 2 steps. Off-grid sigmas are rejected
  (`resume_index` raises) — the model never saw them.
- Flow convention (from `world_engine._denoise_pass`): x starts at pure noise σ=1,
  Euler `x += dσ·v` down the grid ⇒ forward map is `x_s = (1−s)·x0 + s·ε`.
- `_denoise_pass` is fullgraph-compiled and always starts at σ=1; the re-dream tail
  (`warmstart.partial_denoise`) is an **eager** mirror that must call
  `kv_cache.set_frozen(True)` before its model calls (it does).
- `engine.prep_inputs` advances `frame_ts` and dead-reckons `camera_yaw` — call it
  **once per frame** and reuse `inputs` for both the propose chain and the re-dream
  chain (redream_step does this).
- The VAE decoder is a temporal stream: snapshot `engine.vae.get_state()` before the
  provisional decode, `load_state` + re-decode iff the re-dream is accepted — exactly
  one effective decode per frame (same discipline as `rigid_step`).

## What already exists

`examples/warmstart.py` (mirrors rigid.py's layout; registration/store/mask reused
from rigid.py so test_rigid.py already covers them):

- `resume_index(sigmas, s)` / `renoise(x0, eps, s)` — pure, CPU-tested inline only.
- `partial_denoise(engine, x, ctx, j)` — eager Euler tail from grid index j.
- `redream_step(...)` — same contract/info-dict as `rigid_step`, plus `sigma_re`;
  `write_back=True` caches x0_re (the claim), `write_back=False` caches x0_prop while
  scoring the re-dreamed decode (isolates the init/overlay component, exactly as
  `rigid_nowb` isolated the paste).

`examples/test_warmstart.py` implements the CPU test plan below. `examples/warm_probe.py`
implements the tuned arm table, prints condition/source hashes at startup, and writes
per-frame `dx_px` plus `sigma_re` rows to CSV for the decisive fingerprint.

## Experiment design (`examples/warm_probe.py`)

Same return-to-start hysteresis, **tuned settings**: K=64, settle 8, yaw_mag 0.2,
`--M 1 --restamp-offset 4`, 2 scenes × 3 seeds, paired per-trial stats, hard regime
heading ≤ 3.2, CSV schema of fix.csv (+ optionally sigma_re).

| arm | mechanism | purpose |
|---|---|---|
| revisit | none | baseline |
| atlas | attention (tuned) | incumbent replication |
| atlas_nowb | atlas + output overlay (rigid_step, write_back=False) | Phase 8 record replication anchor |
| **redream** | redream_step σ=0.3, write_back=True | **the durability claim** |
| redream_nowb | redream_step σ=0.3, write_back=False | channel isolation (expect ≈ rigid_nowb's +0.45) |
| redream_far | retrieve="farthest", write_back=True | pose-specificity control |
| atlas_redream | atlas + redream (wb) | does init stack with attention like output did? |
| redream75 (optional) | σ=0.75 | dose sensitivity (deeper re-dream = context gets more votes) |

Decision rules:
1. **Headline:** `redream − redream_nowb`. Positive with per-frame |dx| bounded
   (compare against the Phase 8 runaway curve, bench_out/rigid/diag.csv) ⇒ laundering
   restored the durability channel. Negative with growing |dx| ⇒ on-manifold edits
   stall dynamics too ⇒ write it up as the training-side result.
2. `redream_nowb − revisit` ≈ +0.4…0.5 expected (sanity: same overlay content,
   softened by renoise). Much lower ⇒ the re-dream is washing out memory (try σ=0.3
   only, check λ).
3. `redream − redream_far` > 0 required for any pose-registered-memory claim.
4. Stack: `atlas_redream − atlas` vs Phase 8's `atlas_nowb − atlas` (+0.78 tuned).
5. If redream wins, run the triple (atlas + redream wb + output overlay on the scored
   frame) for a record attempt.

Diagnostics to log per frame (info dict already carries them): dx_px, resp, accept,
wrote_back. **The dx-vs-frame curve is the fingerprint** — bounded ⇒ negative
feedback (re-anchoring), monotone growth ⇒ the Phase 8 disease in a new coat.

Runtime: redream arms ≈ 1.6–2× a plain frame (4-step propose + 1–2 step re-dream +
second decode). Pilot first: `--arms revisit,redream,redream_nowb --n-scenes 1
--seeds 1` and eyeball accept rate (expect ~100% like atlas_nowb) and timing before
the 7-arm sweep.

## Runbook

Condition source: this file. Run label: `new_experiment` until a GPU pilot or sweep
has actually completed and written `bench_out/warm/*.csv`.

Do not launch if the GPU is already carrying another large model. The last check before
the runner commit found an active `VLLM::EngineCore` process using about 58 GB, so the
pilot was intentionally not run.

1. Pilot, minimal claim surface:

       uv run --dev python examples/warm_probe.py \
         --arms revisit,redream,redream_nowb \
         --n-scenes 1 --seeds 1 \
         --csv bench_out/warm/pilot.csv

2. If the pilot starts cleanly, writes CSV rows, and has sane accept/timing, run the
   tuned Phase 9 sweep:

       uv run --dev python examples/warm_probe.py \
         --arms revisit,atlas,atlas_nowb,redream,redream_nowb,redream_far,atlas_redream \
         --M 1 --restamp-offset 4 \
         --n-scenes 2 --seeds 3 \
         --csv bench_out/warm/warm.csv

3. Only add `redream75` after the σ=0.3 result is interpretable:

       uv run --dev python examples/warm_probe.py \
         --arms revisit,redream,redream_nowb,redream75 \
         --n-scenes 2 --seeds 3 \
         --csv bench_out/warm/redream75.csv

The runner prints the command, source hashes, scheduler grid, model, seeds, arms,
and metric at startup. Save stdout beside the CSV with `tee` when doing a decisive
run; the CSV alone is enough for metrics, but the log carries the resolved condition.

## CSV and analysis contract

`warm_probe.py` writes one scored pan-back row per `(scene, seed, arm, back_idx)`:

    scene,seed,arm,back_idx,heading,psnr,ssim,projected,dx_px,resp,reject,wb,sigma_re

- `back_idx`: pan-back frame order, starting at 0. Use this for the dx fingerprint.
- `heading`: matched outbound heading used for the PSNR/SSIM reference.
- `projected`: redream/overlay accepted registration for the scored frame.
- `dx_px`: appearance registration shift in full-resolution pixels.
- `resp`: phase-correlation response. A high response with growing `dx_px` is exactly
  the Phase 8 runaway signature; do not dismiss it as a gate miss.
- `reject`: empty, `resp`, or `shift`.
- `wb`: whether the arm wrote the redreamed/projected latent back to KV.
- `sigma_re`: actual grid sigma used by redream, empty for non-redream arms.

Primary plots/tables to make after a run:

- per-trial hard-regime paired deltas:
  `redream - redream_nowb`, `redream_nowb - revisit`, `redream - redream_far`,
  `atlas_redream - atlas`, and `atlas_nowb - atlas`.
- mean `abs(dx_px)` vs `back_idx` for `redream`, `redream_nowb`, and if present
  Phase 8 `bench_out/rigid/diag.csv`.
- accept/wrote-back/reject rates by arm.

Result language is constrained by provenance:

- If only the pilot ran, call it a pilot and do not claim a Phase 9 result.
- If `redream - redream_nowb` is positive and `dx_px` remains bounded, the result is
  "laundered write-back restored the durability channel" and the next run is the
  triple-stack record attempt.
- If `redream - redream_nowb` is negative and `dx_px` grows like Phase 8, the result is
  "on-manifold memory-bearing context still stalls dynamics" and the next doc should
  frame this as training-side evidence, not an inference-side tuning failure.
- If `redream_nowb - revisit` is weak, do not interpret write-back yet; first inspect
  whether σ=0.3 washed out the memory band or whether registration was rejected.

## Test plan (test_warmstart.py, CPU, no model)

- `resume_index`: grid hits (0.3→3, 0.75→2, 0.9→1), off-grid raises, σ=1.0 and 0.0
  raise, tolerance snap (0.2999999).
- `renoise`: endpoints (s=0 ⇒ x0, s=1 ⇒ ε), linearity, float math preserved.
- `partial_denoise` equivalence on a stub engine (model = deterministic callable):
  resuming from j=0 must equal a full `_denoise_pass`-style loop; j=len−2 runs one step.
- Composite/mask/roll geometry: already covered by test_rigid.py — do not duplicate.

## Stretch directions (in order of value)

1. **Adaptive dosing:** σ per frame from measured disagreement (resp, |dx|, PSNR
   proxy) — memory injection proportional to measured forgetting. Novel control-loop
   framing; trivial to add to redream_step (map signals → {0.3, 0.75, skip}).
2. **warm_fast (the real-time play):** skip the propose pass — register against the
   *previous* frame's decode plus dead-reckoned one-frame delta, renoise the composite
   of the *previous* latent + memory band, run 1–2 steps only. **Faster than the
   4-step baseline** (the SPIN handoff's 2–4× speed note) → play_server payoff.
   Riskier science (no clean provisional to register/score against) — do after the
   decisive run.
3. Triple stack record attempt (see decision rule 5).

## Gotchas carried forward

- Run-to-run nondeterminism is large: only within-trial paired comparisons are valid.
- Never call `prep_inputs` twice in one frame (double-advances the clock and yaw).
- `capture` stores the **committed** latent (x0_final) — on pan-away frames
  project=False so it's the plain proposal, as in Phase 8.
- RigidStore's linear yaw distance is fine — this protocol never wraps a revolution.
- `bench_out/` is gitignored; keep CSVs + logs there (`bench_out/warm/`).
