# Graybox results — surface-vector guidance in a scaled-down B&W world (Phase 8 pilot)

**Code**: `examples/graybox.py` (lab), `examples/test_graybox.py` (8 tests, all
passing), `examples/graybox_tune.py` (sweeps below).

## Why this lab exists

The engine's sampler (`WorldEngine._denoise_pass`) is an Euler integrator over a
velocity field v; nothing external ever enters the integration ("the closed
vector"). Phases 3–7 shaped the context that *produces* v. Phase 8 asks what happens
when a geometry-certified memory term is added to v itself:

    v' = v + lam * M (.) (x0_hat - align(x_mem)) / sigma

Before touching the engine, the mechanism is developed where every variable is
controllable and ground truth exists: a grayscale panorama world (the `test_slam.py`
harness), a surrogate flow-matching engine with the two real failure modes built in
(per-lap **forgetting** via belief write-back, **regime drift** via a gain/bias tone
walk — the 1-D stand-in for palette/VAE drift), one **dynamic element** (a blinking
beacon whose phase the engine tracks from its own outputs), and the real barnacle:
unmodified `YawSLAM` + `AtlasStore` holding lap-1 frames keyed by SLAM pose.
Guidance switches on at loop closure. CPU-only, ~9 s per 4-lap run.

## Finding 1 — a guidance vector with a net translation stalls the world (negative result, load-bearing)

First implementation registered the memory by pose alone: roll the retrieved
keyframe by `slam.pose − kf.pose`. Keyframe poses carry the map's accumulated
odometry bias (~11 px after lap 1), so the guidance pulled rendered content
slightly *backward*; the odometry — which measures rendered content — then lagged
further, retrieval regressed, and the pull grew: −11 px → −39 px in 15 frames,
`mem_ncc` → 0, run destroyed, fidelity *worse* than no guidance.

**Rule**: the guidance vector must carry zero net translation. Pose retrieval only
*selects* the keyframe; the applied shift comes from phase-correlating the keyframe
against the current `x0_hat` (appearance registration), with a response gate — no
certified registration, no claim. Position errors belong to the SLAM/reloc channel;
g is only allowed to correct texture and tone. After this change `mem_ncc` = 0.90
and the `refine` diagnostic shows the ~7 px map bias being absorbed harmlessly.
The real-engine analog: restamp/warp memories into register before they touch v,
or the memory force leaks into the world's dynamics.

## Finding 2 — registered guidance repairs forgetting durably

Baseline static RMSE per lap: **7.5 / 16.0 / 20.0 / 22.6** (the forgetting curve).
Guided (lam=0.35, static mask, gainbias align): **7.5 / 9.4 / 12.3 / 12.7**.
Late-lap error cut ~45%, and the repair is durable because the corrected frame is
written back into the engine's belief — one projected frame re-anchors the
autoregressive loop; guidance does not have to fight the same error every lap.

## Finding 3 — the Phase 7 dynamics break reproduces, and the class split fixes it

| mask | late-lap rmse | beacon corr (laps 2–3) |
|---|---|---|
| `all` (undifferentiated surface) | 12.5 | **−0.89 … −0.29** (broken) |
| `static` (beacon cells released) | 12.5 | **1.00** (intact) |

An undifferentiated mask pins the beacon toward stale lap-1 intensity; the engine's
tracker locks onto its own frozen outputs (emergent, not scripted). Splitting the
surface into two classes restores dynamics at zero fidelity cost. This is
"different surfaces relay different profile requirements" in the smallest world
that can express it.

## Finding 4 — tone alignment is a profile choice, not a bugfix

| align | late-lap rmse | tone err | seam |
|---|---|---|---|
| `gainbias` (geometry-only) | 12.5 | 0.552 | 12.1 |
| `none` (raw memory) | 9.4 | **0.110** | 12.9 |

Raw memory pulls masked cells back to lap-1 tone (5× less regime drift in the
output) but pays in boundary seams; aligned memory is geometry-only and lets tone
drift pass through. Neither dominates — they are different profiles. The natural
composition (align locally, correct regime globally with the fitted gain/bias) is
the graybox analog of the per-layer orthogonal-warp idea and is untested here.

## Sweeps (late-lap means, `graybox_tune.py`)

- **lam (E1)**: benefit saturates by lam≈0.1, flat plateau through lam=1.0
  (rmse ~12.5), catastrophic at lam=2 (overshoot past the memory — in this
  surrogate lam is exactly the final blend weight). Wide safe range.
- **schedule (E2)**: degenerate in the surrogate — the analytic target makes the
  terminal Euler step history-erasing, so all schedules coincide. Only answerable
  in the real engine, where the network's implicit target moves with x.
- **retrieval pose (E4)**: oracle pose ≈ SLAM pose (12.42 vs 12.49) once appearance
  registration does the fine alignment. Retrieval only needs to fetch the right
  keyframe; the pose estimate does not need to be metrically true.

## What this buys the engine work

1. The v-space injection point and its safety rules (zero net translation,
   evidence-gated claims, class-split masks) are now de-risked and unit-tested.
2. The engine experiment (store lap-1 latents in the atlas; blend at the final
   denoise step over a SLAM-certified mask) has a predicted outcome per failure
   mode and a diagnostic (`refine`, `mem_ncc` analogs) for each.
3. Open questions that only the engine can answer: guidance schedule (E2), whether
   latent-space registration needs the orthogonal/Householder warp beyond
   shift+gain/bias (the graybox's 1-D regime axis collapses that question), and
   the per-cell staticity classifier at real texture statistics.
