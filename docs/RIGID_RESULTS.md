# Phase 8b — Rigid Latent Projection: from decisive negative to diagnosed mechanism

**Branch:** `spin-persistence` · **Date:** 2026-07-06 · **Code:** `examples/rigid.py`,
`examples/rigid_probe.py`, `examples/test_rigid.py` (14 CPU tests)
**Protocol:** return-to-start hysteresis, identical to `atlas_probe.py` (K=64, settle 8,
pan away 32, pan back 32, score each pan-back frame against the pan-away frame at the
same heading). 2 scenes × 3 seeds; paired per-trial stats, hard regime = heading ≤ 3.2.

## The idea

Phases 3–7 shaped the *context* that produces the sampler's velocity field and topped
out at ~+1.7 dB — the frozen attention under-weights memory (the attention wall).
Phase 8 constrains the sampler's *output* instead: after the final Euler step, the
SLAM-certified band of a registered memory keyframe replaces the same band of the
provisional latent x0,

    x0' = (1 − lam·M) ⊙ x0 + lam·M ⊙ warp(x0_mem)

and x0' (not x0) is written into the KV context, so the correction re-anchors the
autoregressive rollout. The graybox lab (`docs/GRAYBOX_RESULTS.md`) de-risked this in a
surrogate: appearance-registered shifts only (zero-net-translation rule), response
gating, class-split masks, and it predicted terminal-step projection ≡ per-step
guidance with a stable plateau through lam=1.

## Run 1 (probe.csv): decisive negative — but with a shape

| paired, hard regime (n=6) | Δ dB | wins |
|---|---|---|
| atlas − revisit | **+0.88 ± 0.61** | 6/6 |
| rigid − revisit | **−1.26 ± 0.96** | 0/6 |
| rigid − rigid_far (wrong-pose control) | −0.26 ± 0.41 | 1/6 |

Averaged over the pan-back, projection loses and is indistinguishable from the
wrong-pose control. But per-frame deltas showed the failure is not a constant:

- **Pan-back frames 1–9:** rigid **+0.7…+2.0 dB**, ≈ atlas, and cleanly separated from
  rigid_far (−0.3…−1.5) → pose-registered output projection *does* work on the real
  engine while memory is fresh.
- **Frames 10–14:** sharp collapse to −3.6 dB; rigid and rigid_far converge.
- **Frames 15–32:** partial recovery toward −0.9 dB.

## Run 2 (diag.csv): the channel-splitter — write-back is the whole disease

Instrumented sweep (per-frame accept/|dx|/response/reject logged) with two new arms:
`rigid_nowb` (project and score, but cache the **unprojected** latent — isolates the
KV write-back channel) and `rigid_soft` (lam 0.35), plus a 4×-stricter gate arm.

| paired, hard regime (n=6) | Δ dB | wins |
|---|---|---|
| rigid − revisit (replication) | −1.34 ± 0.85 | 0/6 |
| **rigid_nowb − revisit** | **+0.45 ± 0.13** | **6/6** |
| rigid_nowb − rigid | +1.80 ± 0.74 | 6/6 |
| rigid_soft − revisit | −0.66 ± 0.48 | 0/6 |
| rigid_gate − rigid | +0.09 ± 0.15 | 4/6 (tie) |

**Mechanism (read off the per-frame table):** with write-back on, the applied
registration shift |dx| grows monotonically ~10 px → ~250 px across one pan-back;
without write-back it stays ≤ 20 px through frame ~22. Writing back a *rolled* paste
stalls the world's commanded rotation (the Phase-7 "forgetting breaks dynamics"
failure, here induced by the paste), the world falls further behind the map, the next
paste needs a bigger roll — more wrap-around contamination, more fractional-roll blur —
and the loop runs away. The response gate cannot catch it because registration is
*confident* throughout (resp 0.3–0.5): it is tracking content the paste itself put
there. Gate strictness is irrelevant (rigid_gate ≈ rigid); harm scales with lam
(soft = half the damage); removing write-back removes all of it.

This is the graybox's zero-net-translation rule violated through the back door: the
shift is appearance-computed (allowed), but once the paste enters the context, the
"appearance" being registered against is the paste's own echo — net translation is no
longer zero, it compounds.

## Run 3 (fix.csv): write-back is dead in ALL forms; the stack wins

| paired, hard regime (n=6) | Δ dB | wins |
|---|---|---|
| atlas − revisit (replication #3) | +0.93 ± 0.54 | 6/6 |
| rigid_nowb − revisit (replication) | +0.46 ± 0.13 | 6/6 |
| **atlas_nowb − revisit (stacked)** | **+1.42 ± 0.74** | **6/6** |
| **atlas_nowb − atlas** | **+0.48 ± 0.20** | **6/6** |
| rigid_anchor (wb only \|dx\| ≤ 12 px) − revisit | −0.45 ± 0.57 | 2/6 |
| anchor_pure (anchor, score raw decode) − revisit | −0.77 ± 0.49 | 0/6 |
| rigid_nowb − nowb_far (pose control) | +0.25 ± 0.30 | 5/6 (leans) |

1. **The anchor rule failed.** Writing back only well-aligned pastes (23% of frames,
   |dx| ≤ half a latent cell) still loses, and `anchor_pure` shows the pure context
   effect of an aligned write-back is *negative* (−0.77). A hard latent edit inside the
   KV context is intrinsically OOD for the frozen model — alignment gating cannot fix
   it. Terminal-endpoint context editing is dead, not mistuned. Durable memory
   injection must go through attention (atlas) or the latent-*init* channel
   (warmstart), never through post-hoc context edits.
2. **The two surviving channels stack additively**: attention-side atlas (+0.93) and
   output-side projection (+0.46) combine to +1.42, 6/6 — architecturally orthogonal
   in practice, not just in principle. Per-frame deltas confirm the runaway trough is
   gone (no write-back), positive nearly everywhere.
3. **Honesty note:** the paste channel's pose-specificity is +0.25 (5/6) over a
   wrong-pose paste, which itself gets +0.21 — roughly half the raw paste gain is
   cosmetic (any plausible texture beats forgotten mush), half is registered memory.

## Run 4 (stack_tuned.csv): the stack at tuned atlas density — new inference-side record

Same protocol at the Phase-6 tuned settings (capture every frame M=1, restamp offset 4):

| paired, hard regime (n=6) | Δ dB | wins |
|---|---|---|
| atlas − revisit | +2.62 ± 0.74 | 6/6 |
| **atlas_nowb − revisit** | **+3.40 ± 0.91** | **6/6** |
| atlas_nowb − atlas | +0.78 ± 0.21 | 6/6 |

The projection layer's contribution *grows* at tuned density (+0.48 → +0.78; denser
keyframes = fresher registration references). **+3.40 dB doubles the previous
inference-side best (+1.67, Phase 6 tuned atlas)** — all frozen weights, flag-gated,
zero training.

## Verdict

- Output-side projection works **only as an overlay** (never write the paste into KV).
- The write-back runaway is a real dynamics failure mode of memory injection —
  document it in any training-side pitch: the frozen model treats in-context latent
  edits as adversarial, which is itself evidence that persistence must be trained in.
- Best current inference-side recipe: **tuned atlas (attention) + registered band
  projection (output overlay) = +3.40 dB hard-regime**.
- Next lever: atlas_warmstart (renoise retrieved x0 to σ ∈ {0.3, 0.75}, resume
  sampling) — the only remaining path that puts memory into the rollout's own
  dynamics through in-distribution sampling rather than OOD edits.

## Honest caveats

- `rigid_nowb`'s +0.45 dB includes pasting near-reference content into ~25% of scored
  pixels; `anchor_pure` exists to separate real context repair from that cosmetic
  component — and found the context component *negative*; the stack's +0.78 over
  atlas is therefore part overlay-cosmetic, part registered memory (the +0.25
  pose-specific component). The qualitative claim "the certified band shows the
  remembered scene" survives either way; PSNR attribution does not fully.
- Run-to-run nondeterminism is large; only within-trial paired comparisons are valid.
- All findings at yaw_mag 0.2, K=64, band mask only.

## Honest caveats

- `rigid_nowb`'s +0.45 dB includes pasting near-reference content into ~25% of scored
  pixels; `anchor_pure` exists to separate real context repair from that cosmetic
  component.
- Run-to-run nondeterminism is large; only within-trial paired comparisons are valid.
- All findings at yaw_mag 0.2, K=64, M=2 capture density, band mask only.
