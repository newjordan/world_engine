# NORTHSTAR — From Benchmark to World

**Status:** Final handoff goal sheet, 2026-07-07.
**Audience:** an autonomous agent (or chain of agents) running long loops on this
problem for days at a time, plus the human who reviews at gates.
**Prerequisite reading, in order:** `docs/HANDOFF_REPORT_2026-07-07.md`,
`docs/WARMSTART_RESULTS.md`, `docs/RIGID_RESULTS.md`. This document does not repeat
their content; it builds on it.

---

## 0. Mission

Turn the spin-persistence research line into two products:

1. **The science:** a public, reproducible benchmark + report establishing what
   persistence can and cannot be bolted onto a frozen autoregressive video world
   model — and the first inference-side recipe that passes it.
2. **The game:** a playable vertical slice where the persistence machinery is not
   plumbing but the *mechanic* — a world that only stays real where you've mapped it.

The "holy fuck" moment, concretely: **a browser window where you spin 360° in a
dreamed world, and everything is still there — then you walk away, come back, and
it's STILL there — live, ≥20 fps, shareable as a weblink and a 30-second MP4.**
Everything in this document is sequenced toward that moment and the paper-grade
evidence behind it.

---

## 1. Ground truth (do not re-derive, do not re-litigate)

Nine phases of controlled experiments established, at n=6 paired trials,
hard regime (heading ≤ 3.2), K=64, Waypoint-1.5-1B:

| Channel | Verdict | Evidence |
|---|---|---|
| Raw latent write-back into KV (rigid projection) | **DEAD** | `rigid − revisit = −1.34 dB`, 0/6; dx runaway 10→250 px (Phase 8) |
| Sampler-laundered latent write-back into KV (re-dream) | **DEAD** | `redream − redream_nowb = −4.28 dB`, 0/6; dx 130 px vs 17 px (Phase 9) |
| No-write-back output overlay (image channel) | **ALIVE** | `rigid_nowb +0.45 dB` 6/6; `redream_nowb +1.37 dB` 6/6 |
| Tuned pose atlas + output overlay (`atlas_nowb`, M=1, offset 4) | **RECORD** | `+3.42 ± 0.92 dB` over revisit, 6/6, replicated twice (+3.40 Phase 8, +3.42 Phase 9) |

**The one-sentence scientific insight:** *on this frozen model, any memory-bearing
latent written into the KV context — no matter how it is laundered — stalls the
model's own dynamics and produces the large-dx failure fingerprint; memory only
survives in channels the model never sees as context.*

**Standing stop rules (binding):**
- No more sigma/dose/scheduler variants of KV latent write-back. Two verified
  negatives in the same representation basin. A third is not evidence.
- Valid new work must **change the representation/search space** or **change the
  claim**. Every phase below satisfies this test.
- Paired within-trial comparisons only. Run-to-run nondeterminism is large; never
  compare independent reruns as if they are the same world.
- Carry-over gotchas remain binding: never call `engine.prep_inputs` twice per
  frame; snapshot/rewind the VAE decode stream around provisional decodes;
  `partial_denoise` requires `kv_cache.set_frozen(True)`; renoise sigmas must snap
  to the live bf16 scheduler grid; `bench_out/` is gitignored — preserve artifacts.

**The crucial distinction this goal sheet is built on:**
`atlas_nowb` is **cosmetic persistence**. The displayed frame is corrected, but the
model's internal world state has still forgotten — stop compositing and the world
reverts. That is fine for a product (the compositor can simply always run) but it
is not durability. **True durability** means the model's own rollout stays correct
after the intervention ends. No experiment so far has achieved true durability —
but one legal channel has never been tested (Phase 10).

---

## 2. The three tracks

### Track A — The Memory Compositor (product; inference-only; ships now)
Persistence lives entirely outside the model: pose-indexed atlas, registration,
reprojection, confidence-weighted overlay. The model is a real-time hallucinating
renderer; the compositor is the memory system. This track powers the demo and the
game regardless of how the science goes, because cosmetic persistence at 20+ fps
*is* the product experience.

### Track B — Trained persistence (science; requires training infra)
The negative results say frozen-model KV editing is dead — the honest fix is
training: LoRA fine-tunes that teach the model to tolerate or exploit
memory-bearing context, or post-training on loop-closure trajectories so the model
learns to close loops itself. This repo is inference-only; Track B needs a trainer
built or borrowed, and rented GPUs (vast.ai is available to this user). Gated
behind Tracks A/C evidence — do not start here.

### Track C — Hybrid geometric memory (bridges A and B)
The planar atlas dies under translation. Lift generated frames into geometry
(monodepth → pose-keyed point cloud or 3D Gaussians), render memory views from
geometry, and feed them through the *legal* channels (overlay, and — if Phase 10
succeeds — front-door re-anchoring). Geometry also gives the game collision,
minimaps, and object anchors for free.

---

## 3. Phase queue

Phases are numbered continuing from Phase 9. Each phase follows the established
convention: `docs/<NAME>_PLAN.md` (hypothesis, arms, exact runner command, gates)
→ run → `docs/<NAME>_RESULTS.md` (headline table, diagnostics, verdict, boundary,
stop rule) → commit. CPU tests must stay green
(`uv run --dev pytest examples/test_*.py -q`); new modules get CPU tests in the
`test_warmstart.py` style.

---

### **Phase 10 — Front-door re-anchoring (`append_frame`)** ⟵ START HERE

**Why first:** it is the single highest-value untested experiment. Phases 8–9
wrote latents into KV — the back door. `engine.append_frame(img)` is the model's
*designed* conditioning entry: pixel space in, through the VAE encoder,
on-manifold by construction. The Phase 9 boundary section explicitly does not
cover it. This is a genuinely different representation channel, so it passes the
stop-rule test.

**Hypothesis:** the write-back failure basin is specific to *off-manifold latent
edits entering KV*, not to memory-bearing content per se. If so, feeding the atlas
reconstruction through the encoder front door will restore the model's internal
state at revisit **without** the dx runaway — true durability, not cosmetic.

**Method:** extend `examples/warm_probe.py` (or a new `examples/anchor_probe.py`
sharing its protocol) with arms:

- `revisit` — baseline (unchanged)
- `atlas_nowb` — incumbent comparator (must replicate ≈ +3.4 dB or the run is suspect)
- `anchor` — at return-to-start detection, synthesize the atlas reconstruction for
  the current pose, call `append_frame` with it, continue the rollout normally
- `anchor_blend` — same, but append a confidence-weighted blend of atlas
  reconstruction and the model's own current frame (registration response as weight)
- `anchor_cadence_k` — anchor every k control steps during the return window, not
  just once (k ∈ {1, 4}; pick 2 values max, this is not a dose sweep)

Note Waypoint-1.5 appends 4-frame batches at 720p — the reconstruction must be a
`[4, 720, 1280, 3]` sequence; decide and document whether to repeat one
reconstruction ×4 or reproject 4 micro-poses.

**Primary metric — the sustained-durability curve:** PSNR vs ground-truth start
view for **K frames AFTER the last intervention**, not just at the return frame.
`atlas_nowb`'s curve collapses the moment the overlay stops (verify and plot this —
it is the money figure). If `anchor`'s curve stays high with zero further
intervention, that is the first true-durability result of the whole project.

**Secondary metrics:** the dx fingerprint (mean accepted |dx| must stay in the
~7–20 px healthy band, not the 60–130 px disease band); dynamics responsiveness
after anchoring (world must still respond to yaw commands — reuse the Phase 7
dynamics check).

**Gates:**
- **PASS:** `anchor` (any variant) sustained-durability beats `atlas_nowb`-with-
  overlay-stopped by ≥ +2 dB at 16 frames post-intervention, 5/6+, healthy dx.
  → True durability exists inference-side. Promote anchoring into every later
  phase; add translation anchoring to Phase 13; Track B drops to backburner.
- **PARTIAL:** anchoring helps at the return frame but decays within 16 frames.
  → Measure the decay constant; a cadence-anchored compositor is still a better
  product than overlay-only. Fold into Phase 12.
- **FAIL:** anchoring triggers the stall fingerprint (dx runaway, dynamics dead).
  → Major negative result #3: *no* inference-side channel gives durability, front
  door included. This upgrades the write-up (Phase 16) from "KV editing fails" to
  "persistence on frozen interactive video models requires training." Track B
  becomes the only durability path; product proceeds on pure compositing.
- **Kill criterion:** if `atlas_nowb` fails to replicate within ±1 dB of +3.4,
  stop and debug the harness before believing anything else in the run.

Budget: 1–2 days including the plan doc, CPU tests, pilot, full sweep
(2 scenes × 3 seeds minimum; scale to 4×4 if the effect is within noise).

---

### **Phase 11 — The Permanence Gauntlet (benchmark suite v1)**

**Why:** one scenario (yaw-only return-to-start) cannot carry the claim "cool
world model memory system." A battery can — and a committed, reproducible battery
is the scientific artifact other people can run on *their* models. This is what
turns the project from a series of private experiments into a field contribution.

Build `examples/gauntlet.py` + `docs/GAUNTLET.md` defining:

- **G1 Spin** — 360° return-to-start (the existing protocol, imported as-is).
- **G2 Glance** — look away 90°, hold 8–32 settle frames, look back. Short-horizon
  occlusion memory; easiest to pass, good floor.
- **G3 Out-and-back** — translate forward N *content-defined* units, turn, return.
  ⚠ Binding calibration warning from Phase 9: command counting is NOT degrees or
  meters. Prior calibration was ~805–1063 command units/revolution; legs are
  hundreds of frames. Define distance by registration displacement against the
  atlas, not by command count. Expect the planar atlas to fail here — that failure
  curve is Phase 13's motivation, so measure it, don't hide it.
- **G4 Patrol** — repeat G1 five times consecutively. Metric: PSNR-at-return vs
  loop index (drift/accumulation curve). Tests whether memory compounds or decays.
- **G5 Landmark re-ID** — place a distinctive generated object in view (pick
  scenes/seeds where one exists; document selection), leave, return; score
  landmark survival with a patch-level LPIPS/feature match, not full-frame PSNR.
- **G6 Marathon** — 5-minute free-roam session with scripted revisits every ~60 s.
  Memory management, VRAM, atlas growth, latency stability.

Metrics module: PSNR, LPIPS, SSIM at revisit; sustained-durability curve (from
Phase 10); dx fingerprint; dynamics responsiveness; fps/latency. Every gauntlet
run emits one CSV + one self-contained HTML report page (reuse
`examples/analyze_warm.py` machinery).

**Gate:** baseline table committed for `revisit`, `atlas_nowb`, and the best
Phase 10 arm across G1–G4 minimum. G5/G6 may land in a v1.1.
Budget: 2–3 days. G3's content-defined distance calibration is the hard part;
timebox it to one day and ship G1/G2/G4 first if it slips.

---

### **Phase 12 — The live demo (Milestone M1: the holy-fuck moment)**

Integrate the memory compositor into `examples/play_server.py`:

- Atlas capture, registration, and confidence-weighted overlay running live in the
  browser loop (WASD + mouse). INT8 + pacer previously hit ~29 fps; the compositor
  budget is **≤ 10 ms/frame** — profile with `examples/prof.py`, move registration
  to GPU or every-Nth-frame if needed.
- A **memory toggle key** (e.g. `M`): hold a spin with memory OFF (world smears
  and forgets), toggle ON, spin again (world holds). The before/after being *live
  and player-controlled* is what makes it land emotionally — this is the
  360p turnaround MP4, but you're the one steering.
- If Phase 10 PASSED: an **anchor key** that visibly "crystallizes" the current
  view into the world (this is also the core game verb — see §4).
- On-screen HUD: pose estimate, atlas keyframe count, registration confidence,
  fps. Debug honesty doubles as spectacle.

**Deliverables:** cloudflared weblink demo; a 30–60 s screen-capture MP4 of the
spin-away-return sequence; both linked from the gauntlet report page.
**Gate:** ≥ 20 fps sustained with compositor on; toggle demonstrably changes
persistence on camera. Budget: 2–3 days.

---

### **Phase 13 — Translation memory (the geometry ladder)**

The planar atlas assumes pure rotation. To walk around the world and have it
persist, climb this ladder, stopping at the first rung that passes G3:

1. **Homography+** — full homography (not just dx) registration against atlas
   keyframes; handles small translations cheaply. 1 day.
2. **Depth-warp** — run an off-the-shelf monodepth model on generated frames
   (frozen, external — this does not touch Waypoint's context, so it is a legal
   channel); store depth with each atlas keyframe; reproject memory views with
   parallax. 2–3 days.
3. **Gaussian world** — fuse pose-keyed RGBD into a persistent 3D Gaussian /
   point-cloud store; render memory views from geometry; overlay (and, if Phase 10
   passed, anchor) from the geometric render. This is Track C proper: the world
   state now lives in an explicit structure the game can also query for collision
   and minimaps. 4–7 days.

**Gate per rung:** G3 PSNR-at-return improvement over the rung below, 5/6+, at
≤ 2× the compositor latency budget. **Kill:** if depth on *generated* frames is
too inconsistent to fuse (flicker between 4-frame batches), document it as a
finding — "dreamed worlds lack metric consistency" is itself a result — and hold
at rung 1 for the product while flagging depth-consistency as a Track B training
target.

---

### **Phase 14 — The game mechanic prototype (Milestone M4)**

Working title: **CARTOGRAPHER**. Design thesis: *make the tech's weakness the
game's premise.* The world is a dream; it only persists where you've mapped it.

- **Core verb — Anchor:** the player plants an anchor (Phase 12's anchor key) to
  crystallize the current vista. Anchored views persist (atlas keyframe + geometry
  fuse + front-door anchor if available); unanchored space re-dreams itself when
  you look away. The persistence machinery is 1:1 diegetic.
- **Core resource — Anchors are finite.** Choosing *what* deserves permanence is
  the strategy layer. Atlas memory management (LRU, keyframe budget) is literally
  the game's economy.
- **Core loop — Get home:** you are dropped far from a landmark, free-roam
  (the world mutating behind you), and must anchor a breadcrumb trail you can
  actually navigate back along. Success is measurable: it is G3/G4 with a player
  in the loop. The minimap is rendered from the Phase 13 geometry store.
- **Failure aesthetic:** registration failures and drift present as "the dream
  shifting" — fog, shimmer, low-confidence desaturation driven by the live
  registration response value. Bugs become weather.
- Prototype scope: one scene/prompt, 5-minute session, anchor + get-home loop,
  win/lose condition. Built inside `play_server.py`'s stack; no engine work.

**Gate:** one full get-home run captured on video where success visibly depended
on anchoring. Playtest with ≥ 2 humans; log whether they *felt* the world
persist. Budget: 3–5 days after Phases 12–13.

---

### **Phase 15 — Trained persistence (Track B; gated)**

Open only when: (a) Phase 10 FAILED (training is the only durability path), or
(b) Phases 10–14 are done and the product ceiling is compositor-bound. Then:

- **B1 Memory-tolerance LoRA:** self-generate training data from this very
  harness — rollouts where context frames are replaced by atlas/geometry
  reconstructions, target = the true continuation. Train a LoRA so the model stops
  stalling on memory-bearing context. Success criterion is *the Phase 8/9 negative
  flipping sign*: the same `redream`/`rigid` write-back arms that lost by −3 to
  −4 dB should win with the LoRA loaded. Beautiful symmetry: the failed
  experiments become the eval.
- **B2 Loop-closure post-training:** fine-tune on trajectories containing genuine
  revisits so the model learns to close loops natively. Data from scripted
  gauntlet rollouts and/or Owl-Control recordings if accessible.
- **B3 Register/memory tokens:** learned memory slots via added cross-attention
  (LoRA-style adapters), read-only from the rollout's perspective — a non-KV
  durability channel by construction.

Infra: no trainer exists in this repo. Budget 1–2 weeks to stand one up (the 1B
model is small enough for a single rented A100/H100 node via vast.ai) before any
result. Do not let an agent burn days here without the explicit gate above.

---

### **Phase 16 — The write-up and release (Milestone W1)**

Two artifacts, both largely writable **today** — do not wait for Phases 10–15:

1. **Tech report / arXiv-style post:** *"Writing memory into a frozen world
   model's context doesn't work — and what does."* The Phase 6–9 arc is a clean,
   publishable story: two controlled negatives with a mechanistic fingerprint
   (dx runaway, dynamics stall), a replicated positive recipe (+3.4 dB, 6/6, ×2),
   and an exact channel-splitter (`redream` vs `redream_nowb`) isolating the
   failure to the KV write itself. Add Phase 10's verdict as the capstone
   whichever way it goes — PASS and FAIL both upgrade the paper.
2. **Repo release:** the Permanence Gauntlet (Phase 11) + the compositor + the
   demo MPVs as a public benchmark others can run against their own world models.
   The claim "here is a benchmark your world model will fail, and the recipe that
   passes it" is the credibility engine for everything else.

---

## 4. Milestone ladder (what "done" looks like at each rung)

| # | Milestone | Proof artifact |
|---|---|---|
| W1 | Negative-result write-up drafted | `docs/PAPER.md` draft committed |
| M1 | Live persistent spin in browser, memory toggle, ≥20 fps | weblink + 30 s MP4 |
| M2 | Permanence Gauntlet v1 + baseline table | `docs/GAUNTLET.md` + report page |
| M3 | True-durability verdict (Phase 10, either direction) | `docs/ANCHOR_RESULTS.md` |
| M4 | Walkable persistence (translation, ≥1 geometry rung) | G3 pass + video |
| M5 | CARTOGRAPHER get-home vertical slice | playtest video, 2+ humans |
| M6 | Trained-persistence result (if gated open) | Phase 8/9 arms flipping sign |

Suggested order: **W1 draft → Phase 10 → Phase 11 → Phase 12 (M1) → Phase 13 →
Phase 14 (M5) → Phase 15/16.** W1 first because it costs nothing but writing and
locks in the story while it's fresh; Phase 10 next because its verdict steers
everything downstream.

---

## 5. Operating rules for the long-running agent

1. **Gate every ~24 h of compute.** Write an intermediate results note, check it
   against the phase's kill criteria, and stop the phase if a gate says stop. No
   phase runs >3 days without a committed RESULTS doc.
2. **Replicate the incumbent in every sweep.** `atlas_nowb` at +3.4 ± 1 dB is the
   canary; if it doesn't replicate, the harness is broken — fix before proceeding.
3. **Paired stats or nothing.** n=6 (2 scenes × 3 seeds) minimum; 4×4 for headline
   claims; report ± as before; wins/n alongside every delta.
4. **Respect the dead channel.** Any proposal that writes latents into KV on the
   frozen model is out of scope unless a Track B LoRA is loaded. No exceptions,
   no matter how clever the laundering.
5. **New representation or new claim** — every new experiment must state, in its
   PLAN doc, which one it changes. If it can't, it's a dose tweak; drop it.
6. **Preserve artifacts before cleaning.** `bench_out/` is gitignored; MP4s, CSVs,
   and report pages are the demo trail. Copy keepers into `docs/` assets or an
   archive dir before any cleanup.
7. **Doc convention is load-bearing.** PLAN → run → RESULTS → update `HANDOFF.md`.
   Future agents (and the human) navigate by these docs, not by chat history.
8. **Keep CPU tests green** and add them for every new module. They are what let
   a fresh agent trust the harness without a GPU.
9. **Commit locally; never push without the user.**
10. **When in doubt, ship the demo.** Between a marginal +0.3 dB and making the
    live demo smoother or the video more striking, choose the demo. The science is
    already publishable; the product is what makes people care.

---

## 6. Decision tree (condensed)

```
Phase 10 (front-door anchoring)
├─ PASS  → true durability inference-side.
│          Anchoring becomes core: fold into Gauntlet, demo, game verb.
│          Track B → backburner. Ceiling: full persistent walkable world.
├─ PARTIAL → cadence-anchored compositor. Product path unchanged,
│          decay constant becomes a Gauntlet metric and a Track B target.
└─ FAIL  → third major negative: no inference-side durability at all.
           Write-up upgrades to "persistence requires training."
           Product = pure compositor (still ships M1–M5).
           Track B opens as the only durability path.

Phase 13 (geometry ladder)
├─ Homography+ passes G3      → cheap walkable world; ship it.
├─ Depth-warp needed & works  → RGBD atlas; minimap/collision unlocked.
└─ Depth on dreams incoherent → finding in itself; product holds at rung 1;
                                depth-consistency becomes a Track B objective.
```

---

*Previous handoff: `docs/HANDOFF_REPORT_2026-07-07.md`. Standing record:
`atlas_nowb +3.42 ± 0.92 dB, 6/6` (Phases 8–9, replicated). The dead channel is
KV latent write-back; the untested door is `append_frame`; the world is waiting.*
