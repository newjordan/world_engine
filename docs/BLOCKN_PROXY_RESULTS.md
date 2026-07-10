# BlockN lane results - LegoGen v2 semantic pipeline, mechanics proxy

> **DRAFT — UNVERIFIED.** Agent-drafted from the CSV; verify per docs/HANDOFF_2026-07-10.md Task 2 before citing, then remove this banner.

**Status:** complete (aggregate; no visual packet this pass).
**Branch:** `spin-persistence`.
**Lane:** surreal/OOD primitives (`docs/SURREAL_PRIMITIVES_PLAN.md`); asset
condition `bench_assets_blockn/PROVENANCE.md` (BlockN, formerly LegoGen,
commit `9370714`). This is a second, harder asset distribution — complements,
does not replace, `bench_assets_blocks` (v1) — and must not be pooled with
`bench_assets` biome aggregates or with the v1 `mechanics_proxy` numbers.
**Run label:** `mechanics_proxy`.

## Condition

`Overworld/Waypoint-1.5-1B`, CUDA, no quantization, `K=64`, `settle=8`,
`yaw_mag=0.2`, `M=1`, restamp offset 4, post16 PSNR vs start. 4 scenes x 3
seeds, seed base 1234. Assets: `bench_assets_blockn` — four B/W surreal scenes
generated through BlockN's documented front-door public API
(`BuildRequest -> core.architecture.service.build() -> BuildResult.scene`,
`docs/ARCHITECTURE.md` "Public API Example"), one per architect family,
spanning BlockN's simple-to-complex spectrum instead of v1's single classical-
temple family:

| scene | architect | variant | landmark |
|---|---|---|---|
| `seed_00` | bridge (circulation) | truss, width 14, 12+1 blocks | column, LEFT |
| `seed_01` | arcade (classical) | monumental, 5 bays, 13+3 blocks | floating slab (hovering y=8), RIGHT |
| `seed_02` | structural-grid (structure) | slab-frame, 4 bays x 3 levels, 136+14 blocks | stair-to-nowhere, RIGHT |
| `seed_03` | spire-cluster (spire) | citadel, 31+5 blocks | arch portal, LEFT |

Every build passed BlockN's hard quality gate (`quality == "pass"`); driver
`scripts/bench_render_blockn.py` (in-repo to `Blocks`, uncommitted) composes
each main product plus a flanking landmark product and renders through the
same off-screen PyVista path as v1. Full details in
`bench_assets_blockn/PROVENANCE.md`.

Probe run from a **clean worktree at `16dde6d`** (`overworld-wt-blocks`,
detached HEAD, verified clean at analysis time — the Phase 10g frontier, two
commits behind this repo's current `spin-persistence` tip). Source hashes
from `run.log`:

```text
script=6ccd09c949481551 anchor_probe=6ccd09c949481551 anchor=cdd28604b439fb40 plan=fb07722c4424b97c
```

(v1 blocks used `anchor_probe=f7a55aa94cce51d7 anchor=de3aa308191920e2` from
worktree `1af98df`; the hashes differ here because Phase 10g touched
`anchor.py`/`anchor_probe.py` between the two runs — expected, not a
condition mismatch.)

## Artifacts

- CSV: `bench_out/blockn_proxy/anchor_blockn.csv` (5795 rows)
- Log: `bench_out/blockn_proxy/run.log`
- Report: `bench_out/blockn_proxy/report/index.html` +
  `bench_out/blockn_proxy/report/summary.json` (generated this pass with the
  existing `examples/analyze_anchor.py`, unmodified — pure CSV/stdlib, no GPU)
- Assets + provenance: `bench_assets_blockn/`
- No visual packet this pass (would require a GPU rerun; out of scope for
  this CPU-only analysis pass — see Boundary).

## Aggregate Result (n=12 paired trials)

Hard-regime canary:

```text
atlas_nowb - revisit = +3.36 +/- 2.25 dB, 12/12
```

Post16 paired deltas:

| arm | vs `atlas_nowb` | vs `revisit` | wins (atlas / revisit) | accept |
|---|---:|---:|---:|---:|
| `anchor4_full` | `+1.24 +/- 3.23 dB` | `+2.88 +/- 2.46 dB` | `6/12` / `11/12` | `88%` |
| `anchor4_full_pose02` | `+1.24 +/- 3.23 dB` | `+2.88 +/- 2.46 dB` | `6/12` / `11/12` | `88%` |
| `anchor4_full_far` | `-0.99 +/- 3.70 dB` | `+0.65 +/- 4.19 dB` | `2/12` / `5/12` | `77%`* |

`accept` here is the console/run.log per-micro-crop admission rate (mean
across trials of the fraction of the K micro-yaw crops that cleared
`resp_min`/`max_shift_frac`). *For `anchor4_full_far`, `77%` is per-micro; the
trial-level "committed at least one crop" rate is `92%` (11/12) — see
Diagnostics below for why these differ and why the console's aggregate
`|dx|` prints as `nan`.

Accepted-anchor fingerprint (from `run.log`, verified against the CSV): mean
`|dx| 20.4 px`, mean resp `0.371`. Compare v1: `+3.28 dB` / `96%` accept /
`14.1 px` mean `|dx|` — **v2 is a materially weaker, noisier transfer**, not
a replication of v1's result on a new skin. See per-trial table and
Diagnostics for exactly where the weakness lives.

### Per-scene x seed paired post16 deltas

| scene (product) | seed | `anchor4_full` − `atlas_nowb` | `anchor4_full` − `revisit` | `atlas_nowb` − `revisit` (canary) | `anchor4_full_far` − `atlas_nowb` |
|---|---:|---:|---:|---:|---:|
| `seed_00` (bridge/truss) | 1234 | `+1.74` | `+2.34` | `+0.59` | `+5.38` |
| `seed_00` | 1235 | `+2.74` | `+4.91` | `+2.17` | `-4.58` |
| `seed_00` | 1236 | **`-1.90`** | `+0.78` | `+2.68` | `-2.68` |
| `seed_01` (arcade) | 1234 | **`-1.54`** | `+0.45` | `+1.99` | `-1.06` |
| `seed_01` | 1235 | **`-0.72`** | `+3.16` | `+3.87` | `-2.92` |
| `seed_01` | 1236 | **`-1.07`** | **`-0.73`** | `+0.34` | `-3.96` |
| `seed_02` (structural-grid) | 1234 | **`-1.10`** | `+1.43` | `+2.53` | `-0.56` |
| `seed_02` | 1235 | **`-2.51`** | `+0.86` | `+3.37` | `+7.63` |
| `seed_02` | 1236 | `+4.41` | `+5.56` | `+1.15` | `-2.15` |
| `seed_03` (spire-cluster) | 1234 | `+4.28` | `+5.79` | `+1.51` | `-2.77` |
| `seed_03` | 1235 | `+7.90` | `+7.03` | `-0.87` | `-2.62` |
| `seed_03` | 1236 | `+2.66` | `+2.93` | `+0.27` | `-1.57` |

Bold = loss. **The 6 losses vs `atlas_nowb`** are `seed_00/1236`, all three
`seed_01` seeds, and `seed_02/1234` + `seed_02/1235` — i.e. every loss but one
comes from the two most repetitive-bay scenes (`seed_01` arcade 5-bay,
`seed_02` grid 4x3); `seed_03` (spire-cluster, non-repeating assembly) has
zero losses on any comparison and the largest average margin. The single
loss vs `revisit` (`seed_01/1236`, `-0.73 dB`) sits inside the same scene.

## Diagnostics

### 1. The seed_01/1234 "disease band" trial (mean `|dx| 97.6 px`)

`anchor4_full` on this scene/seed uses a single loop-closure anchor event (no
cadence token → `_cadence` is `None`, so there is exactly one
`_append_anchor` call per trial, batching `K` micro-yaw crops). For this
trial the CSV's one `intervention` row reads:

```text
scene=seed_01 seed=1234 dx_px=97.60 resp=0.0802 reject=partial anchor_count=1
```

`run.log` confirms `accept 50%`: only half the micro-crops cleared
`resp_min=0.05`/`max_shift_frac=0.35`; the ones that *did* clear still carried
a mean `|dx|` of 97.6 px at a resp of 0.08 — barely above the response floor,
not a confident match. This is a genuinely marginal, near-garbage admission,
not a healthy anchor: post16 for this trial is `12.32 dB` vs `atlas_nowb`'s
`13.86 dB` (`-1.54 dB`, one of the 6 losses) and `revisit`'s `11.87 dB`
(`+0.45 dB` — barely better than doing nothing). The anchor did not actively
hurt relative to no intervention, but it bought almost nothing, and it lost
to the cheap cosmetic-overlay canary.

### 2. dx-band flags (7-20 px "healthy" band, from v1)

Applying v1's calibrated 7-20 px band to all 12 `anchor4_full` trials: only
**2 of 12** land inside it (`seed_00/1236` at `10.2 px`, `seed_01/1235` at
`13.0 px`). The other 10 split into two failure clusters rather than
scattering around the band:

- **Too tight (< 7 px, 5 trials):** `seed_00/1234` (2.3), `seed_00/1235`
  (2.8, itself a `reject=partial` trial), `seed_02/1236` (5.7, also
  `partial`), `seed_03/1234` (4.3), `seed_03/1236` (1.1). Small dx here
  mostly co-occurs with high resp on `seed_03` (healthy, near-static camera
  content) but is separately suspicious on `seed_01/1236` (see #4 below).
- **Too wide (> 20 px, 5 trials):** `seed_01/1234` (97.6, the disease trial
  above), `seed_02/1234` (28.7, partial), `seed_02/1235` (73.7, partial),
  plus the two far-arm-adjacent large numbers are separate (far arm table
  below).

v1's 7-20 px band was calibrated on one classical-temple scene family; it
does not describe v2's distribution well — the fingerprint here is bimodal,
not a tight band with one outlier. That is itself a finding (see
Interpretation #2).

### 3. Why `anchor4_full_far`'s aggregate `mean |dx|` prints as `nan`

Root cause confirmed as a **harness console-print artifact, not missing
data**. `_info_stats()` in `examples/anchor_probe.py` returns
`dx_mean = nan` for any single trial with zero accepted micro-crops (empty
list → `float("nan")`, by design). The final console aggregate then does
`_mean([s["dx_mean"] for s in proj_stats[arm]])` — an **unweighted mean of
per-trial means**, computed with plain `sum()/len()`. Python's `sum()`
propagates `nan` unconditionally, so **any single zero-accept trial poisons
the entire arm-level aggregate**, regardless of the other 11 trials.

For `anchor4_full_far` in this run, exactly one trial
(`seed_00/1236`, `accept 0%`) produced no admitted crop, which is enough to
nan the whole arm. Confirmed **not new to v2**: the identical bug fires on
the v1 blocks run (`bench_out/blocks_proxy/run.log` line 124:
`anchor4_full_far: ... mean |dx| nanpx`), where it happens far more often —
**7 of 12** v1 trials were zero-accept for the far arm (`accept 27%`) vs
**1 of 12** here (`accept 77%`).

Recomputing with `examples/analyze_anchor.py` (which pools dx directly over
logged CSV rows rather than averaging per-trial means, so a zero-accept
trial simply contributes no row instead of a `nan`) gives a real, finite
number both times:

| | v1 blocks | v2 blockn |
|---|---:|---:|
| far-arm trials with >=1 accepted crop | 5/12 | 11/12 |
| far-arm pooled mean `\|dx\|` (px) | `150.18` | `155.09` |

So the underlying magnitude is consistent across asset sets (~150 px, as
expected for an adversarial farthest-pose retrieval), but **v2's far control
is accepting far more often** (77% vs 27% per-micro, 92% vs 42% trial-level)
while still carrying ~150 px of shift — i.e. the resp-only floor is waving
through large-shift, likely wrong-pose keyframes more readily on the v2
assets than on v1's. This is a real, if secondary, finding: see Interpretation
#3.

### 4. `seed_01/1236` — a periodicity look-alike, this time inside the gated arm

`seed_01` is the monumental 5-bay arcade (repeating classical bays). At seed
1236 the `anchor4_full` (near/nearest-mode, gated) intervention admitted at
`dx=-0.02 px, resp=0.914` — and the adversarial `anchor4_full_far`
(farthest-mode, ungated) control for the *same trial* admitted at
`dx=0.03 px, resp=0.911`, essentially the identical match. Nearest and
farthest retrieval collapsing onto the same near-perfect-looking keyframe is
the same wrong-pose-alias signature v1 flagged in its Random-compact scene
(there: `resp 0.910, dx 0.1 px`, ungated far control only). The difference
here: it shows up in the **gated near-mode arm itself**, not only the
ungated far control — the repeating bay geometry makes "nearest" and
"farthest" pose retrieval indistinguishable by phase correlation, so pose
admission doesn't have two visibly-different candidates to discriminate
between in the first place. Outcome: this trial is one of the 6
`anchor4_full` losses vs `atlas_nowb` (`-1.07 dB`) and the *only* loss vs
`revisit` (`-0.73 dB`); the far-arm control on the same trial loses badly
(`-3.96 dB` vs `atlas_nowb`, `-3.61 dB` vs `revisit`).

### 5. pose02 vs plain

**Identical in every trial, confirmed at full row granularity** (not just
post16): all 1164 CSV rows for `anchor4_full_pose02` are byte-identical to
`anchor4_full` across every column (`psnr`, `ssim`, `dx_px`, `resp`,
`reject`, `anchor`, `alpha`, `anchor_count`). Same conclusion as v1: the mild
`yaw_mag=0.2` protocol keeps retrieval pose-consistent enough that the
`pose02` gate (reject if `|target_yaw - kf_frame_yaw| > 0.2`) never fires.
No divergence found.

## Interpretation

1. **The memory stack transfers to BlockN's v2 distribution, but more weakly
   and noisily than to v1.** The atlas overlay canary still lands 12/12
   (`+3.36 dB`), so the model still carries the B/W style and the harness
   check is sound. But `anchor4_full` wins only 6/12 vs `atlas_nowb` (v1:
   12/12) and its accepted-anchor fingerprint is bimodal rather than banded
   (2/12 trials in the 7-20 px band vs v1's near-universal 14.1 px average
   with one outlier). This is a real regression in registration quality on
   the new asset distribution, not a re-run of v1's result.
2. **Repetitive/periodic structural members drive the weak transfer, not
   the distribution as a whole.** Of the 4 scenes, the two built from
   repeating bays — `seed_01` (5-bay arcade) and `seed_02` (4x3 slab-frame
   grid) — produce 5 of the 6 losses and all of the `reject=partial`
   admission events outside `seed_00`'s single truss-scene partial
   (`seed_00`: 1/3 partial, 1/3 loss; `seed_01`: 1/3 partial + 1 alias-like
   clean admission, 3/3 loss; `seed_02`: 3/3 partial, 2/3 loss). `seed_03`
   (spire-cluster, a non-repeating assembly) has zero partial admissions and
   zero losses, with the largest win margins in the set (up to `+7.90 dB`).
   This is a 4-scene, `mechanics_proxy`-labeled observation — suggestive, not
   proven — but it is mechanistically plausible: repeating bays/trusses give
   phase correlation multiple equally-plausible offsets to lock onto, which
   should manifest as exactly what we see (partial admission and/or
   suspiciously perfect look-alike matches). A natural next probe would vary
   bay count on one architect family to test this directly.
3. **Pose-admission gating is, if anything, more load-bearing on v2 than
   v1, not less.** The far-arm's adversarial farthest-pose control accepts
   far more often here (77% per-micro / 92% per-trial) than on v1 (27% /
   42%) while still carrying the same ~150 px of registration shift — the
   resp-only floor is passing large-shift, likely-wrong-pose keyframes
   through more readily on these assets. And unlike v1, the periodicity
   look-alike this time reached the *gated* arm (`seed_01/1236`, Diagnostics
   #4), not only the ungated far control. Response-based admission alone
   remains provably insufficient; the case for pose-distance gating is at
   least as strong here as it was in v1 (even though the mild `yaw_mag=0.2`
   protocol never actually triggers the `pose02` gate in either asset set —
   Diagnostics #5).
4. **`anchor4_full` still beats doing nothing.** 11/12 wins vs `revisit`
   (`+2.88 dB`) hold up even with the weaker `atlas_nowb` comparison, and
   even the disease-band trial (Diagnostics #1) was a wash rather than a
   loss vs `revisit`. The front-door anchor is not actively harmful on this
   distribution; it just isn't reliably better than the cosmetic-overlay
   canary the way it was on v1.

## Boundary

- `mechanics_proxy` label: 12 paired trials on a second, harder asset
  distribution; no cross-distribution claims, and this run must not be
  pooled with the v1 `bench_assets_blocks` numbers or the biome aggregates.
  The biome canary target (`+3.4 +/- 1 dB`) does not apply; the in-run canary
  (12/12 positive) is the harness check, and it passes.
- No visual packet was generated in this pass (CPU-only analysis; a rerun to
  produce `visual_review/` frames would need GPU time, out of scope here).
- `bench_assets_blockn_sunset/` (BlockN's per-request style palettes on the
  sunset gradient) exists per `PROVENANCE.md` but has not been probed.
- The scene-repetition hypothesis in Interpretation #2 is drawn from 4
  scenes x 3 seeds; it is a lead for a follow-up probe (e.g. bay-count sweep
  on one architect), not a validated causal claim.
