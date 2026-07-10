# Blocks lane results - LegoGen primitive architecture, mechanics proxy

**Status:** complete (aggregate + visual packet).
**Branch:** `spin-persistence`.
**Lane:** surreal/OOD primitives (`docs/SURREAL_PRIMITIVES_PLAN.md`); asset
condition `bench_assets_blocks/PROVENANCE.md` (LegoGen commit `6c147f5`).
**Run label:** `mechanics_proxy`. This is a new asset distribution and must not
be pooled with `bench_assets` biome aggregates.

## Condition

`Overworld/Waypoint-1.5-1B`, CUDA, no quantization, `K=64`, `settle=8`,
`yaw_mag=0.2`, `M=1`, restamp offset 4, post16 PSNR vs start. 4 scenes x 3
seeds, seed base 1234. Assets: `bench_assets_blocks` — four black/white surreal
LegoGen classical-architecture scenes (Doric monumental, Ionic canonical,
Random compact, Doric canonical), each with one asymmetrical landmark. Frontier
code at `anchor_probe=f7a55aa94cce51d7 anchor=de3aa308191920e2` (identical to
the Phase 10f frontier), run from a clean worktree at `1af98df`.

## Artifacts

- CSV: `bench_out/blocks_proxy/anchor_blocks.csv` (5789 rows)
- Log: `bench_out/blocks_proxy/run.log`
- Report: `bench_out/blocks_proxy/report/index.html`
- Assets + provenance: `bench_assets_blocks/`
- Contact sheet: `docs/permanence_assets/blocks_contact_sheet.png`

## Aggregate Result (n=12 paired trials)

Hard-regime canary:

```text
atlas_nowb - revisit = +2.44 +/- 1.17 dB, 12/12
```

Post16 paired deltas:

| arm | vs `atlas_nowb` | vs `revisit` | wins | accept |
|---|---:|---:|---:|---:|
| `anchor4_full` | `+3.28 +/- 2.83 dB` | `+5.84 +/- 2.81 dB` | `12/12` | `96%` |
| `anchor4_full_pose02` | `+3.28 +/- 2.83 dB` | `+5.84 +/- 2.81 dB` | `12/12` | `96%` |
| `anchor4_full_far` | `-1.76 +/- 2.95 dB` | `+0.80 +/- 3.02 dB` | `4/12` | `27%` |

Accepted-anchor fingerprint: mean `|dx| 14.1 px` (healthy band), mean resp
`0.309`. One trial (Doric-canonical scene, seed 1234) accepted at `35.5 px`,
above the biome-calibrated 7-20 px band; flagged, not hidden.

## Interpretation

1. **The memory stack transfers to the OOD primitive-architecture
   distribution.** The model carries the black/white surreal style, the atlas
   overlay canary lands 12/12, and front-door temporal anchoring wins every
   paired trial with a margin (+3.28 dB over `atlas_nowb`) comparable to the
   biome result (+4.35 dB, Phase 10f). Registration responses are healthy.
2. **Pose aliasing is worse in primitive worlds — pose admission matters more
   here, not less.** In one trial (Random-compact scene, seed 1236) the
   ungated far-pose control registered a *wrong-pose* keyframe at response
   `0.910` with `dx 0.1 px` — a near-perfect content match at the wrong yaw —
   anchored it, and lost `-5.5 dB` vs revisit. Sparse symmetric geometry
   produces look-alike views that phase correlation cannot distinguish.
   Response-based admission alone is provably insufficient on this
   distribution; pose-distance admission is load-bearing.
3. `anchor4_full` and `anchor4_full_pose02` were identical in every trial: the
   mild yaw protocol keeps retrieval pose-consistent, so the pose gate never
   fired. Consistent with Phase 10f on biome.

## Visual Packet

`bench_out/blocks_proxy/visual_review/index.html` — two cases, four arms each,
frames-first contact sheets (start / return-anchor / post16 / post64 / error
heat). Reruns are separate trials from the aggregate CSV (run-to-run
nondeterminism is expected; the packet is a review artifact, not the
benchmark).

- `seed_02_1235_blocks_best`: the model carries the B/W temple+obelisk world;
  `anchor4_full` holds 22.92 dB at post16 in the rerun.
- `seed_02_1236_blocks_alias`: the ungated far control anchors a wrong-pose
  look-alike at `resp 0.923, dx 0.1 px` — and by post16 the rollout has
  **collapsed out of the B/W distribution entirely** into an FPS-HUD
  hallucination (12.51 dB), the model's training prior surfacing. Wrong-pose
  memory injection through the front door does not degrade gracefully; it can
  break the dream. This is the strongest qualitative argument yet for
  admission gating, and a failure-anthology keeper.

## Boundary

- `mechanics_proxy` label: 12 paired trials on a new asset distribution; no
  cross-distribution claims. The biome canary target (+3.4 +/- 1 dB) does not
  apply here; the in-run canary (12/12 positive) is the harness check.
- v2 assets from today's BlockN push (commit `9370714`,
  `bench_assets_blockn/`) exist but have not been probed yet.
