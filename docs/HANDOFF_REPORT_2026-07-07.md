# Handoff Report - spin-persistence - 2026-07-07

## Stop-State Summary

- Branch: `spin-persistence`
- Workspace: `/home/frosty40/overworld`
- Latest result commit before this handoff report: `9ba0bc2f456a3b57c89a7c9b446b5fdf04cf9782`
- Handoff report commit: the doc-only commit containing this file, subject
  `Add Phase 9 handoff report`
- Push state: local commits are not pushed in this handoff.
- Report timestamp: `2026-07-07T10:55:11-05:00`
- Current conclusion: Phase 9 warmstart / re-dream is complete and negative for durability.
- Current online review server: `python3 -m http.server 8767 --bind 0.0.0.0 --directory bench_out/warm`, observed as PID `607953`.

## Start Here

1. Read this file.
2. Read `docs/WARMSTART_RESULTS.md` for the controlled Phase 9 result.
3. Read `docs/WARMSTART_PLAN.md` only if you need the original Phase 9 condition and runner contract.
4. Read `docs/RIGID_RESULTS.md` for the Phase 8 comparator and the write-back failure mechanism.
5. Open the online review report if the server is still alive:
   - LAN: `http://192.168.1.176:8767/report/index.html`
   - Alternate LAN: `http://192.168.1.165:8767/report/index.html`
   - Tailnet: `http://100.124.153.1:8767/report/index.html`

If the server is down, restart it from the repo root:

```bash
python3 -m http.server 8767 --bind 0.0.0.0 --directory bench_out/warm
```

## Commit Chain To Know

```text
<current> Add Phase 9 handoff report
9ba0bc2 Document Phase 9 warmstart sweep result
3338c8e Snap warmstart sigma to live scheduler grid
cd75410 Document Phase 9 warmstart runbook
95ab61d Phase 9 warmstart runner and CPU tests
4b41fce Phase 9 plan: warmstart/re-dream - laundered memory write-back (handoff, untested)
2b97dbc Phase 8: rigid latent projection - write-back killed, atlas+overlay stack sets +3.40 dB record
e028533 Phase 7 sweep: dynamics finding replicates 6/6; fidelity gain is a null
a57a23a Phase 7: SLAM loop closure for 360 spin persistence
185c721 Phase 6: pose-indexed world atlas - inference-side scene permanence
```

## Phase 9 Controlled Result

Condition source: `docs/WARMSTART_PLAN.md`

Run label: `new_experiment`

Runner:

```bash
uv run --dev python examples/warm_probe.py \
  --arms revisit,atlas,atlas_nowb,redream,redream_nowb,redream_far,atlas_redream \
  --M 1 --restamp-offset 4 \
  --n-scenes 2 --seeds 3 \
  --csv bench_out/warm/warm.csv
```

Resolved run condition from `bench_out/warm/warm.log`:

- Model: `Overworld/Waypoint-1.5-1B`
- Device: `cuda`
- Protocol: yaw-only return-to-start hysteresis
- `K=64`, `settle=8`, `yaw_mag=0.2`
- Scenes/seeds: 2 scenes x 3 seeds, seed base 1234
- Hard regime: heading <= 3.2
- Tuned atlas density: `M=1`, restamp offset 4
- Requested redream sigma: `0.3`
- Live scheduler grid: `[1.0, 0.8984375, 0.75, 0.30078125, 0.0]`
- Actual redream grid point: `0.30078125`
- Source hashes printed by runner: `warm_probe=92e3a6d39a43f6c5`, `warmstart=9b46430c4b20add9`, `plan=06ba3091098916fc`

Artifacts:

- Metrics page: `bench_out/warm/index.html`
- Full report page: `bench_out/warm/report/index.html`
- Raw CSV: `bench_out/warm/warm.csv` with 1344 rows
- Log: `bench_out/warm/warm.log`
- Summary JSON: `bench_out/warm/summary.json`
- Pilot CSV: `bench_out/warm/pilot.csv`
- Invalid first pilot log: `bench_out/warm/pilot_failed_offgrid_20260707T0012.log`

Primary result:

| paired, hard regime, n=6 | delta PSNR | wins |
|---|---:|---:|
| `redream - redream_nowb` | `-4.28 +/- 1.63 dB` | `0/6` |
| `redream - revisit` | `-2.90 +/- 1.34 dB` | `0/6` |
| `redream_nowb - revisit` | `+1.37 +/- 0.85 dB` | `6/6` |
| `atlas_nowb - revisit` | `+3.42 +/- 0.92 dB` | `6/6` |
| `atlas_nowb - atlas` | `+0.80 +/- 0.21 dB` | `6/6` |
| `atlas_redream - atlas` | `-3.72 +/- 1.19 dB` | `0/6` |

Projection diagnostics:

| arm | accept | write-back | mean accepted dx | mean response |
|---|---:|---:|---:|---:|
| `atlas_nowb` | 100% | 0% | 7.5 px | 0.538 |
| `redream` | 87% | 87% | 130.3 px | 0.268 |
| `redream_nowb` | 99% | 0% | 17.4 px | 0.487 |
| `redream_far` | 43% | 43% | 119.8 px | 0.106 |
| `atlas_redream` | 98% | 98% | 61.7 px | 0.387 |

Interpretation:

- Laundered write-back does not restore durability.
- The re-dreamed output is useful when it is not written back (`redream_nowb +1.37 dB`, 6/6).
- Writing the re-dreamed latent into KV makes the rollout worse than revisit and far worse than the no-write-back isolation arm.
- The dx fingerprint is the same failure basin as Phase 8: memory-bearing context edits stall dynamics and require large corrective registration shifts.
- The incumbent no-write-back inference recipe replicated: tuned atlas plus output overlay (`atlas_nowb`) is +3.42 dB over revisit, matching Phase 8's +3.40 dB record within run noise.

## Phase 8 Comparator

Source: `docs/RIGID_RESULTS.md`

Phase 8 established:

- Raw rigid write-back lost: `rigid - revisit = -1.34 +/- 0.85 dB`, 0/6.
- No-write-back overlay helped: `rigid_nowb - revisit = +0.45 +/- 0.13 dB`, 6/6.
- Tuned atlas plus output overlay set the prior record: `atlas_nowb - revisit = +3.40 +/- 0.91 dB`, 6/6.
- Write-back caused a dx runaway around 10 px to 250 px across the pan-back; no-write-back stayed bounded.
- Alignment gates did not fix write-back; the failure is intrinsic to hard latent edits entering the frozen model's KV context.

Phase 9 tested whether sampler laundering could make that write-back legal. It did not.

## Online Review Artifacts

The report page was created under `bench_out/warm/report/` and served from `bench_out/warm`.

Verified before this handoff:

- `http://192.168.1.176:8767/report/index.html` returned `200 OK`.
- `http://192.168.1.165:8767/report/index.html` returned `200 OK`.
- `http://100.124.153.1:8767/report/index.html` returned `200 OK`.
- `http://192.168.1.176:8767/turnaround_360/atlas_before_after_h264.mp4` returned `200 OK`.
- `bench_out/warm/report/screenshot.png` was captured from Chromium and visually inspected.

Important boundary: `bench_out/` is gitignored. The online report, CSVs, logs, screenshots, and MP4s are local artifacts, not tracked repo files. The source docs copied into `bench_out/warm/report/source/` are convenience copies for the online page, not canonical sources.

## 360p Turnaround MP4

User-requested visual artifact:

- Review page: `bench_out/warm/turnaround_360/index.html`
- Browser-friendly MP4: `bench_out/warm/turnaround_360/atlas_before_after_h264.mp4`
- Original MP4: `bench_out/warm/turnaround_360/atlas_before_after.mp4`
- Stills: `bench_out/warm/turnaround_360/atlas_before_after_stills.png`, `bench_out/warm/turnaround_360/atlas_returned_to_start.png`
- Log: `bench_out/warm/turnaround_360/run.log`

Command:

```bash
uv run --dev python examples/atlas_demo.py \
  --out bench_out/warm/turnaround_360 \
  --K 64 --settle 8 --seed 1236 --scene 0 \
  --yaw-mag 0.2 --M 1 --offset 4 --k 1 --fps 30
```

Properties:

- H.264 MP4
- `1292x360`
- 288 frames
- 9.6 seconds
- Visual condition: no-memory baseline left, pose atlas right
- Run log still: max-advantage heading 1.6, +4.26 dB

Boundary: this is a controlled visual artifact from `examples/atlas_demo.py`; it is not a Phase 9 redream result.

## Verification Run At Handoff

CPU tests were rerun on 2026-07-07 before writing this report:

```bash
uv run --dev pytest examples/test_warmstart.py -q
# 13 passed in 0.61s

uv run --dev pytest examples/test_rigid.py examples/test_graybox.py -q
# 22 passed in 65.76s
```

Online report reachability was checked with `curl -I` against LAN and tailnet URLs.

## Stop Rule

Do not keep launching sigma-0.3 laundered write-back runs or simple scheduler-dose variants as if they are new evidence.

The pilot plus full sweep are two verified negative runs in the same representation class and failure basin:

- memory-bearing write-back causes large dx;
- the rollout degrades;
- no-write-back scoring remains useful;
- writing memory-bearing latents into KV is the failure channel.

`redream75` is a dose diagnostic only if explicitly requested. It is not a new representation.

## Valid Next Work

Valid next work must change the representation/search space or change the claim:

1. Write up the training-side result: persistence cannot be bolted onto this frozen model by writing memory-bearing latents into context, even after sampler laundering.
2. Pursue no-write-back image-channel improvements only, clearly scoped as overlay/init quality rather than durability.
3. Design a genuinely different mechanism, such as trained persistence, context training, or a non-KV durability channel.
4. Revisit `warm_fast` only as a different runtime/mechanism claim, not as more sigma-0.3 laundered write-back.
5. If pursuing the older 180-degree-out-and-back idea, do not use command counting as degrees. Prior calibration was roughly 805-1063 command units per revolution, so each leg is hundreds of frames and needs content-defined degrees.

## Gotchas

- Run-to-run nondeterminism is large. Preserve paired within-trial comparisons; do not compare independent reruns as if they are identical worlds.
- Never call `engine.prep_inputs` twice per frame. It advances `frame_ts` and `camera_yaw`; `redream_step` already reuses `inputs`.
- VAE decode is a temporal stream. Snapshot before provisional decode, then rewind and re-decode only on accept.
- `partial_denoise` must run with `kv_cache.set_frozen(True)`.
- Off-grid renoise sigmas are intentionally rejected. The live bf16 grid value for 0.3 is `0.30078125`.
- `bench_out/` is ignored by git. Preserve important local artifacts before cleaning.
- The local review URLs only work while this host and the `8767` server are alive.

## What Not To Claim

- Do not claim Phase 9 restored durability.
- Do not claim the 360p MP4 is a redream result.
- Do not claim online access is public internet deployment; it is local LAN/tailnet hosting.
- Do not call future same-basin sigma tweaks new evidence unless the user explicitly requested that diagnostic.
