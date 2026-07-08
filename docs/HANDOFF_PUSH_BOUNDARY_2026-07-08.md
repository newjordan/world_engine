# Handoff — Push the World-Model Boundary

**Written:** 2026-07-07T21:41:01-05:00 local host time.
**Repo:** `/home/frosty40/overworld`
**Branch:** `spin-persistence`
**HEAD at handoff:** `48f9ca7a644c8e9c3e9650260ddca839a02b37f6`
**Status:** Phase 10 is controlled-complete. The next agent should not re-run the
same front-door anchor sweep as if it is new evidence.

## Read First

1. `docs/ANCHOR_RESULTS.md` — latest controlled result and boundary.
2. `bench_out/anchor/visual_review/index.html` — frames-first visual review. Use
   this before the numeric report.
3. `docs/NORTHSTAR.md` — long-horizon sequence after Phase 10.
4. `docs/HANDOFF_REPORT_2026-07-07.md` — Phase 9 context and historical stop rules.

## Current Stop State

Working tree has uncommitted Phase 10 work:

```text
 M HANDOFF.md
?? docs/ANCHOR_PLAN.md
?? docs/ANCHOR_RESULTS.md
?? docs/HANDOFF_PUSH_BOUNDARY_2026-07-08.md
?? examples/analyze_anchor.py
?? examples/anchor.py
?? examples/anchor_probe.py
?? examples/anchor_visual.py
?? examples/test_anchor.py
```

Do not discard these. They are the Phase 10 runner, visual-review surface, results
doc, and this handoff.

Server still running:

```text
python3 -m http.server 8771 --bind 0.0.0.0 --directory /home/frosty40/overworld/bench_out/anchor
PID: 1074573
```

Verified URLs:

- Visual review: `http://100.124.153.1:8771/visual_review/index.html`
- Numeric report: `http://100.124.153.1:8771/report/index.html`

Current GPU state at handoff: no anchor jobs running. GPU is occupied by a separate
Nemotron vLLM server:

```text
VLLM::EngineCore PID 1418258, about 46968 MiB GPU memory
vLLM parent PID 1417863, served model:
/home/frosty40/models/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4
host 127.0.0.1 port 18101
```

Before launching any Waypoint run, check GPU state again:

```bash
nvidia-smi
ps -eo pid,ppid,stat,comm,args | rg 'anchor_probe|anchor_visual|VLLM::EngineCore|vllm serve'
```

## What Was Actually Proven

Phase 10 tested front-door `append_frame` anchoring on Waypoint-1.5-1B:

```bash
uv run --dev python examples/anchor_probe.py \
  --arms revisit,atlas_nowb,anchor,anchor_blend,anchor_k1,anchor_k4 \
  --M 1 --restamp-offset 4 \
  --n-scenes 2 --seeds 3 \
  --run-label new_experiment \
  --csv bench_out/anchor/anchor.csv
```

Result:

| comparison | delta PSNR | wins |
|---|---:|---:|
| `atlas_nowb - revisit` hard-regime canary | `+3.47 +/- 1.00 dB` | `6/6` |
| `anchor - atlas_nowb` at post16 | `-0.78 +/- 1.11 dB` | `1/6` |
| `anchor_blend - atlas_nowb` at post16 | `-1.10 +/- 0.70 dB` | `0/6` |
| `anchor_k1 - atlas_nowb` at post16 | `-0.75 +/- 0.40 dB` | `0/6` |
| `anchor_k4 - atlas_nowb` at post16 | `-0.51 +/- 1.46 dB` | `3/6` |

Verdict: front-door anchoring as implemented did **not** produce true durability.
`anchor_k4` is the best product-side signal: it kept mean accepted dx healthy
(`13.3 px`) and sometimes looked better in visual cases, but it still failed the
Phase 10 gate.

## Important Artifact Boundaries

Use these:

- `bench_out/anchor/anchor.csv` — controlled n=6 aggregate.
- `bench_out/anchor/anchor.log` — command, source hashes, resolved condition.
- `bench_out/anchor/report/index.html` — numeric report.
- `bench_out/anchor/visual_review/index.html` — frames-first visual review.
- `bench_out/anchor/visual_review/manifest.json` — visual packet provenance.
- `bench_out/anchor/visual_review.log` — visual rerun log.

Do **not** cite these as results:

- `bench_out/anchor/pilot.csv`
- `bench_out/anchor/pilot_report/`

Reason: first pilot left active atlas pins installed during post-window scoring.
It is explicitly invalid for post-window interpretation. See
`bench_out/anchor/PILOT_INVALID_NOTES.md`.

`pilot_fixed.csv` and `cadence_pilot.csv` are scout artifacts only.

## What Not To Do Next

- Do not rerun `anchor`, `anchor_blend`, `anchor_k1`, or `anchor_k4` unchanged and
  call it new evidence. That representation was tested n=6.
- Do not make another numeric-only report. Any world-model result needs frames
  first: raw start, return/anchor, post16, post64, and error heat.
- Do not claim true durability from `anchor_k4`; it is a product-side hint, not a
  scientific pass.
- Do not compare selected visual reruns to the aggregate as if they were the
  aggregate. The visual packet explains modes; `anchor.csv` carries the statistic.
- Do not clean `bench_out/anchor/` unless keepers are copied somewhere durable.

## Best Next Moves To Push the Boundary

### 1. Phase 10b: Four-frame front-door anchor, not repeated single frame

The biggest untested weakness in Phase 10 is that the anchor append batch repeats
one RGB reconstruction 4x. Waypoint-1.5 expects a 4-frame temporal chunk. Repeating
one frame may be temporally OOD even though the image enters through the front door.

Create `docs/ANCHOR4_PLAN.md` before running. New representation:

- reconstruct four micro-poses for the `[4,720,1280,3]` append batch;
- use heading estimates around the current return pose, not one static image x4;
- compare against the exact Phase 10 arms and keep `atlas_nowb` canary;
- post-window gate remains post16 vs start, overlay/atlas off.

Minimum arms:

```text
revisit, atlas_nowb, anchor4_linear, anchor4_blend, anchor4_k4
```

This is a valid representation change. It directly attacks the biggest unknown in
the Phase 10 boundary.

### 2. Full-frame / confidence-map image anchoring

Phase 10 only pasted the SLAM-certified band into the current image. That may be
too weak for the VAE/front-door cache to move the model state. A second legal
image-channel experiment is full-frame or soft confidence-map anchoring:

```text
anchor_full, anchor_full_blend, anchor_confmap
```

This must be framed as changing the reconstruction representation, not as a dose
sweep. Include wrong-pose/farthest controls because a full-frame append can become
cosmetic texture injection fast.

### 3. Move to Phase 11 Gauntlet if pushing product/benchmark

If the goal is to build the field artifact rather than squeeze Phase 10, start
`examples/gauntlet.py` and `docs/GAUNTLET.md`. Preserve the visual-review standard:
every gauntlet run should emit a frames-first HTML report plus CSV.

Minimum shippable Gauntlet v1:

- G1 Spin: import current return-to-start.
- G2 Glance: 90-degree look-away/return.
- G4 Patrol: repeated spin loops and drift curve.

Do not start with G3 translation unless you can spend time on content-defined
distance calibration. The prior notes warn command counts are not degrees/meters.

### 4. Product path: live compositor demo

The product path is still alive even after Phase 10 scientific fail. The durable
experience can be compositor-side:

- integrate `atlas_nowb` / `anchor_k4`-style cadence into `examples/play_server.py`;
- add a memory toggle;
- record the before/after spin video;
- keep fps target >=20.

This is not a true-durability claim. It is the demo/product track.

## Suggested Immediate Resume Command

If continuing science:

```bash
uv run --dev pytest examples/test_anchor.py examples/test_warmstart.py -q
```

Then write `docs/ANCHOR4_PLAN.md` and only then implement `examples/anchor4_probe.py`.
Do not launch without the plan doc and resolved condition printout.

If continuing benchmark/product:

```bash
python3 -m http.server 8771 --bind 0.0.0.0 --directory /home/frosty40/overworld/bench_out/anchor
```

Open the visual packet first:

```text
http://100.124.153.1:8771/visual_review/index.html
```

Then start `docs/GAUNTLET.md` or `examples/play_server.py` integration.

## Verification At Handoff

Last checks run successfully before this handoff:

```bash
python3 -m py_compile examples/anchor.py examples/anchor_probe.py \
  examples/analyze_anchor.py examples/anchor_visual.py examples/test_anchor.py

uv run --dev pytest examples/test_anchor.py examples/test_warmstart.py -q
# 20 passed

git diff --check
# clean
```

Visual packet was browser-screenshot inspected:

```text
bench_out/anchor/visual_review/screenshot.png
```

The screenshot shows frames first, not tables first.
