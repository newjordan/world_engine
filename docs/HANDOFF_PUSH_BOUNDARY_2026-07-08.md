# Handoff - Pose-Admitted Anchor Frontier

**Written:** 2026-07-08T16:19:58-05:00 local host time.
**Repo:** `/home/frosty40/overworld`
**Branch:** `spin-persistence`
**Frontier commit:** `f47aba3 Add pose-admitted anchoring frontier`
**Remote:** pushed to `origin/spin-persistence`
**Status:** Phase 10f is controlled-complete. The local network review site is
up. Do not rerun the same pose-admission sweep and call it new evidence.

## Read First

1. `docs/ANCHOR_POSE_ADMISSION_RESULTS.md` - current aggregate result and video boundary.
2. `docs/ANCHOR_POSE_ADMISSION_PLAN.md` - condition source for the promoted run.
3. `docs/ANCHOR_ADMISSION_RESULTS.md` - dx/response admission ablation that led here.
4. `bench_out/anchor_pose_admission/index.html` - local network landing page.
5. `bench_out/anchor_pose_admission/pan180_scene0_seed1235_20260708T155108-0500/index.html` - MP4 packet.
6. `bench_out/anchor_pose_admission/anchor_pose_admission_20260708T143846-0500_report/index.html` - analyzer report.

## Current Stop State

Frontier code and docs are committed and pushed:

```text
f47aba3 (HEAD -> spin-persistence, origin/spin-persistence) Add pose-admitted anchoring frontier
```

Local network site is running from ignored bench output:

```bash
python3 -m http.server 8784 --bind 0.0.0.0 --directory bench_out/anchor_pose_admission
PID: 2679897
```

Verified URLs at handoff:

- LAN wired: `http://192.168.1.176:8784/`
- LAN wifi: `http://192.168.1.165:8784/`
- Tailnet: `http://100.124.153.1:8784/`
- Direct MP4: `http://192.168.1.176:8784/pan180_scene0_seed1235_20260708T155108-0500/compare_2x2.mp4`
- Aggregate report: `http://192.168.1.176:8784/anchor_pose_admission_20260708T143846-0500_report/index.html`

All returned `200 OK` immediately before this handoff. Browser screenshot:

```text
bench_out/anchor_pose_admission/local_site_screenshot.png
```

No anchor jobs are running. Process scan only found the HTTP server. GPU query on
the GB10 reported utilization `0`; memory fields were `N/A` from `nvidia-smi`.

## What Was Actually Proven

Phase 10f tested pose-distance admission for full-frame, four-frame temporal
front-door anchoring on `Overworld/Waypoint-1.5-1B`.

Condition source:

```text
docs/ANCHOR_POSE_ADMISSION_PLAN.md
```

Long sweep command:

```bash
uv run --dev python examples/anchor_probe.py \
  --condition-source docs/ANCHOR_POSE_ADMISSION_PLAN.md \
  --arms revisit,atlas_nowb,anchor_full,anchor4_full,anchor4_full_pose02,anchor4_full_pose04,anchor4_full_far,anchor4_full_pose02_far,anchor4_full_pose04_far \
  --M 1 --restamp-offset 4 \
  --n-scenes 4 --seeds 4 \
  --run-label new_experiment \
  --csv bench_out/anchor_pose_admission/anchor_pose_admission.csv
```

Resolved condition:

- Model: `Overworld/Waypoint-1.5-1B`
- Device: CUDA
- Quantization: none
- Assets: `bench_assets`
- Protocol: yaw-only return-to-start hysteresis
- `K=64`, `settle=8`, `yaw_mag=0.2`
- Sweep: 4 scenes x 4 seeds, seed base 1234
- Metric: paired post16 PSNR against start
- Canary: `atlas_nowb`, `M=1`, restamp offset 4, `atlas_k=1`

Run artifacts:

- CSV: `bench_out/anchor_pose_admission/anchor_pose_admission_20260708T143846-0500.csv`
- Log: `bench_out/anchor_pose_admission/anchor_pose_admission_20260708T143846-0500.log`
- Report: `bench_out/anchor_pose_admission/anchor_pose_admission_20260708T143846-0500_report/index.html`
- Summary: `bench_out/anchor_pose_admission/anchor_pose_admission_20260708T143846-0500_report/summary.json`

Source hashes recorded in the run:

- `anchor_probe=f7a55aa94cce51d7`
- `anchor=de3aa308191920e2`
- `plan=c65c52a44b6213ff`

Aggregate result:

| arm | post16 delta vs `atlas_nowb` | wins | accept |
|---|---:|---:|---:|
| `anchor4_full` | `+4.35 +/- 2.92 dB` | `15/16` | `94%` |
| `anchor4_full_pose02` | `+4.35 +/- 2.92 dB` | `15/16` | `94%` |
| `anchor4_full_pose04` | `+4.35 +/- 2.92 dB` | `15/16` | `94%` |
| `anchor4_full_far` | `-1.30 +/- 1.34 dB` | `2/16` | `50%` |
| `anchor4_full_pose02_far` | `-1.12 +/- 0.51 dB` | `0/16` | `0%` |
| `anchor4_full_pose04_far` | `-1.12 +/- 0.51 dB` | `0/16` | `0%` |

Canary stayed valid:

```text
atlas_nowb - revisit = +2.94 +/- 0.99 dB, 16/16
```

Interpretation: pose02 and pose04 preserved the full useful `anchor4_full` gain
while rejecting all far-pose control anchors in this sweep. Promote
pose-distance admission over projected micro-frames. Partial chunks are allowed
when every projected micro-frame is pose-consistent.

## MP4 Review Packet

Full-pan review command:

```bash
uv run --dev python examples/anchor_pan_video.py \
  --condition-source docs/ANCHOR_POSE_ADMISSION_PLAN.md \
  --arms revisit,atlas_nowb,anchor4_full,anchor4_full_pose02 \
  --target-deg 180 --yaw-mag 4.0 --cap 500 \
  --scene 0 --seed 1235 \
  --out bench_out/anchor_pose_admission/pan180_scene0_seed1235_20260708T155108-0500
```

Review packet:

```text
bench_out/anchor_pose_admission/pan180_scene0_seed1235_20260708T155108-0500/
```

Important files:

- `compare_2x2.mp4` - H.264, 1920x1080, 30 fps, about 67 seconds.
- `revisit.mp4`
- `atlas_nowb.mp4`
- `anchor4_full.mp4`
- `anchor4_full_pose02.mp4`
- `manifest.json`
- `review_page.png`

Video condition:

- Scene: `seed_00`
- Seed: `1235`
- Target: 180 degrees
- Locked away steps: `215`
- Measured away angle: `180.95130264797103` degrees

Boundary: this MP4 packet is a controlled visual review artifact, not the
aggregate benchmark. In this full 180-degree visual stress case, `anchor4_full`
and `anchor4_full_pose02` both rejected at loop closure (`reject=empty`, low
response), so the video shows the current long-pan re-acquisition boundary. The
aggregate claim remains the Phase 10f yaw return-to-start sweep above.

## Do Not Cite These As Results

Invalid or partial artifacts from the hunt:

- `bench_out/anchor_admission/anchor_admission_20260708T124430-0500.*`
- `bench_out/anchor_admission/anchor_admission_20260708T125551-0500.*`
- `bench_out/anchor_pose_admission/pan180_scene2_seed1235_20260708T154315-0500.*`
- `bench_out/anchor_pose_admission/pan180_calibrate_20260708T154712-0500/`

Reasons: aborted admission controls, diagnostic accept-rate bug before analyzer
fix, incomplete 180-degree content target, and calibration-only run. Use them only
as implementation history.

## Code Changes In Frontier Commit

`f47aba3` includes:

- pose admission arms: `anchor4_full_pose02`, `anchor4_full_pose04`, and far controls;
- dx/response admission scout arms and far controls;
- shared `_admission_failure` path in `examples/anchor_probe.py` and `examples/anchor_visual.py`;
- analyzer fixes for one-shot accept accounting and finite dx/response stats;
- `examples/anchor_pan_video.py` for full-pan MP4 review packets;
- H.264 ffmpeg fallback in the MP4 writer;
- tests for admission fail-closed behavior;
- result docs for Phase 10e and Phase 10f.

## Verification Run Before Commit

Passed:

```bash
python3 -m py_compile examples/anchor.py examples/anchor_probe.py examples/anchor_visual.py \
  examples/anchor_pan_video.py examples/analyze_anchor.py examples/test_anchor.py

uv run --dev pytest examples/test_anchor.py -q
# 16 passed

git diff --check
# clean
```

Browser/site checks:

- `bench_out/anchor_pose_admission/pan180_scene0_seed1235_20260708T155108-0500/review_page.png`
- `bench_out/anchor_pose_admission/local_site_screenshot.png`
- root site and MP4/report URLs returned `200 OK`.

## What Not To Do Next

- Do not rerun unchanged dx/response gates; they were scout arms and are not the
  promoted algorithm.
- Do not claim full 180-degree long-pan permanence is solved. The MP4 stress case
  shows the opposite boundary at loop closure.
- Do not cite the visual packet as aggregate evidence. The CSV/report carry the
  aggregate result.
- Do not create another still-image-only review for this lane. The user asked for
  MP4 pan videos.
- Do not report raw JSON/logs alone. Keep review pages first, raw files second.

## Best Next Moves

### 1. Full 180-degree re-acquisition algorithm

The current system admits good anchors when retrieval is already pose-consistent.
The next representation change should attack re-acquisition after long pan:

- multi-hypothesis retrieval over yaw bins near loop closure;
- content-defined loop-closure search before append;
- temporal chunk reconstruction from the best consistent keyframe sequence;
- explicit no-match state, not silent fallback;
- preserve pose admission as the final gate.

This is a real representation change. It is not a threshold tweak.

### 2. Black-and-white surreal primitive world

The user explicitly prefers a simple black/white surreal environment with
primitive shapes, floating circles, and minimal visual clutter. If building a new
visual condition, make it a separate condition artifact before running. Do not
pretend it is the same as the current `bench_assets` condition.

### 3. Website / sharing path

If the local site dies, restart it:

```bash
python3 -m http.server 8784 --bind 0.0.0.0 --directory bench_out/anchor_pose_admission
```

Then open:

```text
http://192.168.1.176:8784/
```

If using tailnet:

```text
http://100.124.153.1:8784/
```

### 4. Next controlled run shape

Before launching a new long run, write a new plan doc with:

- condition source path;
- arms;
- target metric;
- accepted comparator;
- long-pan criterion;
- MP4 packet requirement.

Then make the runner print source hashes and resolved condition at start, as the
current runners do.
