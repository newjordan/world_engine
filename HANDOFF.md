# Handoff - spin-persistence - 2026-07-07

Read `docs/HANDOFF_REPORT_2026-07-07.md` first. It is the full current handoff report.

## Current State

- Branch: `spin-persistence`
- Latest result commit before this handoff report: `9ba0bc2f456a3b57c89a7c9b446b5fdf04cf9782`
- Handoff report commit: the doc-only commit containing this file, subject
  `Add Phase 9 handoff report`
- Push state: local commits are not pushed in this handoff.
- Phase 8: rigid write-back failure diagnosed; tuned atlas plus no-write-back overlay set
  the inference-side record at `+3.40 +/- 0.91 dB`, 6/6.
- Phase 9: warmstart / re-dream controlled sweep completed; laundered write-back failed.
- Online local review report, while server is alive:
  `http://192.168.1.176:8767/report/index.html`
- Tailnet report:
  `http://100.124.153.1:8767/report/index.html`

## Phase 9 Verdict

Primary comparison, hard regime, n=6:

| comparison | delta PSNR | wins |
|---|---:|---:|
| `redream - redream_nowb` | `-4.28 +/- 1.63 dB` | `0/6` |
| `redream - revisit` | `-2.90 +/- 1.34 dB` | `0/6` |
| `redream_nowb - revisit` | `+1.37 +/- 0.85 dB` | `6/6` |
| `atlas_nowb - revisit` | `+3.42 +/- 0.92 dB` | `6/6` |

Interpretation: the re-dreamed output helps when it is not written back, but writing
that memory-bearing latent into KV still stalls the rollout. Mean accepted dx is
`130.3 px` for `redream` write-back versus `17.4 px` for `redream_nowb`.

Stop rule: do not keep launching sigma-0.3 laundered write-back or simple dose tweaks
as if they are new evidence. `redream75` is only a user-requested diagnostic, not a new
representation.

## Artifacts

- Full handoff: `docs/HANDOFF_REPORT_2026-07-07.md`
- Phase 9 result doc: `docs/WARMSTART_RESULTS.md`
- Phase 9 plan/runbook: `docs/WARMSTART_PLAN.md`
- Phase 8 comparator: `docs/RIGID_RESULTS.md`
- Phase 9 metrics page: `bench_out/warm/index.html`
- Online report page: `bench_out/warm/report/index.html`
- Phase 9 raw CSV/log/summary: `bench_out/warm/warm.csv`, `bench_out/warm/warm.log`,
  `bench_out/warm/summary.json`
- 360p turnaround page: `bench_out/warm/turnaround_360/index.html`
- 360p MP4: `bench_out/warm/turnaround_360/atlas_before_after_h264.mp4`

`bench_out/` is gitignored. Preserve these local artifacts before cleaning.

## Current Server

Observed listener:

```text
0.0.0.0:8767 users:(("python3",pid=607953,fd=3))
```

Restart command from repo root:

```bash
python3 -m http.server 8767 --bind 0.0.0.0 --directory bench_out/warm
```

## Verification At Handoff

```bash
uv run --dev pytest examples/test_warmstart.py -q
# 13 passed in 0.61s

uv run --dev pytest examples/test_rigid.py examples/test_graybox.py -q
# 22 passed in 65.76s
```

Report URLs and linked media were checked with `curl -I`, and
`bench_out/warm/report/screenshot.png` was captured in Chromium and visually inspected.
