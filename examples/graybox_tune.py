# uv run --dev python examples/graybox_tune.py
"""
Tune the surface-vector relationship in the graybox lab (examples/graybox.py).

Four questions, one sweep each:
  E1  lam        how hard may the memory surface pull on v before it hurts?
  E2  schedule   which Euler steps should carry the guidance (all / late / last)?
  E3  align      geometry-only (gainbias) vs tone-correcting (none): the
                 regime-vs-seams trade, quantified.
  E4  pose       oracle retrieval vs SLAM retrieval: with appearance registration
                 doing the fine alignment, how much does retrieval quality matter?

Prints one markdown table per experiment. Metrics are late-lap (laps 3-4 mean):
  rmse   static fidelity error vs truth (lower = memory repairing forgetting)
  dyn    beacon phase correlation (1.0 = dynamics intact)
  seam   boundary gradient excess (tone-mismatch seams at the mask edge)
  tone   output gain error vs truth on fresh cells (regime drift reaching frames)
  ncc    memory-to-view registration quality
"""
import sys

sys.path.insert(0, "examples")
import numpy as np

from graybox import GuidanceCfg, run_graybox

LAPS = 4.0
SEED = 0


def late(vals):
    v = [x for x in vals[2:] if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def row(name, res):
    return (f"| {name:<28} | {late(res['static_rmse']):5.2f} "
            f"| {late(res['beacon_corr']):5.2f} | {late(res['seam']):6.2f} "
            f"| {late(res['gain_err']):5.3f} | {late(res['mem_ncc']):5.2f} |")


HEADER = ("| config                       |  rmse |  dyn  |  seam  | tone  |  ncc  |\n"
          "|------------------------------|-------|-------|--------|-------|-------|")


def main():
    base = run_graybox(n_laps=LAPS, cfg=None, seed=SEED)

    print("## E1 — guidance strength (static mask, gainbias, late sigmas)")
    print(HEADER)
    print(row("baseline (no guidance)", base))
    for lam in (0.1, 0.25, 0.5, 1.0, 2.0):
        res = run_graybox(n_laps=LAPS, cfg=GuidanceCfg(lam=lam), seed=SEED)
        print(row(f"lam={lam}", res))

    print("\n## E2 — guidance schedule (lam=0.35)")
    print("(expected identical in this surrogate: the analytic target makes the")
    print(" terminal Euler step history-erasing — see sample_frame docstring. The")
    print(" schedule question needs the real engine.)")
    print(HEADER)
    for name, sig in (("all sigmas", (1.0, 0.7, 0.45, 0.25, 0.1)),
                      ("late (default)", (0.45, 0.25, 0.1)),
                      ("last only", (0.1,))):
        res = run_graybox(n_laps=LAPS, cfg=GuidanceCfg(guide_sigmas=sig), seed=SEED)
        print(row(name, res))

    print("\n## E3 — tone alignment (lam=0.35, late)")
    print(HEADER)
    for al in ("gainbias", "none"):
        res = run_graybox(n_laps=LAPS, cfg=GuidanceCfg(align=al), seed=SEED)
        print(row(f"align={al}", res))

    print("\n## E4 — retrieval pose source (lam=0.35, late, gainbias)")
    print(HEADER)
    for name, orc in (("slam pose", False), ("oracle pose", True)):
        res = run_graybox(n_laps=LAPS, cfg=GuidanceCfg(oracle_pose=orc), seed=SEED)
        print(row(name, res))

    print("\nbaseline per-lap rmse:", [round(x, 2) for x in base["static_rmse"]])


if __name__ == "__main__":
    main()
