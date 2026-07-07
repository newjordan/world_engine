# uv run --dev pytest examples/test_graybox.py -v
"""
CPU tests for the graybox surface-vector guidance lab (examples/graybox.py).

Each test pins one claim the lab exists to establish:
  1. the closed vector forgets (baseline fidelity decays per lap) while its dynamics
     stay healthy (the beacon self-tracks);
  2. appearance-registered memory guidance repairs the forgetting durably;
  3. an undifferentiated surfacing mask breaks dynamics (the Phase 7 finding, in
     miniature) and the two-class split restores them at no fidelity cost;
  4. tone alignment is a real trade: raw memory fights regime drift but pays in
     seams; aligned memory is geometry-only.

Runs are shared via module-scoped fixtures (each 4-lap run is a few seconds).
"""
import sys

sys.path.insert(0, "examples")
import numpy as np
import pytest

from graybox import GuidanceCfg, run_graybox

LAPS = 4.0


@pytest.fixture(scope="module")
def baseline():
    return run_graybox(n_laps=LAPS, cfg=None, seed=0)


@pytest.fixture(scope="module")
def guided_static():
    return run_graybox(n_laps=LAPS, cfg=GuidanceCfg(mask_mode="static"), seed=0)


@pytest.fixture(scope="module")
def guided_all():
    return run_graybox(n_laps=LAPS, cfg=GuidanceCfg(mask_mode="all"), seed=0)


# ----------------------------------------------------------------- the closed vector
def test_baseline_forgets(baseline):
    """No guidance: content out of view degrades and the degradation is written back,
    so fidelity decays lap over lap — the forgetting curve."""
    rmse = baseline["static_rmse"]
    assert rmse[1] > rmse[0] + 4.0, rmse
    assert rmse[3] > rmse[0] + 10.0, rmse


def test_baseline_dynamics_healthy(baseline):
    """The engine's own dynamics stay phase-locked without any guidance — required so
    a dynamics break can be attributed to the guidance, not the engine."""
    assert min(baseline["beacon_corr"]) > 0.95, baseline["beacon_corr"]


# ------------------------------------------------------------- guidance vs forgetting
def test_guidance_repairs_forgetting(baseline, guided_static):
    """Memory guidance on the static surface halves late-lap error, and the repair is
    durable (write_back re-anchors the belief, so laps 3-4 hold, not just lap 2)."""
    for lap in (1, 2, 3):
        assert guided_static["static_rmse"][lap] < baseline["static_rmse"][lap] - 3.0, \
            (lap, guided_static["static_rmse"], baseline["static_rmse"])


def test_guidance_registration_is_sound(guided_static):
    """The retrieved memory registers to the current view (NCC), and the appearance
    refinement absorbs a nonzero map bias — the term that, applied as a pose-computed
    shift, previously stalled the scroll in a runaway."""
    assert all(v > 0.8 for v in guided_static["mem_ncc"][1:]), guided_static["mem_ncc"]
    assert all(abs(v) > 1.0 for v in guided_static["refine"][1:]), \
        guided_static["refine"]


def test_slam_closes_and_survives_guidance(guided_static):
    """Loop closure fires, measures the true period within 1%, and is never revoked
    while guidance runs — the barnacle must not destabilize its own odometry."""
    slam = guided_static["slam"]
    assert slam["closure"] is not None
    assert abs(slam["period_px"] - 2600.0) < 26.0, slam["period_px"]
    assert slam["n_revoked"] == 0, slam["revoke_log"]


# ------------------------------------------------- surfacing profiles (Phase 7 redux)
def test_undifferentiated_mask_breaks_dynamics(guided_all, guided_static):
    """One profile for all surfaces: the beacon's cells get pinned toward stale lap-1
    intensity, the tracker locks to its own frozen outputs, and phase correlation with
    the true blink collapses. The class-split mask keeps it locked at no fidelity
    cost — different surfaces relay different profile requirements."""
    assert min(guided_all["beacon_corr"][1:3]) < 0.3, guided_all["beacon_corr"]
    assert min(guided_static["beacon_corr"]) > 0.95, guided_static["beacon_corr"]
    assert abs(guided_all["static_rmse"][3] - guided_static["static_rmse"][3]) < 2.0


# --------------------------------------------------------------- tone alignment trade
def test_alignment_trades_regime_for_seams():
    """align='none' pulls masked cells toward lap-1 tone while unmasked cells keep the
    drifted regime — less tone drift in the output, more boundary seam energy.
    align='gainbias' is geometry-only: seams stay near baseline, tone drift passes
    through. Neither dominates; they are different profiles, not a bugfix."""
    raw = run_graybox(n_laps=3.0, cfg=GuidanceCfg(align="none"), seed=0)
    aligned = run_graybox(n_laps=3.0, cfg=GuidanceCfg(align="gainbias"), seed=0)
    assert raw["gain_err"][2] < aligned["gain_err"][2], \
        (raw["gain_err"], aligned["gain_err"])
    assert raw["seam"][2] > aligned["seam"][2], (raw["seam"], aligned["seam"])


def test_sampler_converges_to_render():
    """Sanity: with no guidance the Euler chain lands on the engine's render (the
    schedule's last step has dsig/sig = -1, an exact contraction)."""
    import numpy as np
    from graybox import DriftEngine, GrayWorld, sample_frame

    world = GrayWorld(seed=0)
    engine = DriftEngine(world, seed=1)
    rng = np.random.default_rng(2)
    y = engine.render(0.0, world)
    engine2 = DriftEngine(world, seed=1)
    x = sample_frame(engine2, world, 0.0, None, rng, model_noise=0.0)
    assert float(np.abs(x - y).max()) < 1e-6
