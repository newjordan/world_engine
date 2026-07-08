# uv run --dev python examples/anchor4_probe.py --n-scenes 1 --seeds 1 --run-label scout --csv bench_out/anchor4/pilot.csv
"""
Phase 10b four-frame front-door anchoring probe.

This is a thin entrypoint over anchor_probe.py with Phase 10b defaults: temporal
micro-pose anchor arms, the ANCHOR4 condition source, and anchor4 output paths.
"""
import sys

sys.path.insert(0, "examples")

import anchor_probe  # noqa: E402


def _has_arg(flag):
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


def _default(flag, value):
    if not _has_arg(flag):
        sys.argv.extend([flag, value])


def main():
    _default("--condition-source", "docs/ANCHOR4_PLAN.md")
    _default("--arms", "revisit,atlas_nowb,anchor4_linear,anchor4_blend,anchor4_k4")
    _default("--csv", "bench_out/anchor4/anchor4.csv")
    anchor_probe.main()


if __name__ == "__main__":
    main()
