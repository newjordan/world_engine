# Surreal primitives lane - black/white permanence environments

**Status:** planned.
**Motivation:** user preference and diagnostic clarity.
**Boundary:** this is a new asset distribution and must not be compared to
`bench_assets` biome runs as if it were the same condition.

## Why This Lane Exists

The current benchmark scenes are richly textured natural/street environments.
They are useful because the model handles them, but they can hide what the memory
system is doing. A black/white primitive world is cleaner:

- circles, bars, rings, dots, and sparse landmarks make loop closure obvious;
- wrong-pose memory is easier to spot;
- object permanence can be judged visually without dense texture;
- surreal visual language fits the intended product/game direction.

The current model may not have much training coverage for such abstract worlds.
That is part of the condition, not a reason to avoid it. The lane should be
labeled `surreal_bw` or `ood_primitives`, not folded into the biome aggregate.

## Asset Direction

Create a small `bench_assets_surreal_bw/` set:

- `seed_00.png`: floating white circles on black, one asymmetrical anchor object;
- `seed_01.png`: white bars/rings plus a single gray landmark;
- `seed_02.png`: sparse dots at different scales, with one impossible geometry
  object.

Keep assets at `1280x720`, RGB PNG, repeated 4x by the existing loader.

## First Test

Run the existing best comparator and the current frontier anchor on this asset set:

```bash
uv run --dev python examples/anchor_probe.py \
  --condition-source docs/ANCHOR_FULL_PLAN.md \
  --assets bench_assets_surreal_bw \
  --arms revisit,atlas_nowb,anchor_full,anchor_full_far \
  --n-scenes 2 --seeds 3 \
  --run-label mechanics_proxy \
  --csv bench_out/surreal_bw/anchor_full_surreal_bw.csv
```

Use `mechanics_proxy` because the asset distribution differs from the controlled
biome condition.

## Readout

Primary readout is visual first:

- start;
- return / anchor;
- post16;
- post64;
- error heat;
- wrong-pose control frame.

Metrics still matter, but visual legibility is the purpose of this lane.

## Product Direction

If the model can carry the style, this becomes the preferred demo world:
minimal, surreal, readable, and honest about memory. If the current model cannot
hold the style, keep it as a training target and continue the product demo with
the biome scenes.
