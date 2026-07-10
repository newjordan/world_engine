# bench_assets_blockn — provenance (v2, BlockN semantic pipeline)

Second benchmark asset set for the Overworld surreal-primitives permanence lane,
generated with **BlockN**'s new semantic architecture pipeline (the
`feat/action-plan-simple-to-complex` push). Complements — does not replace —
the v1 classical-temple set in `bench_assets_blocks/`.

- **Primary set (this dir, `bench_assets_blockn/`)**: surreal **B/W** — near-white
  geometry with black edges on a near-black `#050505` background, one mid-gray
  `(150,150,150)` asymmetrical landmark per scene.
- **Comparison set (`../bench_assets_blockn_sunset/`)**: BlockN's per-request
  style palettes (timber / classical / industrial / contemporary) on the
  "Greek Memphis at sunset" gradient (`#18233a` -> `#f5a35f`), gold
  `(232,186,110)` landmark accent.

All 8 PNGs are **1280x720, RGB (3-channel, no alpha), 8-bit** — identical format
to `bench_assets/`. Contact sheet (not here — loaders glob `*.png`):
`docs/permanence_assets/blockn_contact_sheet.png`.

## Source

- Repo: `/home/frosty40/Blocks` (BlockN, formerly LegoGen)
- Branch/commit: `feat/action-plan-simple-to-complex` @
  `9370714054eb37c2ac78c1097a2872f18db4bfd3`
  ("feat: add BlockN architecture and CLI workflow")
- Pipeline: the documented front-door public API
  (`docs/ARCHITECTURE.md` "Public API Example"):
  `BuildRequest -> core.architecture.service.build() -> BuildResult.scene`.
  Every build passed BlockN's hard quality gates (`quality == "pass"`); the
  driver aborts on `fail`, matching the CLI default.
- Driver: `scripts/bench_render_blockn.py` (in-repo, uncommitted). It composes
  TWO front-door builds per scene — the main product plus a small landmark
  product translated to one flank at render time — then renders through the same
  off-screen PyVista camera/shading path as v1 (`scripts/bench_render.py`,
  which gained an optional `zoom` parameter; its default behavior is unchanged).
- Render stack: Python 3.13, pyvista 0.48.4, vtk 9.6.2, trimesh 4.12.2,
  numpy 2.5.1, pillow 12.3.0; off-screen EGL/OSMesa, no DISPLAY.
- Repo health at generation time: `python -m pytest tests/` -> **71 passed,
  32 subtests passed, 0 failed**.

## Reproduce (exact)

```bash
cd /home/frosty40/Blocks   # commit 9370714
# venv from v1 still satisfies requirements.txt (only header/py-version note changed)
./.venv/bin/python scripts/bench_render_blockn.py --style bw \
    --out /home/frosty40/overworld/bench_assets_blockn
./.venv/bin/python scripts/bench_render_blockn.py --style sunset \
    --out /home/frosty40/overworld/bench_assets_blockn_sunset
```

Single scene: `--only seed_02`. The `SCENES` table at the top of
`scripts/bench_render_blockn.py` is the single source of truth. Builds are
deterministic; the SHA-256 document fingerprints below identify the exact
semantic content (same for B/W and sunset — only render-time coloring differs).

## Scenes — BlockN simple-to-complex spectrum

Chosen to exercise the NEW BlockN capabilities across four different architects
and complexity levels (primitive -> assembly), including the new ARCH render
primitive; v1 already covers the legacy classical temples.

| File | product (architect) | variant / size | seed | style | blocks | landmark |
|---|---|---|---|---|---|---|
| seed_00 | bridge (circulation) | truss, width 14 | 3 | timber | 12+1 | lone **column** (h 6.5, seed 1), LEFT |
| seed_01 | arcade (classical) | monumental, 5 bays | 9 | classical | 13+3 | **floating slab** (3.2x0.3x3.2, seed 1) hovering at y=8, RIGHT |
| seed_02 | structural-grid (structure) | slab-frame, 4 bays x 3 levels | 17 | industrial | 136+14 | free-standing **stair-to-nowhere** (seed 1), RIGHT |
| seed_03 | spire-cluster (spire) | citadel | 29 | contemporary | 31+5 | single-bay **arch portal** (arcade bays=1, w5 h4.5, seed 1), LEFT |

All requests use `detail=presentation`. Landmark placement: offset from the main
product's bounding box by `sign*(span_x/2 + max(2, span_x*margin_frac) +
l_span_x/2)` in X (margin_frac 0.18 default; 0.30 seed_02; 0.34 seed_03),
`1.5*sign` in Z; seed_01's slab is additionally raised `+8.0` in Y (surreal
floating anchor). Per-scene camera zoom: 1.3 default, 1.15 seed_02, 1.2 seed_03.

Document fingerprints (SHA-256, `BuildDocument.fingerprint()`):

| scene | main | landmark |
|---|---|---|
| seed_00 | `e11f8c7c66ce...` | `4e816cc80baf...` |
| seed_01 | `d14806154a17...` | `f6ebba65c4b4...` |
| seed_02 | `1e36b1df1063...` | `3b78dc7b0686...` |
| seed_03 | `051e2e0a179b...` | `5fc11813f922...` |

Equivalent front-door CLI builds (geometry only; the driver does the same via
the service API, then merges + renders):

```bash
python -m blockn make bridge          --variant truss      --width 14 --seed 3  --style timber       --detail presentation
python -m blockn make column          --height 6.5                    --seed 1  --style timber       --detail presentation
python -m blockn make arcade          --variant monumental --bays 5   --seed 9  --style classical    --detail presentation
python -m blockn make slab            --width 3.2 --depth 3.2         --seed 1  --style classical    --detail presentation
python -m blockn make structural-grid --variant slab-frame --bays 4 --levels 3 --seed 17 --style industrial --detail presentation
python -m blockn make stair                                           --seed 1  --style industrial   --detail presentation
python -m blockn make spire-cluster   --variant citadel               --seed 29 --style contemporary --detail presentation
python -m blockn make arcade          --variant roman --bays 1 --width 5 --height 4.5 --seed 1 --style contemporary --detail presentation
```

### Style controls

- **B/W**: structural block colors luma-remapped toward white
  (`bright = 255 - (255-luma)*0.22`, clamped 200-255, preserving material
  ordering); landmark forced to gray `(150,150,150)`; background `#050505`,
  edges `#000000`.
- **Sunset**: BlockN palette colors kept as built; landmark forced to gold
  `(232,186,110)`; `set_background("#18233a", top="#f5a35f")`, edges `#2b1b16`;
  ambient/diffuse/specular shading mirrors `gui/viewport.py`.
- Camera: explicit +Y-up 3/4 iso (`view_dir=(1,0.62,1.15)`, distance
  `1.05*bbox_diag`, `view_up=(0,1,0)`), per-scene zoom as above.

## Verification (rendered output)

| set | file | size | mode | mean | min | max | dark(<20) |
|---|---|---|---|---|---|---|---|
| B/W | seed_00 | 1280x720 | RGB | 15.2 | 0 | 200 | 0.92 |
| B/W | seed_01 | 1280x720 | RGB | 18.6 | 0 | 222 | 0.90 |
| B/W | seed_02 | 1280x720 | RGB | 34.4 | 0 | 196 | 0.80 |
| B/W | seed_03 | 1280x720 | RGB | 26.3 | 0 | 195 | 0.86 |
| sunset | seed_00..03 | 1280x720 | RGB | 101-107 | 22 | 245 | 0.00 |

B/W = mostly dark with bright geometry (not all-black, not all-white); sunset
matches the viewport gradient.

## Loader note

This directory contains ONLY `seed_00..seed_03.png` plus this `PROVENANCE.md`
(non-PNG, ignored by the loaders). The contact sheet lives outside the probe
dir at `docs/permanence_assets/blockn_contact_sheet.png` because Overworld
loaders glob `*.png` in the assets dir.
