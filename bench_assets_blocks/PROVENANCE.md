# bench_assets_blocks — provenance

Procedurally generated classical-architecture benchmark scenes rendered from
**LegoGen** for the Overworld surreal-primitives permanence lane
(`docs/SURREAL_PRIMITIVES_PLAN.md`).

- **Primary set (this dir, `bench_assets_blocks/`)**: surreal **B/W** — bright
  white temple geometry with black wireframe edges on a near-black background,
  one distinctive asymmetrical mid-gray landmark per scene.
- **Comparison set (`../bench_assets_blocks_sunset/`)**: same geometry rendered
  in LegoGen's default *"Greek Memphis at sunset"* viewport look (original
  material grays, dark-blue → sunset-orange gradient sky, gold landmark accent).

All 8 PNGs are **1280×720, RGB (3-channel, no alpha), 8-bit** — identical format
to `bench_assets/seed_00.png`.

## Source

- Repo: `/home/frosty40/Blocks` (LegoGen)
- Commit: `6c147f55b65171e80ae8b16f7722e0c23e4b99c2` ("Add all project files for LegoGen block builder")
- Generator: `core/generators/classical.generate_classical_building` (same build
  path as `cli.py generate`).
- Renderer/driver: `scripts/bench_render.py` (added in-repo; off-screen PyVista
  plotter mirroring `gui/viewport.py` shading + a +Y-up 3/4 isometric camera).
- Render stack: Python 3.13, pyvista 0.48.4, vtk 9.6.2, trimesh 4.12.2,
  numpy 2.5.1, pillow 12.3.0. Off-screen (no DISPLAY needed; VTK 9.6 wheel
  falls back to EGL/OSMesa).

## Reproduce (exact)

```bash
cd /home/frosty40/Blocks
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install "numpy>=1.24.0" "trimesh>=4.0.0" \
    "pyvista>=0.43.0" "vtk>=9.2.0" python-dotenv

# primary surreal B/W set
./.venv/bin/python scripts/bench_render.py --style bw \
    --out /home/frosty40/overworld/bench_assets_blocks

# sunset comparison set
./.venv/bin/python scripts/bench_render.py --style sunset \
    --out /home/frosty40/overworld/bench_assets_blocks_sunset
```

Single scene: add `--only seed_02`. The scene table (order/preset/size/seed/
landmark) lives at the top of `scripts/bench_render.py` and is the single source
of truth. Rendering is deterministic (fixed seeds); re-running overwrites the
same filenames.

## Scenes

Every scene is a Greek temple (podium + colonnade + entablature + pediment +
stairs) plus exactly one asymmetrical landmark offset to one flank so a
loop-closure / permanence cue is visually obvious. `podium_height_actual =
height // 3`; collision/support disabled (matches `cli.py`).

| File | order | preset | W×H×D | seed | front_cols | apron | slope/entab/overhang | landmark |
|------|-------|--------|-------|------|-----------|-------|----------------------|----------|
| seed_00 | Doric | monumental | 22×6×16 | 7 | 8 | yes | 1.15 / 1.10 / 1.20 | tall thin **monolith**, far LEFT |
| seed_01 | Ionic | canonical | 16×5×12 | 21 | 6 | no | 1.0 / 1.0 / 1.0 | single cube **floating** above/RIGHT of roof |
| seed_02 | Random | compact | 12×4×10 | 42 | (auto) | no | 1.0 / 1.0 / 1.0 | lone slender **obelisk** (tall cylinder), RIGHT |
| seed_03 | Doric | canonical | 20×5×14 | 101 | 6 | yes | 1.0 / 1.0 / 1.0 | free-standing **trilithon portal** (2 posts + lintel), LEFT-FRONT |

Landmarks are generated programmatically relative to each temple's bounding box
(`add_landmark()` in the driver). In the B/W set the landmark is mid-gray
`(150,150,150)` against white geometry — "the single gray landmark" of the plan.
In the sunset set it takes the generator's gold access-marker accent
`(232,186,110)`.

### Style controls

- **B/W** (`--style bw`): structural block grays (generator range 112–232) are
  remapped toward white via `bright = 255 - (255 - luma)*0.22` (clamped 200–255),
  preserving relative material ordering so column/entablature/pediment form stays
  legible. Background `#050505`; black edges `#000000`.
- **Sunset** (`--style sunset`): original generator colors preserved; background
  `set_background("#18233a", top="#f5a35f")`; brown edges `#2b1b16`. Mirrors
  `gui/viewport.py` (`VIEWPORT_*` constants, ambient/diffuse/specular shading).
- Camera (both): explicit `[cam_pos, center, view_up=(0,1,0)]` 3/4 iso view;
  `view_dir=(1,0.62,1.15)`, distance `1.05×bbox_diag`, `zoom(1.3)`. (VTK's
  built-in `"iso"` treats +Z as up and tips this +Y-up geometry on its side, so
  the camera is set explicitly.)

## Verification (rendered output)

| set | file | size | mode | mean | min | max |
|-----|------|------|------|------|-----|-----|
| B/W | seed_00 | 1280×720 | RGB | 29.9 | 0 | 205 |
| B/W | seed_01 | 1280×720 | RGB | 25.1 | 0 | 203 |
| B/W | seed_02 | 1280×720 | RGB | 28.7 | 0 | 204 |
| B/W | seed_03 | 1280×720 | RGB | 40.2 | 0 | 204 |
| sunset | seed_00..03 | 1280×720 | RGB | ~106 | 22 | 245 |

B/W set = mostly dark background (~80% pixels < 20) with bright geometry
(not all-black, not all-white). Sunset set matches the viewport gradient.

The contact sheet (2 rows × 4 cols: top B/W, bottom sunset, labeled) lives at
`docs/permanence_assets/blocks_contact_sheet.png` for human review.

## ⚠ Loader note (important)

The Overworld asset loaders glob **`*.png`** in the asset dir, e.g.
`examples/play_server.py`: `sorted(glob.glob(os.path.join(a.assets, "*.png")))`
and `examples/demo_state_restore_ghosting.py` takes `[0]`. Only the four
`seed_*.png` scene seeds may live in this directory; review images belong in
`docs/permanence_assets/`.

`PROVENANCE.md` is not a PNG and is ignored by the loaders. The
`SURREAL_PRIMITIVES_PLAN.md` example uses a different dir name
(`bench_assets_surreal_bw`); rename/copy the seed PNGs there if you want to match
that command exactly.
