# level_viewer

![level_viewer_2025-01-07_08-41-02](https://github.com/user-attachments/assets/8cfb7cab-e8bc-4378-a3e8-c305d1bafc74)

`level_viewer` is a Rust + Bevy desktop tool for exploring and inspecting assets from **American McGee's Scrapland**.

It can discover game archives, browse packed content, load level/model data, visualize geometry and materials, inspect nodes/collision data, and export reconstructed data for external tools such as Blender.

## Highlights

- Loads Scrapland `.packed` archives and exposes them through an in-app browser.
- Parses core Scrapland formats (`SM3`, `CM3`, `DUM`, `AMC`, `EMI`).
- Renders level geometry, materials, lightmaps, and animated textures.
- Provides in-app debug UI (inspector windows, post-processing controls, node visualization, collision toggles).
- Includes `dump.zip` exporter and a `blender_import.py` helper pipeline.

## Requirements

- Rust toolchain (stable; edition `2024` is used by this crate).
- A local Scrapland installation containing `data*.packed` files.
- GPU/driver support suitable for Bevy 0.18 desktop rendering.

Optional (Blender workflow):

- Blender with Python support.
- Python dependencies listed in `pyproject.toml` if you run the helper outside Blender's embedded environment.

## Getting Started

### 1) Build

```bash
cargo build --release
```

### 2) Point the viewer to Scrapland data

The app resolves the game install path in this order:

1. `SCRAPLAND_DIR` environment variable.
2. Steam autodetection (`App ID 897610`).
3. Manual folder picker dialog.

Example (PowerShell):

```powershell
$env:SCRAPLAND_DIR = "C:\Games\Steam\steamapps\common\Scrapland"
```

### 3) Run

```bash
cargo run --release
```

You can optionally pass an initial packed-path argument (file or level directory), for example:

```bash
cargo run --release -- Levels/Outskirts
```

## Controls And UI

### Camera / navigation

- `W`: forward throttle
- `S`: reverse
- `A` / `D`: strafe
- Mouse move: yaw + pitch
- Right mouse: boost
- `Shift`: turn boost

### Global toggles

- `F1`: show/hide UI
- `F2`: enable/disable lightmaps
- `F3`: enable/disable wireframe
- `F4`: show/hide node gizmos
- `Esc`: quit

### Adjustment shortcuts

- `Numpad +` / `Numpad -`: increase/decrease lightmap exposure
- `Numpad *` / `Numpad /`: scale lightmap exposure
- `Numpad 1` / `Numpad 2`: adjust depth-of-field aperture

## Export Workflow

Use the in-app **Export** window (`Export!` button) after loading a level.

The exporter writes `dump.zip` in the current working directory and includes:

- `obj/`: per-object mesh buffers and metadata
- `mat/`: material JSON
- `tex/`: regular textures (PNG)
- `lightmaps/`: lightmap textures (PNG)

## Blender Import Helper

`blender_import.py` demonstrates importing `dump.zip` into Blender and recreating meshes/material assignments.

Current script behavior:

- Resets Blender scene to factory-empty at startup.
- Reads `dump.zip` and reconstructs meshes, UVs, and basic material data.
- Expects the zip path to be set in the script (`zip_path = ...`).

Typical usage:

1. Export from `level_viewer` to produce `dump.zip`.
2. Edit `zip_path` in `blender_import.py`.
3. Run script in Blender's scripting workspace.

## Development

Fast validation:

```bash
cargo fmt --all
cargo check
```

Linting:

```bash
cargo clippy --all-targets --all-features
```

Tests:

```bash
cargo test
```

Shader parser test requires `SHADER_FILE`:

```powershell
$env:SHADER_FILE='C:\path\to\shader.psh'; cargo test pixel_shader::test::test -- --nocapture
```

## Project Layout

- `src/main.rs`: app setup, Bevy systems, runtime UI, scene loading/rendering.
- `src/parser.rs`: binary structs/parsers for Scrapland file formats.
- `src/asset_loader.rs`: integration between packed assets and Bevy asset source.
- `src/materials.rs`: custom material/shader behavior.
- `src/export.rs`: `dump.zip` export pipeline.
- `src/pixel_shader.rs`: pixel shader parsing experiments/tests.
- `blender_import.py`: Blender-side import helper for exported data.

## Known Limitations

- Some parser and shader paths are still experimental.
- Blender helper script is functional but not a polished add-on.
- Animation tooling and full shader parity are still in progress.

## License

MIT. See `LICENSE`.
