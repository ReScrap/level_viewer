# Shader Assignment Findings

This note summarizes how Scrapland assigns pixel shaders to materials, based on `Scrap.exe` analysis in Ghidra (`/remaster_update/Scrap.exe [main]`).

## Key Functions

- `AssignShader` at `0x006a9050`
  - Chooses a shader mode name from material flags and maps.
  - Applies `(+shader:Name)` override from material name.
  - Resolves the mode through `FindShader`.
- `FindShader` at `0x006a8ff0`
  - Case-insensitive match against `ShaderTable[i].name`.
- `CallShaderRender` at `0x0064b860`
  - Dispatches `ShaderTable` or `ShaderTableLmap` render function.
- `SetupMaterial` at `0x0064d4a0`
  - Calls `CallShaderRender(mode, mat, pass)`.
  - Uses `mode + 0x100` for lightmap pass variants.

## Shader Tables

- `ShaderTable` is at `0x00831e50` (`ShaderDef[26]`).
- `ShaderTableLmap` is at `0x00832150` (`ShaderDef[27]`).
- The shader source path pointer array starts at `Lightmap_Shader` (`0x00831dd0`) and contains `.psh` paths such as:
  - `bmp/lightmap.psh`
  - `bmp/envmap.psh`
  - `bmp/maskenvmap.psh`
  - `bmp/glowmapmaskenvmap.psh`
  - `bmp/cloudtest.psh`
  - post-process shaders (`Bloom*`, `MotionBlur*`, `DUDV*`, etc.)

The effective entry shape inferred from data and usage is:

- `index`
- `name`
- `shader_path` (nullable)
- `init_fn` (nullable)
- per-shader state bytes (`flags[4]`, `flags[5]` copied by `AssignShader`)
- `render_fn` (called by `CallShaderRender`)

## Material To Shader Decision (from `AssignShader`)

Inputs considered:

- map slot 3 (bump)
- map slot 4 (glow)
- material env-map flag (`params.has_env_map`)
- whether diffuse map is env-enabled (`IsEnvMap(map0)`)

Decision:

- no glow + no bump + env condition -> `MaskEnvmap`
- no glow + bump + env condition -> `MaskEnvBump`
- glow + no bump + env condition -> `GlowmapMaskEnvmap`
- glow + bump + env condition -> `GlowmapMaskEnvBump`
- glow without env condition -> `Glowmap`
- non-glow fallback -> `EnvMap` if env flag, else `Diffuse`

Special case:

- `DiffuseNoLit` overrides the above when:
  - `dif_alpha != 0`
  - `src_blend == D3DBLEND_ZERO`
  - `dst_blend == D3DBLEND_SRCCOLOR`
  - no z-bias attribute check failure (`atrib & 0x800` path)

Material-name override:

- `GetShaderOverride` extracts `(+shader:<name>)` from material name and replaces the auto-selected shader.

## Notes Relevant To Export/Import

- A material can have an auto shader or explicit `(+shader:...)` override.
- Many gameplay shaders map directly to local `shaders/*.psh` files in this repository.
- Some modes are fixed-function/no `.psh` payload in the table (`shader_path == null`), so exporter should keep assignment even when assembly is unavailable.
