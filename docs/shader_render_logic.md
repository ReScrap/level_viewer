# Shader Render Logic Matrix

This document captures reverse-engineered shader render behavior from Ghidra and how it is mapped into exporter JSON (`shaders.json.gz`) for Blender reconstruction.

Program analyzed: `/remaster_update/Scrap.exe [main]`

## Core Pattern

Most shader render functions follow this decision:

- If `R_PShaders` is disabled, pixel shader handle is null, or `R_PSMask` blocks the shader bit:
  - Use fixed-function fallback path (texture stages / combiners).
- Else:
  - Bind pixel shader.
  - Upload constants and bind extra maps as required.
  - Restore stage/shader state on cleanup (`param_2 & 2` path).

## EngineVars used by render logic

- Env mapping:
  - `R_EnvMapViewDep`, `R_EnvMapScale`, `R_EnvMapOffset`
  - `R_EnvBlend`
- Env bump:
  - `R_EnvBumpScale`, `R_EnvBumpBias`
- Clouds:
  - `R_CloudVel1x`, `R_CloudVel1y`, `R_CloudVel2x`, `R_CloudVel2y`
  - `R_CloudScale1`, `R_CloudScale2`
  - `R_CloudEmi`, `R_CloudR`, `R_CloudG`, `R_CloudB`, `R_CloudA`
- Glow flicker:
  - `R_GlowFlickTile`, `R_GlowFlickRot`, `R_GlowFlickBump`, `R_GlowFlickBRot`, `R_GlowFlickMod`

## Render Functions And Modes

Notes:

- Addresses below are render function entrypoints referenced by `ShaderTable` / `ShaderTableLmap`.
- Some names are inferred from shader table strings and matching `.psh` files.

| Shader mode (inferred) | Render function | Key behavior |
|---|---:|---|
| `Diffuse` | `0x00645020` | basic diffuse stage setup/fallback |
| `DiffuseNoLit` | `0x0064a9a0` | pixel-shader-only path with `R_PSMask & 0x20000000` gate |
| `EnvMap` | `0x00645390` | `SetupEnvMap`, optional PS, map slot 2 usage |
| `EnvMapLightmap` | `0x00645770` | env map + extra stage, stage 2 texture control |
| `MaskEnvmap` | `0x00645c20` | env blend branch (`R_EnvBlend`) + map slot 2 |
| `MaskEnvmapLightmap` | `0x00646020` | mask env + extra stage routing |
| `MaskEnvBump` | `0x006464f0` | env bump setup (`FUN_0064b4c0`) + bump/env maps |
| `MaskEnvBumpLightmap` | `0x00646aa0` | mask env bump + extra stage/lightmap path |
| `Glowmap` | `0x00647100` | glow over diffuse, optional PS constant/color helper |
| `GlowmapLightmap` | `0x00647400` | glow + lightmap path with helper `FUN_006450c0` fallback |
| `GlowmapMaskEnvmap` | `0x00647590` | glow + env map + env blend |
| `GlowmapMaskEnvBump` | `0x006482c0` | glow + env bump + env blend (multi-stage) |
| `Clouds` (`CloudTest`) | `0x00648b80` | multi-UV scroll layers (`FUN_0064bb70`) + cloud tint constants |
| `Electric` | `0x006498c0` | animated UV layers + color constants |
| `Fire` | `0x00649c20` | animated UV layers + color constants |
| `Glass` | `0x00649f30` | env map path with PS/fallback split |
| `Waves` | `0x0064a050` | bump/env helper + additional map transforms |
| `BloomFilter` | `0x0064a560` | post-process style constants/UV jitter |
| `BloomBlur` | `0x0064a6a0` | simple PS on/off gate |
| `BloomTarget` | `0x0064a700` | simple PS on/off gate |
| `MotionBlurAdd` | `0x0064a760` | simple PS on/off gate |
| `MotionBlurTarget` | `0x0064a7c0` | simple PS on/off gate |
| `BlurTarget` | `0x0064a820` | simple PS on/off gate |
| `DUDVFilter` | `0x0064a880` | simple PS on/off gate |
| `DUDVTarget` | `0x0064a8e0` | simple PS on/off gate |
| `RadialBlurTarget` | `0x0064a940` | simple PS on/off gate |

## PSMask Block Bits (mapped)

Mapped in exporter (`src/pixel_shader.rs`) via `shader_psmask_block_bit`:

- `EnvMap`: `0x400`
- `EnvMapLightmap`: `0x800`
- `MaskEnvmap`: `0x10`
- `MaskEnvmapLightmap`: `0x100`
- `MaskEnvBump`: `0x8`
- `MaskEnvBumpLightmap`: `0x80`
- `Glowmap`: `0x1000`
- `GlowmapMaskEnvmap`: `0x4000`
- `GlowmapMaskEnvBump`: `0x10000`
- `GlowmapMaskEnvmapLightmap`: `0x20000`
- `GlowmapMaskEnvBumpLightmap`: `0x100000`
- `Clouds`: `0x200`
- `GlowFlick`: `0x8000`
- `Electric`: `0x10000`
- `Fire`: `0x20000`
- `Glass`: `0x40000`
- `Waves`: `0x80000`
- `BloomFilter`: `0x100000`
- `BloomBlur`: `0x200000`
- `BloomTarget`: `0x400000`
- `MotionBlurAdd`: `0x800000`
- `MotionBlurTarget`: `0x1000000`
- `BlurTarget`: `0x2000000`
- `DUDVFilter`: `0x4000000`
- `DUDVTarget`: `0x8000000`
- `RadialBlurTarget`: `0x10000000`
- `DiffuseNoLit`: `0x20000000`

## Exporter/Importer Mapping Notes

- Exporter now serializes per-material:
  - `engine_vars` snapshot (from `Level.config`)
  - `render_logic` (shader-kind flags + UV layer motions + tint/flick metadata)
- Blender importer uses this metadata to:
  - build reflection/env mapping transforms,
  - apply UV layer animation intent to texture nodes,
  - reconstruct Clouds blend/tint behavior,
  - keep expression-tree assembly as the base graph.

The node graph is an approximation of runtime D3D8 stage behavior; the metadata preserves the intent needed for high-fidelity iteration.
