# `src/parser.rs` Structure Reference

This document describes every Rust data structure declared in `src/parser.rs`.

Coverage check: 66/66 `struct`/`enum` declarations are documented here (including private/internal helpers like `VertexAnim`, `AniStreamHeader`, and `EmptyAMC`).

It combines:
- parser behavior from `binrw` attributes in `src/parser.rs`
- cross-checks against Ghidra (`/remaster_update/Scrap.exe [main]`) decompilation of the engine loaders (`read_SCN`, `read_MD3D`, `read_LFVF`, `read_MAT`, `read_MAP`, `read_ANI`, `read_NAM`, `read_EVA`, `read_AMC`, `read_CMSH`, `read_QUAD`, `read_DUM`, `read_SM3`, `read_CM3`, `read_EMI`, `read_SPR3`, `read_SUEL`, `read_CAM`, `read_LUZ`, `read_PORT`)

Notes:
- "Block" means a tagged binary chunk with a 4-byte magic and trailing size bookkeeping handled by `compute_size`.
- Many assertions in parser structs mirror runtime checks visible in Ghidra.

## Core helper structures

### `PackedEntry` (struct)
- Purpose: Entry metadata in a `.packed` archive index.
- Fields: `path` (`PascalString`), `size` (`u32`), `offset` (`u32`).

### `PackedHeader` (struct, block magic `BFPK`)
- Purpose: Header for packed archives.
- Invariants: `version == 0`.
- Payload: `files: Vec<PackedEntry>` counted by `num_files`.

### `Table<SIZE, T>` (generic struct)
- Purpose: Common counted table format used across blocks.
- Layout: `num_entries`, `entry_size`, then `data`.
- Invariant: `entry_size == SIZE` at read time.

### `Optional<T>` (generic struct)
- Purpose: On-disk optional value wrapper.
- Layout: `has_value: u32` then optional `value`.
- Behavior: serialized as plain value/`null` through `Serialize` impl.

### `PascalString` (struct)
- Purpose: Scrapland string format (`u32 length` + bytes, null-terminated content allowed).
- Behavior: decoder strips trailing null bytes; encoder always writes a final null byte.

### `IniSection` (struct)
- Purpose: One INI text block made of counted `PascalString` lines.

### `INI` (struct, block magic `INI\0`)
- Purpose: Serialized configuration text container.
- Invariants: stores block size and section count; format text is rebuilt line-by-line.
- Helpers: `data()` parses merged text into `configparser::Ini`.

### `RGBA` (struct)
- Purpose: 8-bit RGBA color.
- Helper: `as_array()` returns normalized `[f32; 4]`.

### `TexCoords` (tuple struct)
- Purpose: Texture coordinates with runtime dimension count (1-4 components per channel).

### `Vertex` (struct)
- Purpose: Decoded vertex payload driven by `FVF` flags.
- Always present: `xyz`.
- Optional fields by FVF bits: `normal`, `point_size`, `diffuse`, `specular`, `tex_0..tex_7`.

### `Pos` (enum, 3-bit)
- Purpose: Position encoding class inside `FVF` (`XYZ`, `XYZRHW`, `XYZB1..XYZB5`).

### `FVF` (bitfield struct, 32-bit)
- Purpose: Direct3D fixed-function vertex declaration mirror.
- Fields: `pos`, toggles (`normal`, `diffuse`, etc.), `tex_count`, per-stage tex dimension descriptors.
- Helper: `tex_dims()` maps encoded dimension to coordinate count.

## Vertex buffers and mesh payload

### `LFVFInner` (struct)
- Purpose: Core vertex buffer blob inside `LFVF` when format is non-empty.
- Invariants:
  - `vert_fmt` must match format implied by `fmt_id`.
  - `vert_size` must match format-specific expected stride.
- Payload: counted `Vec<Vertex>`.

### `LFVF` (struct, block magic `LFVF`)
- Purpose: Versioned vertex list container.
- Invariants: `version == 1`, `fmt_id in 0..=0x11`.
- Payload: optional `LFVFInner` (`fmt_id == 0` means empty/no vertices).
- Ghidra cross-check: `read_LFVF` validates version, `get_fvf(fmt_id)`, and stride (`vert_fmt_size`).

### `MD3D_Tris` (struct)
- Purpose: Triangle index list (`Vec<[u16; 3]>`) for `MD3D`.
- Invariant: `tri_size == 6`.

### `MD3D_TriSeg` (struct)
- Purpose: Triangle segmentation plane info.
- Fields: `plane_distance_xor`, `normal`.

### `MD3D_Segment` (struct)
- Purpose: Connectivity/edge segment metadata.
- Fields: two triangle indices and two vertex indices.

### `MD3D_Skin` (struct)
- Purpose: Skinning influence payload (up to 3 bones per vertex entry).

### `MD3D` (struct, block magic `MD3D`)
- Purpose: D3D mesh node geometry payload.
- Invariants: `version == 1`.
- Key fields:
  - `name`, `tris`, `verts`, original-vertex remap tables, segment tables, tri flags.
  - `skin_ref` + optional inlined `skin` table.
  - material reference (`mat_index`) and recursive `child` mesh.
  - bounds: `subtree_bbox`, `local_bbox`, `bbox_center`.
  - `tex1_b1_flag_from_convert_0x4` and `tex1_b1_flag_from_convert_0x8` are booleans read as nonzero ints.
- Ghidra cross-check: `read_MD3D` reads these two flags via `read_int_nonzero` at object offsets `+0x50/+0x51` and validates vertex/triangle side-table lengths.

## Scene graph object payloads

### `NodeData` (enum)
- Purpose: Tagged object payload attached to a `Node`.
- Tag values map to runtime object classes:
  - `Dummy` (`0x0`)
  - `TriangleMesh` (`0xa1000001`)
  - `D3DMesh(Box<MD3D>)` (`0xa1000002`)
  - `Camera(CAM)` (`0xa2000004`)
  - `Light(LUZ)` (`0xa3000008`)
  - `Ground(SUEL)` (`0xa4000010`)
  - `SistPart` (`0xa5000020`)
  - `Graphic3D(SPR3)` (`0xa6000040`)
  - `Flare` (`0xa6000080`)
  - `Portal(PORT)` (`0xa7000100`)
- Ghidra cross-check: same constants dispatched in `read_SCN` when creating node objects.

### `SPR3` (struct, block magic `SPR3`)
- Purpose: Billboard/sprite-like 3D graphic object data.
- Invariants: `version == 1`.
- Fields: sprite position/scale, primary and secondary map names, diffuse modulation color.

### `SUEL` (struct, block magic `SUEL`)
- Purpose: Ground object info (scene-level ground descriptor, not full AMC collision tree).
- Invariants: `version == 1`.
- Fields: bbox, dimensions, grid size, triangle count, collision bbox.

### `CAM` (struct, block magic `CAM\0`)
- Purpose: Camera node payload.
- Invariants: `version == 1`.
- Fields: angles, origin, target, clip planes, range, FOV, physical/view aspect, mode.

### `LightType` (enum, `u32`)
- Purpose: Light class discriminator.
- Values: `Point=5000`, `Spot=5001`, `Directional=5002`.

### `LUZ` (struct, block magic `LUZ\0`)
- Purpose: Light node payload.
- Invariants: `version == 1`.
- Fields: sector, type, shadow toggle, transform/color/attenuation terms, `active` flag.

### `PORT` (struct, block magic `PORT`)
- Purpose: Portal linkage between scene nodes/sectors.
- Invariants: `version == 1`.
- Fields: two side node indices, width, height.

### `NodeFlags` (enum bit positions)
- Purpose: Bit semantics for `Node.flags` (visibility, collision, lighting, bone/effect tags, etc.).
- Encoded as bit indices in `u32` via `parse_node_flags`/`encode_node_flags`.

### `Node` (struct)
- Purpose: Scene graph node record in `SCN`.
- Fields: indices/refs, `flags`, animation mask, naming (`name`, `parent`), transform data, optional `info` (`INI`) and optional typed `content` (`NodeData`).
- Ghidra cross-check: field order matches `read_SCN` node loop (indices -> flags -> names -> transforms -> INI -> object data).

## Materials and maps

### `MAP` (struct, block magic `MAP\0`)
- Purpose: Texture map slot parameters.
- Invariants: `version in 2..=3`.
- Fields: texture name, filtering flags, env/tile/mirror flags, `texture_type`, UV displacement/scale, bump quantity, optional v3 UV matrix + angle.
- Ghidra cross-check: `read_MAP` accepts only v2/v3 and reads extra UV transform only for v3.

### `TextureType` (enum)
- Purpose: Known map-effect modes.
- Values: `None`, `FxLava`, `FxScroll`, `FxNewsPanel`, or passthrough `Unknown(u8)`.

### `BlendMode` (enum, `u32`)
- Purpose: D3D blend factors for material source/destination blend fields.

### `CmpFunc` (enum, `u32`)
- Purpose: Z/alpha compare function enum used in material props.

### `MatPropAttrib` (enum bit positions)
- Purpose: Material attribute bits (collision, fog, shareable, alpha test, shader, z-bias, etc.).

### `MatProps` (struct)
- Purpose: Core material render-state group.
- Invariants: `sub_material == 0` in this serialized form.
- Fields: blend modes, two-sided, fog/zwrite/zfunc, and attribute bitset.

### `MAT` (struct, block magic `MAT\0`)
- Purpose: Full material definition.
- Invariants: `version in 1..=3`.
- Fields: optional name (v2+), color terms, spec terms, `mat_props`, five optional map slots (`diffuse`, `metallic`, `env`, `bump`, `glow`).
- Version-specific map count: v1/v2 store only the first 3 map slots on disk; v3 stores all 5. The parser normalizes to 5 entries in memory and fills missing trailing slots as `None`.
- Ghidra cross-check: `read_MAT` breaks after 3 maps when `version < 3`.

### `LightColor` (struct)
- Purpose: Pair of color and scalar intensity, used by scene ambient/background.

## Scene container and animation payload

### `SCN` (struct, block magic `SCN\0`)
- Purpose: Main scene graph payload for `SM3`/`CM3`.
- Invariants: `version == 1`; node section marker must be `1`.
- Fields: model/root names, node/user INI blocks, ambient/background light, scene bbox, materials, nodes, optional animation (`ANI`).
- Ghidra cross-check: `read_SCN` enforces version 1, reads materials then marker `TRUE` then nodes then optional animation file.

### `VertexAnim` (private struct)
- Purpose: Per-vertex animated triangle references in EVA data.

### `EVA` (struct, block magic `EVA\0`)
- Purpose: Vertex animation bank.
- Invariants: `version == 1`.
- Fields: counted optional `VertexAnim` entries.

### `AniTrackType` (enum bit positions)
- Purpose: Animation channel mask bits (`Position`, `Rotation`, `FOV`, `Color`, `Intensity`, `Visibility`, `EVA`).
- Helpers: `size()` per sample, `mask()` bit flag.

### `BlockInfo` (struct)
- Purpose: Decoded metadata for each animation data block in `NAM`.
- Fields: block byte size, element size, stream/optimized flags, track type.

### `AnimFrame` (struct)
- Purpose: Single-frame decoded view composed from `AnimTracks`.

### `AnimTracks` (struct)
- Purpose: Per-channel decoded animation arrays (`Vec`s per optional channel).
- Helper: `get_frame(frame)` merges available channels into one `AnimFrame`.

### `AniStreamHeader` (private struct)
- Purpose: Header prefix used by optimized+streamed ANI blocks.
- Fields: block `size`, `start_frame`, `num_frames`.

### `NAM` (struct, block magic `NAM\0`)
- Purpose: One animated object track descriptor.
- Invariants:
  - `version == 1`
  - `cm3_flags` must fit supported mask
  - `opt_flags` and `stm_flags` are constrained to supported low bits only (`& 0xfff8 == 0`)
- Fields: frame range, active channel flags, parsed `tracks`, optional `EVA`.
- Ghidra cross-check: `read_NAM` asserts `(flags & 0xffffef60)==0` and validates opt/stream masks without requiring `0x8000` in file data.

### `NABK` (struct, block magic `NABK`)
- Purpose: Raw animation byte bank used by `ANI` track blocks.

### `ANI` (struct, block magic `ANI\0`)
- Purpose: Scene animation container.
- Invariants: `version == 2`.
- Fields: fps, frame bounds, object/node counts, node-to-track map, raw `NABK` data bytes, per-object `NAM` tracks.
- Helper: `get_track(index)` decodes channel payloads from `NABK` using per-block metadata.
- Ghidra cross-check: `read_ANI` enforces version 2 and reads nested `NABK` before `NAM` list.

## Top-level scene files

### `SM3` (struct)
- Purpose: Static scene file (`SM3\0` outer tag handled by `Data`).
- Fields: timestamp magic (`0x6515f8`), two dependency timestamps, embedded `SCN`.
- Ghidra cross-check: `read_SM3` checks same magic and timestamp freshness before `read_SCN(..., false)`.

### `CM3` (struct)
- Purpose: Dynamic/animated scene variant (`CM3\0` outer tag handled by `Data`).
- Fields: same layout as `SM3`.
- Ghidra cross-check: `read_CM3` mirrors `SM3` flow but calls `read_SCN(..., true)`.

## Dummies and collision structures

### `Dummy` (struct)
- Purpose: One dummy locator entry (name + transform + optional INI).
- Field note: `has_next` is stored in file but list structure is represented as `Vec<Dummy>` in parser.

### `DUM` (struct)
- Purpose: Dummy list file.
- Invariants: `version == 1`, trailing `end_marker == 0`.
- Fields: counted `dummies`.
- Ghidra cross-check: `read_DUM` decrements expected count while walking has-next chain and verifies final count hits zero.

### `QUAD` (struct, block magic `QUAD`)
- Purpose: Recursive quad collision subdivision node.
- Invariants: `version == 1`.
- Fields: mesh id, triangle index table, base/active height ranges, optional child quad.

### `CMSH_Tri` (struct)
- Purpose: One collision triangle entry in `CMSH`.
- Fields: cached stamp, normal, plane distance, vertex indices, collision flags.

### `CMSH` (struct, block magic `CMSH`)
- Purpose: Collision mesh block.
- Invariants: `version == 2`, `collide_mesh_size == 0x34`.
- Fields: zone/sector metadata, mesh identifiers, bbox, vertex and triangle tables.
- Ghidra cross-check: `read_CMSH` enforces size/version and validates zone-to-sector relationship.

### `EmptyAMC` (private struct, block magic `AMC\0`)
- Purpose: Terminal empty AMC block with `size == 0` appended after main `AMC` body.

### `AMC` (struct)
- Purpose: Full collision database for a level.
- Invariants: `version == 100`, `amc_version_code == 0` (code comment notes OG game may use 1).
- Fields:
  - global collision bbox and totals
  - base `cmsh` pair + per-sector collision mesh pairs
  - grid parameters
  - quadtree roots (`quads`) and optional `final_quad`
  - trailing empty `AMC` block marker (`_empty_amc`)
- Ghidra cross-check: `read_AMC` enforces version 100/version_code 0, then reads all nested `CMSH`/`QUAD` sections and expects a second empty `AMC` block.

## EMI geometry structures

### `TriV104` (struct)
- Purpose: TRI payload used inside `EMI` with version-aware fields.
- Fields: optional zone name for version >= `0x69` (105), material/map keys, triangle indices, geometry `LFVF`, lightmap `LFVF`.

### `TRI` (struct, block magic `TRI\0`)
- Purpose: One renderable mesh object record in `EMI`.
- Fields: flags, name, zone index, and version-dependent `TriV104` payload.
- Ghidra cross-check: in `read_EMI`, TRI is parsed inline (`read_first_block("TRI")`) with the same ordering and version-105 zone-name handling.

### `EMI_Textures` (struct)
- Purpose: Linked texture key table entry.
- Layout: `key` + optional tuple `(path1, u32, path2)`; list terminates with `key == 0` sentinel.

### `EMI` (struct)
- Purpose: Material + render mesh file.
- Invariants: `version in 103..=105`.
- Fields: keyed material list, sentinel-terminated texture map table, counted `TRI` list.
- Writer helpers preserve original ordering and sentinel termination.

## Top-level data dispatch and level aggregate

### `Data` (enum)
- Purpose: File-type discriminator for parsed binary payloads.
- Tags: `SM3`, `CM3`, `DUM`, `AMC`, `EMI`.
- Helper: `dependencies()` extracts texture dependencies where meaningful.

### `Level` (struct)
- Purpose: High-level aggregate for a complete level directory (`map/` assets + configs).
- Fields: INI configs, core parsed files (`emi`, `sm3`, `dum`, `amc`), source path, resolved dependency map.

### `ParsedData` (enum)
- Purpose: Returned value for `MultiPackFS::parse_file`; either full `Level` directory parse or one `Data` file.

## Virtual filesystem helper module (`multi_pack_fs`)

### `Entry` (struct)
- Purpose: Directory listing entry (path, size, is_file).

### `MultiPackFS` (struct)
- Purpose: User-facing navigator over aggregated `.packed` files (`MultiPack` + current virtual path state).
- Provides: path navigation, listing, dependency resolution, file open/parse helpers.

## Test structure

### `Test` (struct, block magic `TEST`)
- Purpose: Local parser test fixture for roundtrip/size computation experiments.
- Fields: computed `size`, count `n`, and vector of `[f32; 3]`.

## Ghidra corroboration summary

The following parser invariants are directly mirrored in game loader assertions/decompilation:
- `LFVF`: version 1, `fmt_id` bounds, FVF and stride checks (`read_LFVF`, `0x00641620`).
- `MAP`/`MAT`: map/material version windows and map-slot handling (`read_MAP`, `read_MAT`).
- `SCN`/`NodeData`: node/object type tags and read order (`read_SCN`, `0x00650220`).
- `MD3D`: name check against node, table size consistency, two conversion booleans at `+0x50/+0x51` (`read_MD3D`, `0x006b14f0`).
- `ANI`/`NAM`/`EVA`: version checks, supported flag masks, optional EVA branch (`read_ANI`, `read_NAM`, `read_EVA`).
- `DUM`: version 1 and has-next chain semantics (`read_DUM`, `0x0066af30`).
- `CMSH`/`AMC`/`QUAD`: collision versions, sizes, recursive quad read, trailing empty `AMC` block (`read_CMSH`, `read_AMC`, `read_QUAD`).
- `EMI`/`TRI`: `EMI` versions 103-105 and inline `TRI` parsing including v105 zone-name behavior (`read_EMI`, `0x0068dbc0`).
- `SM3`/`CM3`: timestamp magic `0x6515f8` and `SCN` embedding (`read_SM3`, `read_CM3`).

Recent alignment updates based on Ghidra:
- `MAT` map serialization now follows engine behavior: 3 maps for v1/v2, 5 for v3.
- `NAM.opt_flags` now matches file semantics from engine read/write paths (supported low bits only; no forced on-disk `0x8000`).
