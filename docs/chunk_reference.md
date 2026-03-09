# Scrapland Chunk Reference

Binary chunks with magic signatures in Scrapland game files.

## Chunk Table

| Chunk | Magic | Version | Description | Can Appear Inside |
|-------|-------|---------|-------------|-------------------|
| **INI** | `INI\0` | - | Configuration text container | Node, SCN |
| **LFVF** | `LFVF` | 1 | Vertex buffer | MD3D, TriV104 |
| **MD3D** | `MD3D` | 1 | D3D mesh geometry | NodeData::D3DMesh |
| **SPR3** | `SPR3` | 1 | Billboard/sprite object | NodeData::Graphic3D |
| **SUEL** | `SUEL` | 1 | Ground object | NodeData::Ground |
| **CAM** | `CAM\0` | 1 | Camera node | NodeData::Camera |
| **LUZ** | `LUZ\0` | 1 | Light node | NodeData::Light |
| **PORT** | `PORT` | 1 | Portal linkage | NodeData::Portal |
| **MAP** | `MAP\0` | 2-3 | Texture map slot | MAT |
| **MAT** | `MAT\0` | 1-3 | Material definition | SCN, EMI |
| **SCN** | `SCN\0` | 1 | Scene graph container | SM3, CM3 |
| **EVA** | `EVA\0` | 1 | Vertex animation bank | NAM |
| **NAM** | `NAM\0` | 1 | Animation track | ANI |
| **NABK** | `NABK` | - | Animation data bank | ANI |
| **ANI** | `ANI\0` | 2 | Animation container | SCN |
| **SM3** | `SM3\0` | - | Static scene file | Data |
| **CM3** | `CM3\0` | - | Animated scene file | Data |
| **DUM** | `DUM\0` | 1 | Dummy locator list | Data |
| **QUAD** | `QUAD` | 1 | Quad collision node | AMC |
| **CMSH** | `CMSH` | 2 | Collision mesh | AMC |
| **AMC** | `AMC\0` | 100 | Collision database | Data |
| **TRI** | `TRI\0` | - | Renderable mesh | EMI |
| **EMI** | `EMI\0` | 103-105 | Material + mesh file | Data |

---

## Detailed Documentation

---

## INI

**Magic:** `INI\0`

**Purpose:** Serialized configuration text, used for node properties and scene metadata.

**Version:** None (variable)

**Key Fields:**
- `sections: Vec<IniSection>` - Counted list of text sections

**Contains:** `IniSection`

**Appears In:** Node, SCN

**Notes:** Format text is rebuilt line-by-line. The `data()` helper parses merged text into a `configparser::Ini`.

---

## LFVF

**Magic:** `LFVF`

**Version:** 1

**Purpose:** Versioned vertex buffer container.

**Key Fields:**
- `fmt_id: u32` - Format ID (0-17)
- `inner: Option<LFVFInner>` - Vertex data when fmt_id != 0

**Contains:** `LFVFInner` (optional)

**Appears In:** MD3D, TriV104

**Ghidra:** `read_LFVF` validates version, FVF format, and stride.

**Notes:** fmt_id 0 means empty/no vertices. Format IDs map to specific FVF flags and vertex sizes.

---

## MD3D

**Magic:** `MD3D`

**Version:** 1

**Purpose:** D3D mesh node geometry payload.

**Key Fields:**
- `name: PascalString` - Mesh name
- `tris: MD3D_Tris` - Triangle indices
- `verts: LFVF` - Vertex buffer
- `vert_orig: Table<2, u16>` - Original vertex remap table
- `tri_seg: Table<0x10, MD3D_TriSeg>` - Triangle segments
- `segments: Table<8, MD3D_Segment>` - Edge segments
- `vert_orig_pos: Table<0xc, [f32; 3]>` - Original vertex positions
- `tri_flags: Table<4, u32>` - Triangle flags
- `skin_ref: u32` - Skin reference
- `skin: Option<Table<0x10, MD3D_Skin>>` - Optional skinning data
- `mat_index: i32` - Material index
- `tex1_b1_flag_from_convert_0x4: u32` - Conversion flag (bool)
- `tex1_b1_flag_from_convert_0x8: u32` - Conversion flag (bool)
- `subtree_bbox: [[f32; 3]; 2]` - Bounding box
- `local_bbox: [[f32; 3]; 2]` - Local bounds
- `bbox_center: [f32; 3]` - Bounding box center
- `child: Option<Box<MD3D>>` - Recursive child mesh

**Contains:** LFVF, Tables, MD3D_Skin (optional), MD3D (optional)

**Appears In:** NodeData::D3DMesh

**Ghidra:** `read_MD3D` validates table sizes and reads conversion flags via `read_int_nonzero` at offsets +0x50/+0x51.

---

## SPR3

**Magic:** `SPR3`

**Version:** 1

**Purpose:** Billboard/sprite-like 3D graphic object.

**Key Fields:**
- `pos: [f32; 3]` - Position
- `scale: [f32; 2]` - Scale
- `primary_map_name: PascalString` - Primary texture
- `secondary_map_name: PascalString` - Secondary texture
- `diffuse_mod: RGBA` - Diffuse modulation color

**Contains:** None

**Appears In:** NodeData::Graphic3D

**Ghidra:** `read_SPR3`

---

## SUEL

**Magic:** `SUEL`

**Version:** 1

**Purpose:** Ground object descriptor (scene-level, not collision).

**Key Fields:**
- `bbox: [[f32; 3]; 2]` - Bounding box
- `dims: [f32; 3]` - Dimensions
- `grid_size: [u32; 2]` - Grid dimensions
- `num_tris: u32` - Triangle count
- `collision_bbox: [[f32; 3]; 2]` - Collision bounds

**Contains:** None

**Appears In:** NodeData::Ground

**Ghidra:** `read_SUEL`

---

## CAM

**Magic:** `CAM\0`

**Version:** 1

**Purpose:** Camera node payload.

**Key Fields:**
- `angles: [f32; 3]` - Rotation angles
- `origin: [f32; 3]` - Camera position
- `target: [f32; 3]` - Look-at target
- `clip: [f32; 2]` - Near/far clip planes
- `range: [f32; 2]` - View range
- `fov: f32` - Field of view
- `phys_aspect_ratio: f32` - Physical aspect ratio
- `view_aspect_ratio: f32` - View aspect ratio
- `mode: u32` - Camera mode

**Contains:** None

**Appears In:** NodeData::Camera

**Ghidra:** `read_CAM`

---

## LUZ

**Magic:** `LUZ\0`

**Version:** 1

**Purpose:** Light node payload.

**Key Fields:**
- `sector: u32` - Sector index
- `light_type: LightType` - Point/Spot/Directional
- `shadows: u8` - Shadow enable flag
- `pos: [f32; 3]` - Position
- `dir: [f32; 3]` - Direction
- `color: RGBA` - Light color
- `power: f32` - Intensity
- `attenuation: [f32; 2]` - Attenuation factors
- `hotspot: f32` - Spotlight hotspot angle
- `falloff: f32` - Spotlight falloff
- `mult: f32` - Multiplier
- `radiosity_coeff: f32` - Radiosity coefficient
- `active: bool` - Active flag

**Contains:** None

**Appears In:** NodeData::Light

**Ghidra:** `read_LUZ`

---

## PORT

**Magic:** `PORT`

**Version:** 1

**Purpose:** Portal linkage between scene nodes/sectors.

**Key Fields:**
- `side_node_indices: [i32; 2]` - Two side node indices
- `width: f32` - Portal width
- `height: f32` - Portal height

**Contains:** None

**Appears In:** NodeData::Portal

**Ghidra:** `read_PORT`

---

## MAP

**Magic:** `MAP\0`

**Version:** 2-3

**Purpose:** Texture map slot parameters.

**Key Fields:**
- `texture: PascalString` - Texture path
- `filter: u8` - Filter mode
- `max_anisotropy: u8` - Anisotropy level
- `is_square: u8` - Square texture flag
- `is_env: u8` - Environment map flag
- `tile: u8` - Tiling flags
- `mirror: u8` - Mirror flags
- `texture_type: TextureType` - Effect type (None, FxLava, FxScroll, FxNewsPanel)
- `displacement: [f32; 2]` - UV displacement
- `scale: [f32; 2]` - UV scale
- `quantity: f32` - Bump mapping scale
- `uv_matrix: Option<[f32; 2]>` - V3+ UV transform matrix
- `angle: Option<f32>` - V3+ rotation angle

**Contains:** None

**Appears In:** MAT

**Ghidra:** `read_MAP` accepts v2/v3 only; extra UV transform only for v3.

---

## MAT

**Magic:** `MAT\0`

**Version:** 1-3

**Purpose:** Full material definition.

**Key Fields:**
- `name: Option<PascalString>` - Material name (v2+)
- `ambient_override: RGBA` - Ambient override
- `diffuse_mod: RGBA` - Diffuse modifier
- `diffuse: RGBA` - Diffuse color
- `specular: RGBA` - Specular color
- `glow: RGBA` - Emissive/glow
- `spec_power: f32` - Specular power
- `spec_mult: f32` - Specular multiplier
- `mat_props: MatProps` - Material properties
- `maps: [Optional<MAP>; 5]` - 5 map slots (diffuse, metallic, env, bump, glow)

**Contains:** MAP (5 slots)

**Appears In:** SCN, EMI

**Ghidra:** `read_MAT` breaks after 3 maps when version < 3.

**Notes:** V1/V2 stores 3 maps on disk; V3 stores 5. Parser normalizes to 5 entries.

---

## SCN

**Magic:** `SCN\0`

**Version:** 1

**Purpose:** Main scene graph container for SM3/CM3.

**Key Fields:**
- `model_name: PascalString` - Model name
- `root_node: PascalString` - Root node name
- `node_props: Optional<INI>` - Node properties
- `ambient: LightColor` - Ambient light
- `background: LightColor` - Background light
- `bbox: [[f32; 3]; 2]` - Scene bounding box
- `mat: Vec<MAT>` - Materials
- `nodes: Vec<Node>` - Scene nodes
- `ani: Optional<ANI>` - Animation data

**Contains:** INI (optional), MAT, Node, ANI (optional)

**Appears In:** SM3, CM3

**Ghidra:** `read_SCN` enforces version 1, reads materials then marker then nodes then optional animation.

---

## EVA

**Magic:** `EVA\0`

**Version:** 1

**Purpose:** Vertex animation bank.

**Key Fields:**
- `num_verts: u32` - Vertex count
- `verts: Vec<Optional<VertexAnim>>` - Per-vertex animation data

**Contains:** VertexAnim

**Appears In:** NAM

**Ghidra:** `read_EVA`

---

## NAM

**Magic:** `NAM\0`

**Version:** 1

**Purpose:** One animated object track descriptor.

**Key Fields:**
- `start_frame: u32` - Start frame
- `frames: u32` - Frame count
- `cm3_flags: BTreeSet<AniTrackType>` - Active channel flags
- `opt_flags: u32` - Optimization flags (low bits only)
- `stm_flags: u32` - Streaming flags (low bits only)
- `tracks: Vec<BlockInfo>` - Track block metadata
- `eva: Option<EVA>` - Vertex animation data

**Contains:** EVA (optional)

**Appears In:** ANI

**Ghidra:** `read_NAM` asserts `(flags & 0xffffef60)==0`.

**Notes:** Channel types: Position, Rotation, FOV, Color, Intensity, Visibility, EVA.

---

## NABK

**Magic:** `NABK`

**Purpose:** Raw animation byte bank for ANI track data.

**Key Fields:**
- `data: Vec<u8>` - Raw bytes

**Contains:** None

**Appears In:** ANI

**Notes:** Data is accessed via NAM's BlockInfo metadata to decode into actual animation values.

---

## ANI

**Magic:** `ANI\0`

**Version:** 2

**Purpose:** Scene animation container.

**Key Fields:**
- `fps: f32` - Frames per second
- `last_frame: u32` - Last frame (stored first!)
- `first_frame: u32` - First frame (stored second!)
- `num_objects: u32` - Object count
- `track_map: Vec<Option<u8>>` - Node-to-track mapping
- `data: Vec<u8>` - Raw NABK animation data
- `tracks: Vec<NAM>` - Per-object tracks

**Contains:** NABK, NAM

**Appears In:** SCN

**Ghidra:** `read_ANI` enforces version 2 and reads NABK before NAM list.

**Notes:** Frame bounds order is last_frame, then first_frame in the file.

---

## SM3

**Magic:** `SM3\0`

**Purpose:** Static scene file (no animation).

**Key Fields:**
- `timestamp_magic: u32` - Must be 0x6515f8
- `dependency_timestamp_a: DateTime<Utc>` - Dependency timestamp A
- `dependency_timestamp_b: DateTime<Utc>` - Dependency timestamp B
- `scene: SCN` - Scene data

**Contains:** SCN

**Appears In:** Data

**Ghidra:** `read_SM3` checks magic and timestamps before `read_SCN(..., false)`.

---

## CM3

**Magic:** `CM3\0`

**Purpose:** Dynamic/animated scene file.

**Key Fields:** Same as SM3 (timestamp magic, dependencies, scene)

**Contains:** SCN

**Appears In:** Data

**Ghidra:** `read_CM3` mirrors SM3 flow but calls `read_SCN(..., true)` for animation support.

---

## DUM

**Magic:** `DUM\0`

**Version:** 1

**Purpose:** Dummy locator list file.

**Key Fields:**
- `num_dummies: u32` - Dummy count
- `dummies: Vec<Dummy>` - Dummy entries
- `end_marker: u32` - Must be 0

**Contains:** Dummy

**Appears In:** Data

**Ghidra:** `read_DUM` walks has-next chain and verifies final count.

---

## QUAD

**Magic:** `QUAD`

**Version:** 1

**Purpose:** Recursive quad collision subdivision node.

**Key Fields:**
- `mesh_id: u32` - Mesh identifier
- `triangle_indices: Table<2, u16>` - Triangle indices
- `base_height_range: [f32; 2]` - Base height range
- `active_height_range: [f32; 2]` - Active height range
- `child: Optional<Box<QUAD>>` - Child quad (recursive)

**Contains:** QUAD (optional)

**Appears In:** AMC

**Ghidra:** `read_QUAD`

**Notes:** Forms a quadtree structure for collision mesh queries.

---

## CMSH

**Magic:** `CMSH`

**Version:** 2

**Purpose:** Collision mesh block.

**Key Fields:**
- `collide_mesh_size: u32` - Must be 0x34
- `zone_name: PascalString` - Zone name
- `triangle_mesh_flags: u16` - Flags
- `sector: u16` - Sector index
- `mesh_unique_id: u16` - Unique ID
- `load_mesh_slot: u8` - Load slot
- `is_out_sector: u8` - Out sector flag
- `bbox: [[f32; 3]; 2` - Bounding box
- `verts: Table<0xc, [f32; 3]>` - Vertices
- `tris: Table<0x1c, CMSH_Tri>` - Triangles

**Contains:** CMSH_Tri

**Appears In:** AMC

**Ghidra:** `read_CMSH` enforces version 2, size 0x34.

---

## AMC

**Magic:** `AMC\0`

**Version:** 100

**Purpose:** Full collision database for a level.

**Key Fields:**
- `amc_version_code: u32` - Must be 0
- `collision_bbox: [[f32; 3]; 2]` - Global collision bounds
- `total_triangles: u32` - Total tri count
- `quad_grid_bbox: [[f32; 3]; 2]` - Quad grid bounds
- `quad_grid_center: [f32; 3]` - Grid center
- `cmsh: [CMSH; 2]` - Base collision meshes
- `sector_col: Vec<[CMSH; 2]>` - Per-sector collision
- `grid_size: [u32; 2]` - Grid dimensions
- `grid_scale: [f32; 2]` - Grid scale
- `quads: Vec<QUAD>` - Quad tree roots
- `final_quad: Optional<QUAD>` - Final quad
- `_empty_amc: EmptyAMC` - Trailing empty block marker

**Contains:** CMSH, QUAD, EmptyAMC

**Appears In:** Data

**Ghidra:** `read_AMC` enforces version 100/code 0, reads nested CMSH/QUAD, expects trailing empty AMC block.

---

## TRI

**Magic:** `TRI\0`

**Purpose:** One renderable mesh object in EMI.

**Key Fields:**
- `flags: u32` - Mesh flags
- `name: PascalString` - Mesh name
- `zone_index: u32` - Zone index
- `data: TriV104` - Version-dependent payload

**Contains:** TriV104

**Appears In:** EMI

**Ghidra:** `read_EMI` parses TRI inline with version-105 zone-name handling.

---

## EMI

**Magic:** `EMI\0`

**Version:** 103-105

**Purpose:** Material and render mesh file.

**Key Fields:**
- `num_materials: u32` - Material count
- `materials: Vec<(u32, MAT)>` - Keyed materials
- `maps: Vec<EMI_Textures>` - Texture key table (sentinel-terminated)
- `num_objs: u32` - Object/triangle count
- `tri: Vec<TRI>` - Renderable meshes

**Contains:** MAT, TRI, EMI_Textures

**Appears In:** Data

**Ghidra:** `read_EMI`

**Notes:** Texture list terminates with key == 0 sentinel.

---

## NodeData (Tagged Enum)

**No Magic** - uses tag values instead

**Purpose:** Tagged object payloads attached to scene nodes.

**Variants:**
| Tag | Variant | Contains |
|-----|---------|----------|
| 0x0 | Dummy | - |
| 0xa1000001 | TriangleMesh | - |
| 0xa1000002 | D3DMesh | MD3D |
| 0xa2000004 | Camera | CAM |
| 0xa3000008 | Light | LUZ |
| 0xa4000010 | Ground | SUEL |
| 0xa5000020 | SistPart | - |
| 0xa6000040 | Graphic3D | SPR3 |
| 0xa6000080 | Flare | - |
| 0xa7000100 | Portal | PORT |

**Appears In:** Node

**Ghidra:** Same constants dispatched in `read_SCN` when creating node objects.

---

## Ghidra Function Reference

| Chunk | Function | Address |
|-------|----------|---------|
| LFVF | read_LFVF | 0x00641620 |
| MD3D | read_MD3D | 0x006b14f0 |
| MAT | read_MAT | - |
| MAP | read_MAP | - |
| ANI | read_ANI | - |
| NAM | read_NAM | - |
| EVA | read_EVA | - |
| AMC | read_AMC | - |
| CMSH | read_CMSH | - |
| QUAD | read_QUAD | - |
| DUM | read_DUM | 0x0066af30 |
| SM3 | read_SM3 | - |
| CM3 | read_CM3 | - |
| EMI | read_EMI | 0x0068dbc0 |
| SCN | read_SCN | 0x00650220 |
| SPR3 | read_SPR3 | - |
| SUEL | read_SUEL | - |
| CAM | read_CAM | - |
| LUZ | read_LUZ | - |
| PORT | read_PORT | - |
