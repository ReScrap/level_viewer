#![allow(clippy::upper_case_acronyms, non_camel_case_types)]
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Debug,
    io::{BufReader, Read, Seek},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, Result};
use bilge::prelude::*;
use binrw::{args, helpers::until_exclusive, prelude::*};
use chrono::{DateTime, Utc};
use configparser::ini::Ini;
use enum_iterator::Sequence;
use indexmap::IndexMap;
use num_derive::ToPrimitive;
use num_traits::ToPrimitive;
use rhexdump::rhexdumps;
use serde::Serialize;
use vfs::VfsPath;
use walkdir::WalkDir;

type IniData = IndexMap<String, IndexMap<String, Option<String>>>;

#[binread]
#[derive(Serialize, Debug, Clone)]
pub(crate) struct PackedEntry {
    pub path: PascalString,
    pub size: u32,
    pub offset: u32,
}

#[binread]
#[br(magic = b"BFPK")]
#[derive(Serialize, Debug)]
pub(crate) struct PackedHeader {
    #[br(temp,assert(version==0))]
    pub version: u32,
    #[br(temp)]
    pub num_files: u32,
    #[br(count=num_files)]
    pub files: Vec<PackedEntry>,
}

#[binread]
#[derive(Serialize, Debug)]
#[br(import(msg: &'static str))]
pub(crate) struct Unparsed<const SIZE: u64> {
    #[br(count=SIZE, try_map=|data: Vec<u8>| Err(anyhow!("Unparsed data: {}\n{}", msg, rhexdumps!(data))))]
    data: (),
}
#[binread]
#[derive(Serialize, Debug)]
pub(crate) struct RawTable<const SIZE: u32> {
    num_entries: u32,
    #[br(assert(entry_size==SIZE))]
    entry_size: u32,
    #[br(count=num_entries, args {inner: args!{count: entry_size.try_into().unwrap()}})]
    data: Vec<Vec<u8>>,
}

#[binread]
#[derive(Serialize, Debug)]
pub(crate) struct Table<T: for<'a> BinRead<Args<'a> = ()> + 'static> {
    num_entries: u32,
    entry_size: u32,
    #[br(count=num_entries)]
    data: Vec<T>,
}

// impl<T: for<'a> BinRead<Args<'a> = ()>> Serialize for Table<T> where T: Serialize {
//     fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer {
//         self.data.serialize(serializer)
//     }
// }

#[binread]
pub(crate) struct Optional<T: for<'a> BinRead<Args<'a> = ()>> {
    #[br(temp)]
    has_value: u32,
    #[br(if(has_value!=0))]
    value: Option<T>,
}

impl<T> std::ops::Deref for Optional<T>
where
    T: for<'a> BinRead<Args<'a> = ()>,
{
    type Target = Option<T>;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T: for<'a> BinRead<Args<'a> = ()> + Debug> Debug for Optional<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value.fmt(f)
    }
}

// impl<T: for<'a> BinRead<Args<'a> = ()> + std::ops::Deref> std::ops::Deref for Optional<T> {
//     type Target = Option<T>;

//     fn deref(&self) -> &Self::Target {
//         &self.value
//     }
// }

impl<T: for<'a> BinRead<Args<'a> = ()> + Serialize> Serialize for Optional<T> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.value.serialize(serializer)
    }
}

#[binread]
#[derive(Serialize, Debug)]
pub(crate) struct Chunk {
    #[br(map=|c:[u8;4]| c.into_iter().map(|v| v as char).collect())]
    magic: Vec<char>,
    size: u32,
    #[br(temp,count=size)]
    _data: Vec<u8>,
}

#[binread]
#[derive(Clone)]
pub(crate) struct PascalString {
    #[br(temp)]
    length: u32,
    #[br(count=length, map=|bytes: Vec<u8>| {
        String::from_utf8_lossy(&bytes.iter().copied().take_while(|&v| v!=0).collect::<Vec<u8>>()).into_owned()
    })]
    pub string: String,
}

impl Serialize for PascalString {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.string.serialize(serializer)
    }
}

impl Debug for PascalString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.string.fmt(f)
    }
}

#[binread]
#[derive(Debug, Serialize)]
pub(crate) struct IniSection {
    #[br(temp)]
    num_lines: u32,
    #[br(count=num_lines)]
    pub sections: Vec<PascalString>,
}

#[binread]
#[br(magic = b"INI\0")]
#[derive(Debug)]
pub(crate) struct INI {
    #[br(temp)]
    _size: u32,
    #[br(temp)]
    num_sections: u32,
    #[br(count=num_sections)]
    pub sections: Vec<IniSection>,
}

impl Serialize for INI {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;
        let blocks: Vec<String> = self
            .sections
            .iter()
            .flat_map(|s| s.sections.iter())
            .map(|s| s.string.clone())
            .collect();
        Ini::new()
            .read(blocks.join("\n"))
            .map_err(Error::custom)?
            .serialize(serializer)
    }
}

#[binread]
#[derive(Debug, Serialize, Clone)]
pub(crate) struct RGBA {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[binread]
#[derive(Debug, Serialize, Clone)]
#[br(import(n_dims: usize))]
pub(crate) struct TexCoords(#[br(count=n_dims)] pub Vec<f32>);

#[binread]
#[derive(Debug, Serialize, Clone)]
#[br(import(vert_fmt: FVF))]
// https://github.com/elishacloud/dxwrapper/blob/23ffb74c4c93c4c760bb5f1de347a0b039897210/ddraw/IDirect3DDeviceX.cpp#L2642
pub(crate) struct Vertex {
    pub xyz: [f32; 3],
    // #[br(if(vert_fmt.pos()==Pos::XYZRHW))] // seems to be unused
    // rhw: Option<f32>,
    #[br(if(vert_fmt.normal()))]
    pub normal: Option<[f32; 3]>,
    #[br(if(vert_fmt.point_size()))]
    pub point_size: Option<[f32; 3]>,
    #[br(if(vert_fmt.diffuse()))]
    pub diffuse: Option<RGBA>,
    #[br(if(vert_fmt.specular()))]
    pub specular: Option<RGBA>,
    #[br(if(vert_fmt.tex_count().value()>=1), args (vert_fmt.tex_dims(0),))]
    pub tex_0: Option<TexCoords>,
    #[br(if(vert_fmt.tex_count().value()>=2), args (vert_fmt.tex_dims(1),))]
    pub tex_1: Option<TexCoords>,
    #[br(if(vert_fmt.tex_count().value()>=3), args (vert_fmt.tex_dims(2),))]
    pub tex_2: Option<TexCoords>,
    #[br(if(vert_fmt.tex_count().value()>=4), args (vert_fmt.tex_dims(3),))]
    pub tex_3: Option<TexCoords>,
    #[br(if(vert_fmt.tex_count().value()>=5), args (vert_fmt.tex_dims(4),))]
    pub tex_4: Option<TexCoords>,
    #[br(if(vert_fmt.tex_count().value()>=6), args (vert_fmt.tex_dims(5),))]
    pub tex_5: Option<TexCoords>,
    #[br(if(vert_fmt.tex_count().value()>=7), args (vert_fmt.tex_dims(6),))]
    pub tex_6: Option<TexCoords>,
    #[br(if(vert_fmt.tex_count().value()>=8), args (vert_fmt.tex_dims(7),))]
    pub tex_7: Option<TexCoords>,
}

#[bitsize(3)]
#[derive(Debug, Serialize, PartialEq, Eq, TryFromBits)]
pub(crate) enum Pos {
    XYZ,
    XYZRHW,
    XYZB1,
    XYZB2,
    XYZB3,
    XYZB4,
    XYZB5,
}

#[bitsize(32)]
#[derive(DebugBits, Serialize, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, TryFromBits)]
pub(crate) struct FVF {
    reserved: bool,
    pub pos: Pos,
    pub normal: bool,
    pub point_size: bool,
    pub diffuse: bool,
    pub specular: bool,
    pub tex_count: u4,
    pub tex: [u2; 8],
    rest: u4,
}

impl FVF {
    fn tex_dims(&self, tex: u8) -> usize {
        let tex = self.tex()[tex as usize].value();
        match tex {
            0 => 2,
            1 => 3,
            2 => 4,
            3 => 1,
            _ => unreachable!(),
        }
    }

    // fn num_w(&self) -> usize {
    //     use Pos::*;
    //     match self.pos() {
    //         XYZ | XYZRHW => 0,
    //         XYZB1 => 1,
    //         XYZB2 => 2,
    //         XYZB3 => 3,
    //         XYZB4 => 4,
    //         XYZB5 => 5,
    //     }
    // }
}

fn vertex_size_from_id(fmt_id: u32) -> Result<u32> {
    let fmt_size = match fmt_id {
        0 => 0x0,
        1 | 8 | 10 => 0x20,
        2 => 0x28,
        3 | 0xd => 0x1c,
        4 | 7 => 0x24,
        5 => 0x2c,
        6 => 0x34,
        0xb => 4,
        0xc => 0x18,
        0xe => 0x12,
        0xf | 0x10 => 0x16,
        0x11 => 0x1a,
        other => bail!("Invalid vertex format id: {other}"),
    };
    Ok(fmt_size)
}

fn vertex_format_from_id(fmt_id: u32, fmt: u32) -> Result<FVF> {
    let fvf = match fmt_id {
        0 => 0x0,
        1 => 0x112,
        2 => 0x212,
        3 => 0x1c2,
        4 => 0x116,
        5 => 0x252,
        6 => 0x352,
        7 => 0x152,
        8 => 0x1c4,
        10 => 0x242,
        other => bail!("Invalid vertex format id: {other}"),
    };
    if fvf != fmt {
        bail!("Vertex format mismatch: {fvf}!={fmt}");
    }
    FVF::try_from(fvf).map_err(|fvf| anyhow!("Invalid vertex format: {fvf:?}"))
}

#[binread]
#[br(import(fmt_id: u32))]
#[derive(Debug, Serialize, Clone)]
pub(crate) struct LFVFInner {
    #[br(try_map=|v:  u32| vertex_format_from_id(fmt_id,v))]
    vert_fmt: FVF,
    #[br(assert(vertex_size_from_id(fmt_id).ok()==Some(vert_size)))]
    vert_size: u32,
    num_verts: u32,
    #[br(count=num_verts, args {inner: (vert_fmt,)})]
    pub data: Vec<Vertex>,
}

#[binread]
#[br(magic = b"LFVF")]
#[derive(Debug, Serialize)]
pub(crate) struct LFVF {
    size: u32,
    #[br(assert(version==1,"invalid LFVF version"))]
    version: u32,
    #[br(assert((0..=0x11).contains(&fmt_id),"invalid LFVF format_id"))]
    fmt_id: u32,
    #[br(if(fmt_id!=0),args(fmt_id))]
    pub inner: Option<LFVFInner>,
}

#[binread]
#[derive(Debug, Serialize)]
pub(crate) struct MD3D_Tris {
    num_tris: u32,
    #[br(assert(tri_size==6,"Invalid MD3D tri size"))]
    tri_size: u32,
    #[br(count=num_tris)]
    pub tris: Vec<[u16; 3]>,
}

#[binread]
#[br(magic = b"MD3D")]
#[derive(Debug, Serialize)]
pub(crate) struct MD3D {
    size: u32,
    #[br(assert(version==1,"Invalid MD3D version"))]
    version: u32,
    pub name: PascalString,
    pub tris: MD3D_Tris,
    pub verts: LFVF,
    unk_table_1: RawTable<2>, // Vert orig
    pub unk_int_1: u32,
    unk_table_2: RawTable<0x10>,
    unk_table_3: RawTable<8>,
    unk_table_4: RawTable<0xc>,
    unk_table_5: RawTable<4>, // Tri Flags
    pub unk_int_2: u32,
    #[br(if(unk_int_2==0))]
    unk_table_6: Option<RawTable<0x10>>,
    unk_int_4: u32,
    unk_int_5: u32,
    unk_int_6: u32,
    #[br(count = 0x18)]
    unk_bytes_1: Vec<u8>,
    #[br(count = 0x18)]
    unk_bytes_2: Vec<u8>,
    #[br(count = 0xc)]
    unk_bytes_3: Vec<u8>,
    has_child: u32,
    #[br(if(has_child!=0))]
    pub child: Option<Box<MD3D>>,
}

#[binread]
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(crate) enum NodeData {
    #[br(magic = 0x0u32)]
    Dummy,
    #[br(magic = 0xa1_00_00_01_u32)]
    TriangleMesh(Unparsed<0x100>), // TODO: Empty or unused?
    #[br(magic = 0xa1_00_00_02_u32)]
    D3DMesh(Box<MD3D>),
    #[br(magic = 0xa2_00_00_04_u32)]
    Camera(CAM),
    #[br(magic = 0xa3_00_00_08_u32)]
    Light(LUZ),
    #[br(magic = 0xa4_00_00_10_u32)]
    Ground(SUEL),
    #[br(magic = 0xa5_00_00_20_u32)]
    SistPart,
    #[br(magic = 0xa6_00_00_40_u32)]
    Graphic3D(SPR3),
    #[br(magic = 0xa6_00_00_80_u32)]
    Flare,
    #[br(magic = 0xa7_00_01_00u32)]
    Portal(PORT),
}

#[binread]
#[br(magic = b"SPR3")]
#[derive(Debug, Serialize)]
pub(crate) struct SPR3 {
    size: u32,
    #[br(assert(version==1,"Invalid SPR3 version"))]
    version: u32,
    pos: [f32; 3],
    unk_1: [u8; 8],
    name_1: PascalString,
    name_2: PascalString,
    unk_2: u32,
}

#[binread]
#[br(magic = b"SUEL")]
#[derive(Debug, Serialize)]
pub(crate) struct SUEL {
    size: u32,
    #[br(assert(version==1,"Invalid SUEL version"))]
    version: u32,
    bbox: [[f32; 3]; 2],
    pos: [f32; 3],
    unk_3: [u8; 4],
    num_nodes: u32,
    unk_4: [u8; 4],
    bbox_2: [[f32; 3]; 2],
}

#[binread]
#[br(magic = b"CAM\0")]
#[derive(Debug, Serialize)]
pub(crate) struct CAM {
    size: u32,
    #[br(assert(version==1,"Invalid CAM version"))]
    version: u32,
    unk_1: [f32; 3],
    origin: [f32; 3],
    target: [f32; 3],
    unk_4: [u8; 4],
    unk_5: [u8; 4],
    unk_6: [u8; 4],
    unk_7: [u8; 4],
    unk_8: [u8; 4],
    unk_9: [u8; 4],
    unk_10: [u8; 4],
    unk_11: [u8; 4],
}

#[binread]
#[derive(Debug, Serialize)]
pub(crate) enum LightType {
    #[br(magic = 5000u32)]
    Point,
    #[br(magic = 5001u32)]
    Spot,
    #[br(magic = 5002u32)]
    Directional,
}

#[binread]
#[br(magic = b"LUZ\0")]
#[derive(Debug, Serialize)]
pub(crate) struct LUZ {
    size: u32,
    #[br(assert(version==1,"Invalid LUZ version"))]
    version: u32,
    unk_1: u32,
    light_type: LightType,
    unk_3: u8,
    pos: [f32; 3],
    rot: [f32; 3],
    col: u32,
    unk_6: [u8; 4],
    unk_7: [u8; 4],
    unk_8: [u8; 4],
    unk_9: [u8; 4],
    unk_10: [u8; 4],
    unk_11: [u8; 4],
    unk_12: [u8; 4],
    unk_13: u32,
}

#[binread]
#[br(magic = b"PORT")]
#[derive(Debug, Serialize)]
pub(crate) struct PORT {
    size: u32,
    #[br(assert(version==1,"Invalid PORT version"))]
    version: u32,
    width: u32,
    height: u32,
    sides: [u32; 2],
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Sequence, Serialize, ToPrimitive)]
#[repr(u8)]
pub enum NodeFlags {
    ROOT,
    GROUP,
    SELECT,
    HIDDEN,
    HVISIBLE,
    NO_CASTSHADOWS,
    NO_RCVSHADOWS,
    NO_RENDER,
    BOXMODE,
    COLLIDE = 0xc,
    NO_COLLIDE,
    NO_ANIMPOS = 0x10,
    NO_TRANS,
    SLERP2,
    EFFECT,
    BONE,
    BIPED,
    NO_TABNODOR,
    TWOSIDES = 0x18,
    RT_LIGHTING,
    RT_SHADOWS,
    NO_LIGHTMAP,
    NO_SECTOR,
    AREA_LIGHT,
}

fn parse_node_flags(flags: u32) -> BTreeSet<NodeFlags> {
    enum_iterator::all::<NodeFlags>()
        .filter_map(|flag| ((flags & (1 << flag.to_u8().unwrap_or(0xff))) != 0).then_some(flag))
        .collect()
}

#[binread]
#[derive(Debug, Serialize)]
pub(crate) struct Node {
    pub node_index: i32,
    pub unk_idx_1: i32,
    pub unk_idx_2: i32,
    #[br(map=parse_node_flags)]
    pub flags: BTreeSet<NodeFlags>,
    pub unk_f20_0x50: i32,
    pub name: PascalString,
    pub parent: PascalString,
    pub pos_offset: [f32; 3],
    pub rot: [f32; 4],
    pub scale: f32,
    pub transform: [[f32; 4]; 4], // 0x40 4x4 Matrix
    pub transform_inv: [[f32; 4]; 4], // 0x40 4x4 Matrix
    pub unk_rot: [f32; 4],
    pub axis_scale: [f32; 3],
    pub info: Optional<INI>,
    pub content: Optional<NodeData>,
}

#[binread]
#[br(magic = b"MAP\0")]
#[derive(Debug, Serialize, Clone)]
pub(crate) struct MAP {
    size: u32,
    #[br(assert((2..=3).contains(&version),"invalid MAP version"))]
    version: u32,
    pub texture: PascalString,
    unk_1: [u8; 7],
    unk_bbox: [[f32; 2]; 2],
    unk_2: f32,
    #[br(if(version==3))]
    unk_3: Option<[u8; 0xc]>,
}

#[binread]
#[derive(Debug, Serialize)]
pub(crate) struct Textures {
    pub diffuse: Optional<MAP>,
    pub metallic: Optional<MAP>,
    pub reflection: Optional<MAP>,
    pub bump: Optional<MAP>,
    pub glow: Optional<MAP>,
}

#[binread]
#[br(magic = b"MAT\0")]
#[derive(Debug, Serialize)]
pub(crate) struct MAT {
    size: u32,
    #[br(assert((1..=3).contains(&version),"invalid MAT version"))]
    version: u32,
    #[br(if(version>1))]
    pub name: Option<PascalString>,
    unk_f: [RGBA; 7],
    unk_data: [RGBA; 0x18 / 4],
    pub maps: Textures,
}

#[binread]
#[br(magic = b"SCN\0")]
#[derive(Debug, Serialize)]
pub(crate) struct SCN {
    // 0x650220
    size: u32,
    #[br(temp,assert(version==1))]
    version: u32,
    model_name: PascalString,
    node_name: PascalString,
    node_props: Optional<INI>,
    /*
    defaults:
    field5_0x14: 0xb0b0b (u32)
    field6_0x18: 1.0 (f32)
    field7_0x1c: 0 (u32)
    field8_0x20: 1.0 (f32)
    field9_0x24+: 0 (u32)  (0x18/4) read
    */
    unk_f_1: [f32; (8 + 8) / 4],
    unk_1: [f32; 0x18 / 4],
    unk_f_2: f32,
    user_props: Optional<INI>,
    #[br(temp)]
    num_materials: u32,
    #[br(count=num_materials)]
    pub mat: Vec<MAT>,
    #[br(temp,assert(unk_3==1))]
    unk_3: u32,
    #[br(temp)]
    num_nodes: u32,
    #[br(count = num_nodes)] // 32
    pub nodes: Vec<Node>,
    ani: Optional<ANI>, // TODO: ?
}

fn convert_timestamp(dt: u32) -> Result<DateTime<Utc>> {
    let Some(dt) = DateTime::from_timestamp(dt.into(), 0) else {
        bail!("Invalid timestamp");
    };
    Ok(dt)
}

#[binread]
#[derive(Debug, Serialize)]
struct VertexAnim {
    n_tr: u32,
    maybe_duration: f32,
    #[br(count=n_tr)]
    tris: Vec<[u8; 3]>,
}

#[binread]
#[br(magic = b"EVA\0")]
#[derive(Debug, Serialize)]
struct EVA {
    size: u32,
    #[br(assert(version==1,"Invalid EVA version"))]
    version: u32,
    num_verts: u32,
    #[br(count=num_verts)]
    verts: Vec<Optional<VertexAnim>>,
}

#[repr(u32)]
pub enum StmFlags {
    ani_pos = 0x1
}

#[repr(u32)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Sequence, Serialize, ToPrimitive)]
pub enum OptFlags {
    anim_pos = 0x1,
    ani_rot = 0x2,
    anim_color_intens = 0x18,
    anim_visible = 0x80
}


#[binread]
#[br(magic = b"NAM\0")]
#[derive(Debug, Serialize)]
struct NAM {
    size: u32,
    #[br(assert(version==1))]
    version: u32,
    primer_frames: u32,
    frames: u32,
    #[br(assert(flags&0xffffef60==0,"Invalid NAM flags"))]
    flags: u32,
    #[br(assert(opt_flags&0xfff8==0,"Invalid NAM opt_flags"))]
    opt_flags: u32,
    #[br(assert(stm_flags&0xfff8==0,"Invalid NAM stm_flags"))]
    stm_flags: u32,
    #[br(if(flags&0x1!=0),count=0xc)]
    unk_flags_1: Option<Vec<u8>>,
    #[br(if(flags&0x2!=0),count=0x10)]
    unk_flags_2: Option<Vec<u8>>,
    #[br(if(flags&0x4!=0),count=4)]
    unk_flags_3: Option<Vec<u8>>,
    #[br(if(flags&0x8!=0),count=4)]
    unk_flags_4: Option<Vec<u8>>,
    #[br(if(flags&0x10!=0),count=4)]
    unk_flags_5: Option<Vec<u8>>,
    #[br(if(flags&0x80!=0),count=1)]
    unk_flags_6: Option<Vec<u8>>,
    #[br(if(flags&0x1000!=0))]
    eva: Option<EVA>,
}

#[binread]
#[br(magic = b"NABK")]
#[derive(Debug, Serialize)]
struct NABK {
    size: u32,
    #[br(temp,count=size)]
    _data: Vec<u8>,
}

#[binread]
#[br(magic = b"ANI\0")]
#[derive(Debug, Serialize)]
struct ANI {
    size: u32,
    #[br(assert(version==2, "Invalid ANI version"))]
    version: u32,
    fps: f32,
    first_frame: u32,
    last_frame: u32,
    num_objects: u32,
    unk_flags: u32,
    num: u32,
    #[br(temp,count=num)]
    _data: Vec<u8>,
    nabk: NABK,
    #[br(count=num_objects)]
    nam: Vec<NAM>,
}

#[binread]
#[br(magic = b"SM3\0")]
#[derive(Debug, Serialize)]
pub(crate) struct SM3 {
    size: u32,
    #[br(temp,assert(const_1==0x6515f8,"Invalid timestamp"))]
    const_1: u32,
    #[br(try_map=convert_timestamp)]
    time_1: DateTime<Utc>,
    #[br(try_map=convert_timestamp)]
    time_2: DateTime<Utc>,
    pub scene: SCN,
}

impl SM3 {
    fn dependencies(&self) -> Vec<String> {
        let mut deps = vec![];
        for mat in &self.scene.mat {
            for map in [
                &mat.maps.diffuse,
                &mat.maps.bump,
                &mat.maps.glow,
                &mat.maps.metallic,
                &mat.maps.reflection,
            ]
            .into_iter()
            .flat_map(|m| m.value.as_ref())
            {
                deps.push(map.texture.string.clone())
            }
        }
        deps
    }
}

#[binread]
#[br(magic = b"CM3\0")]
#[derive(Debug, Serialize)]
struct CM3 {
    size: u32,
    #[br(temp,assert(const_1==0x6515f8,"Invalid timestamp"))]
    const_1: u32,
    #[br(try_map=convert_timestamp)]
    time_1: DateTime<Utc>,
    #[br(try_map=convert_timestamp)]
    time_2: DateTime<Utc>,
    scene: SCN,
}
impl CM3 {
    fn dependencies(&self) -> Vec<String> {
        let mut deps = vec![];
        for mat in &self.scene.mat {
            for map in [
                &mat.maps.diffuse,
                &mat.maps.bump,
                &mat.maps.glow,
                &mat.maps.metallic,
                &mat.maps.reflection,
            ]
            .into_iter()
            .flat_map(|m| m.value.as_ref())
            {
                deps.push(map.texture.string.clone())
            }
        }
        deps
    }
}

#[binread]
#[derive(Debug, Serialize)]
pub(crate) struct Dummy {
    has_next: u32,
    pub name: PascalString,
    pub pos: [f32; 3],
    pub rot: [f32; 3],
    pub info: Optional<INI>,
}

#[binread]
#[br(magic = b"DUM\0")]
#[derive(Debug, Serialize)]
pub(crate) struct DUM {
    size: u32,
    #[br(assert(version==1, "Invalid DUM version"))]
    version: u32,
    num_dummies: u32,
    #[br(count=num_dummies)]
    pub dummies: Vec<Dummy>,
}

#[binread]
#[br(magic = b"QUAD")]
#[derive(Debug, Serialize)]
struct QUAD {
    size: u32,
    #[br(assert(version==1, "Invalid QUAD version"))]
    version: u32,
    mesh: u32,
    table: Table<u16>,
    f_4: [f32; 4],
    #[br(temp)]
    num_children: u32,
    #[br(count=num_children)]
    pub children: Vec<QUAD>,
}

#[binread]
#[br(magic = b"CMSH")]
#[derive(Debug, Serialize)]
struct CMSH {
    size: u32,
    #[br(assert(version==2, "Invalid CMSH version"))]
    version: u32,
    #[br(assert(collide_mesh_size==0x34, "Invalid collision mesh size"))]
    collide_mesh_size: u32,
    zone_name: PascalString,
    unk_1: u16,
    sector: u16,
    unk_2: u16,
    index: u8,
    unk_4: u8,
    bbox_1: [[f32; 3]; 2],
    #[br(temp)]
    _t_1: Table<[f32; 3]>,
    #[br(temp)]
    _t_2: RawTable<0x1c>,
}

#[binread]
#[br(magic = b"AMC\0")]
#[derive(Debug, Serialize)]
struct EmptyAMC {
    #[br(assert(size==0))]
    size: u32,
}

#[binread]
#[br(magic = b"AMC\0")]
#[derive(Debug, Serialize)]
pub(crate) struct AMC {
    size: u32,
    #[br(assert(version==100,"Invalid AMC version"))]
    version: u32,
    #[br(assert(version_code==0, "Invalid AMC version_code"))]
    version_code: u32,
    bbox_1: [[f32; 3]; 2],
    num_tris: u32,
    bbox_2: [[f32; 3]; 2],
    unk: [f32; 3],
    cmsh: [CMSH; 2],
    num_sectors: u32,
    #[br(count=num_sectors)]
    sector_col: Vec<[CMSH; 2]>,
    grid_size: [u32; 2],
    grid_scale: [f32; 2],
    unk_f: [f32; 2],
    #[br(temp)]
    num_quads: u32,
    #[br(count=num_quads)]
    pub quads: Vec<QUAD>,
    #[br(temp)]
    _empty_amc: EmptyAMC,
}

#[binread]
#[br(import(version: u32))]
#[derive(Debug, Serialize)]
pub(crate) struct TriV104 {
    #[br(if(version>=0x69))]
    pub sector_name: Option<PascalString>,
    pub mat_key: u32,
    pub map_key: u32,
    num_tris: u32,
    #[br(count=num_tris)]
    pub tris: Vec<[u16; 3]>,
    pub verts_1: LFVF,
    pub verts_2: LFVF,
}

#[binread]
#[br(magic = b"TRI\0", import(version: u32))]
#[derive(Debug, Serialize)]
pub(crate) struct TRI {
    size: u32,
    pub flags: u32,
    pub name: PascalString,
    pub sector_num: u32, // if 0xffffffff sometimes TriV104 has no name_2 field
    #[br(args(version))]
    pub data: TriV104,
}

#[binread]
#[derive(Debug, Serialize)]
pub(crate) struct EMI_Textures {
    pub key: u32,
    #[br(if(key!=0))]
    pub data: Option<(PascalString, u32, PascalString)>,
}

#[binread]
#[br(magic = b"EMI\0")]
#[derive(Debug, Serialize)]
pub(crate) struct EMI {
    size: u32,
    #[br(assert((103..=105).contains(&version)))]
    pub version: u32,
    pub num_materials: u32,
    #[br(count=num_materials)]
    pub materials: Vec<(u32, MAT)>,
    #[br(parse_with = until_exclusive(|v: &EMI_Textures| v.key==0))]
    pub maps: Vec<EMI_Textures>,
    pub num_lists: u32,
    #[br(count=num_lists,args{inner: (version,)})]
    pub tri: Vec<TRI>,
}

impl EMI {
    fn dependencies(&self) -> Vec<String> {
        let mut deps = vec![];
        for map in &self.maps {
            if let Some((path_1, _, path_2)) = map.data.as_ref() {
                deps.push(path_1.string.clone());
                deps.push(path_2.string.clone());
            }
        }
        for (_, mat) in &self.materials {
            for map in [
                &mat.maps.diffuse,
                &mat.maps.bump,
                &mat.maps.glow,
                &mat.maps.metallic,
                &mat.maps.reflection,
            ]
            .into_iter()
            .flat_map(|m| m.value.as_ref())
            {
                deps.push(map.texture.string.clone())
            }
        }
        deps
    }
}

#[binread]
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(crate) enum Data {
    SM3(SM3),
    CM3(CM3),
    DUM(DUM),
    AMC(AMC),
    EMI(EMI),
}

impl Data {
    fn dependencies(&self) -> Vec<String> {
        match self {
            Data::SM3(sm3) => sm3.dependencies(),
            Data::CM3(cm3) => cm3.dependencies(),
            Data::EMI(emi) => emi.dependencies(),
            _ => vec![],
        }
    }
}

fn parse_file(path: &VfsPath) -> Result<Data> {
    let mut rest_size = 0;
    let mut fh = BufReader::new(path.open_file()?);
    let ret = fh.read_le()?;
    let pos = fh.stream_position().unwrap_or(0);
    eprintln!("Read {} bytes from {}", pos, path.as_str());
    let mut buffer = [0u8; 0x1000];
    if let Ok(n) = fh.read(&mut buffer) {
        if n != 0 {
            eprintln!("Rest:\n{}", rhexdumps!(&buffer[..n], pos));
        }
    };
    while let Ok(n) = fh.read(&mut buffer) {
        if n == 0 {
            break;
        }
        rest_size += n;
    }
    eprintln!("+{rest_size} unparsed bytes");
    Ok(ret)
}

fn load_ini(path: &VfsPath) -> IniData {
    let Ok(data) = path.read_to_string() else {
        return IniData::default();
    };
    Ini::new().read(data).unwrap_or_default()
}

#[derive(Serialize, Debug)]
pub(crate) struct Level {
    pub config: IniData,
    pub moredummies: IniData,
    pub emi: EMI,
    pub sm3: [Option<SM3>; 2],
    pub dummies: DUM,
    pub path: String,
    pub dependencies: BTreeMap<String, String>,
}

impl Level {
    fn load(path: &VfsPath) -> Result<Self> {
        let map_path = path.join("map")?;
        let emi_path = map_path.join("map3d.emi")?;
        let sm3_path = map_path.join("map3d.sm3")?;
        let sm3_2_path = map_path.join("map3d_2.sm3")?;
        let dum_path = map_path.join("map3d.dum")?;
        let config_file = map_path.join("map3d.ini")?;
        let moredummies = map_path.join("moredummies.ini")?;
        let config = load_ini(&config_file);
        let moredummies = load_ini(&moredummies);
        let Data::EMI(emi) = parse_file(&emi_path)? else {
            bail!(
                "Failed to parse EMI at {emi_path}",
                emi_path = emi_path.as_str()
            );
        };

        let sm3 = match parse_file(&sm3_path)? {
            Data::SM3(sm3) => sm3,
            _ => bail!(
                "Failed to parse SM3 at {sm3_path}",
                sm3_path = sm3_path.as_str()
            ),
        };

        let sm3_2 = match parse_file(&sm3_2_path) {
            Ok(Data::SM3(sm3_2)) => Some(sm3_2),
            Ok(_) => bail!(
                "Failed to parse SM3 at {sm3_2}",
                sm3_2 = sm3_2_path.as_str()
            ),
            Err(e) => {
                println!(
                    "Failed to parse {sm3_2_path}: {e}",
                    sm3_2_path = sm3_2_path.as_str()
                );
                None
            }
        };

        let Data::DUM(dummies) = parse_file(&dum_path)? else {
            bail!(
                "Failed to parse DUM at {dum_path}",
                dum_path = dum_path.as_str()
            );
        };

        let mut dependencies = BTreeMap::new();
        let sm3_2_deps: Vec<String> = sm3_2.iter().flat_map(|v| v.dependencies()).collect();
        for dep in [sm3.dependencies(), sm3_2_deps, emi.dependencies()]
            .into_iter()
            .flatten()
        {
            match resolve_dep(&dep, &map_path, &config) {
                Some(res) => {
                    dependencies.insert(dep, res.as_str().to_owned());
                }
                None => {
                    println!("Failed to resolve dependency: {}", dep);
                    continue;
                }
            }
        }
        Ok(Level {
            config,
            moredummies,
            emi,
            sm3: [Some(sm3), sm3_2],
            dummies,
            path: path.as_str().to_owned(),
            dependencies,
        })
    }
}

fn ancestors(path: &VfsPath) -> Vec<VfsPath> {
    let mut ret = vec![];
    let mut path = path.clone();
    loop {
        ret.push(path.clone());
        if path.is_root() {
            break;
        }
        path = path.parent();
    }
    ret
}

fn with_extension(path: &str, ext: &str) -> String {
    PathBuf::from(path)
        .with_extension(ext)
        .as_os_str()
        .to_str()
        .unwrap_or_default()
        .to_owned()
}

fn resolve_dep(dep: &str, level_path: &VfsPath, config: &IniData) -> Option<VfsPath> {
    let root = level_path.root();
    const EXTS: &[&str] = &["png", "bmp", "dds", "tga", "alpha.dds"];
    let tex_path = config
        .get("model")
        .and_then(|config| config.get("texturepath"))
        .cloned()
        .flatten()
        .and_then(|path| root.join(path).ok())
        .map(|path| ancestors(&path))
        .unwrap_or_default();
    for path in ancestors(level_path)
        .into_iter()
        .chain(tex_path.into_iter())
    {
        for &ext in EXTS {
            let dep = with_extension(dep, ext);
            let dep = dep.split('/').collect::<Vec<_>>();
            let Some((&dep_filename, dep_path)) = dep.split_last() else {
                continue;
            };
            for dds in [true, false] {
                let path: String = path
                    .as_str()
                    .split('/')
                    .filter(|v| !v.is_empty())
                    .chain(dep_path.iter().copied())
                    .chain(dds.then_some("dds").into_iter())
                    .chain(std::iter::once(dep_filename))
                    .collect::<Vec<&str>>()
                    .join("/");
                if let Ok(path) = root.join(&path) {
                    if path.exists().unwrap_or(false) {
                        return Some(path);
                    }
                }
            }
        }
    }
    None
}

pub(crate) fn find_packed<P: AsRef<Path>>(root: P) -> Result<Vec<PathBuf>> {
    let mut files = vec![];
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path
            .extension()
            .map(|e| e.to_str() == Some("packed"))
            .unwrap_or(false)
        {
            let path = entry.path().to_owned();
            files.push(path);
        }
    }
    Ok(files)
}

#[derive(Debug)]
pub(crate) enum ParsedData {
    Level(Level),
    Data(Data),
}

pub(crate) mod multi_pack_fs {
    use std::collections::{BTreeMap, HashMap};

    use anyhow::bail;
    use vfs::{SeekAndRead, VfsPath};

    use super::{ParsedData, PathBuf, Result, Serialize};
    use crate::packed_vfs::MultiPack;

    #[derive(Serialize, Debug)]
    pub(crate) struct Entry {
        pub path: String,
        pub size: u64,
        pub is_file: bool,
    }

    pub(crate) struct MultiPackFS {
        fs: VfsPath,
        current: Vec<String>,
    }

    impl MultiPackFS {
        pub(crate) fn new(files: Vec<PathBuf>) -> Result<Self> {
            MultiPack::load_all(&files).map(|fs| MultiPackFS {
                fs: fs.into(),
                current: vec![],
            })
        }

        pub(crate) fn exists(&self, path: &str) -> Result<bool> {
            Ok(self.fs.root().join(path).and_then(|p| p.metadata()).is_ok())
        }

        pub(crate) fn is_level(&self, path: Option<String>) -> Result<bool> {
            let mut root = self.fs.root();
            if let Some(path) = path {
                root = root.join(path)?
            } else {
                for entry in &self.current {
                    root = root.join(entry)?
                }
            }
            let Ok(path) = root.join("map") else {
                return Ok(false);
            };
            let mut ret = true;
            for file in &[
                "map3d.emi",
                "map3d.sm3",
                "map3d.dum",
                "map3d.ini",
                "map3d.amc",
            ] {
                ret &= path.join(file).and_then(|p| p.is_file()).unwrap_or(false);
            }
            Ok(ret)
        }

        pub(crate) fn ls(&self) -> Result<Vec<Entry>> {
            let mut ret = vec![];
            let mut root = self.fs.root();
            for entry in &self.current {
                root = root.join(entry)?;
            }
            for entry in root.read_dir()? {
                let is_file = entry.is_file()?;
                let meta = entry.metadata()?;
                ret.push(Entry {
                    path: entry.as_str().to_owned(),
                    size: meta.len,
                    is_file,
                })
            }
            ret.sort_by(|a, b| {
                let k_1 = (a.is_file, a.path.as_str());
                let k_2 = (b.is_file, b.path.as_str());
                k_1.cmp(&k_2)
            });
            Ok(ret)
        }

        pub(crate) fn entries(&self) -> Result<Vec<Entry>> {
            let mut entries = vec![];
            for res in self.fs.walk_dir()? {
                let res = res?;
                let path = res.as_str().to_owned();
                let meta = res.metadata()?;
                entries.push(Entry {
                    path,
                    size: meta.len,
                    is_file: res.is_file()?,
                });
            }
            Ok(entries)
        }

        pub(crate) fn pwd(&self) -> Result<String> {
            if self.current.is_empty() {
                return Ok("/".to_owned());
            }
            let mut root = self.fs.root();
            for entry in &self.current {
                root = root.join(entry)?
            }
            Ok(root.as_str().to_owned())
        }

        pub(crate) fn cd(&mut self, path: &str) -> Result<()> {
            self.current = self.current.drain(..).filter(|v| !v.is_empty()).collect();
            if path == ".." {
                self.current.pop();
                return Ok(());
            }
            let mut root = self.fs.root();
            for entry in &self.current {
                root = root.join(entry)?;
            }
            root = root.join(path)?;
            if !root.is_dir()? {
                bail!("Can't change directory to a file");
            }
            self.current.push(path.to_owned());
            Ok(())
        }

        pub(crate) fn dependencies(&self, path: &str) -> Result<BTreeMap<String, String>> {
            let mut root = self.fs.root();
            for entry in &self.current {
                root = root.join(entry)?;
            }
            let path = root.join(path)?;
            let data = match path.metadata()?.file_type {
                vfs::VfsFileType::File => {
                    println!("File: {}", path.as_str());
                    let data = super::parse_file(&path)?;
                    data.dependencies()
                        .into_iter()
                        .map(|v| (v.clone(), v))
                        .collect()
                }
                vfs::VfsFileType::Directory => {
                    println!("Level directory: {}", path.as_str());
                    super::Level::load(&path)?.dependencies
                }
            };
            Ok(data)
        }

        pub(crate) fn open_file(&self, path: &str) -> Result<Box<dyn SeekAndRead>> {
            let mut root = self.fs.root();
            for entry in &self.current {
                root = root.join(entry)?
            }
            let path = root.join(path)?;

            if path.is_dir()? {
                bail!("{path} is a directory", path = path.as_str());
            }
            Ok(path.open_file()?)
        }

        pub(crate) fn parse_file(&self, path: &str) -> Result<ParsedData> {
            let mut root = self.fs.root();
            for entry in &self.current {
                root = root.join(entry)?;
            }
            let path = root.join(path)?;
            let data = match path.metadata()?.file_type {
                vfs::VfsFileType::File => {
                    println!("File: {}", path.as_str());
                    ParsedData::Data(super::parse_file(&path)?)
                }
                vfs::VfsFileType::Directory => {
                    println!("Level directory: {}", path.as_str());
                    ParsedData::Level(super::Level::load(&path)?)
                }
            };
            Ok(data)
        }
    }
}
