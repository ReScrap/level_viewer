#![allow(clippy::upper_case_acronyms, non_camel_case_types)]
use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::{Debug, Display},
    io::{BufReader, Cursor, Read, Seek},
    ops::Deref,
    path::{Path, PathBuf},
};

use bilge::prelude::*;
use binrw::{args, helpers::until_exclusive, prelude::*};
use chrono::{DateTime, Utc};
use color_eyre::eyre::{Context, Result, anyhow, bail};
use configparser::ini::{Ini, IniDefault};
use enum_iterator::Sequence;
use indexmap::IndexMap;
use log::warn;
use num_derive::ToPrimitive;
use num_traits::ToPrimitive;
use rhexdump::rhexdumps;
use serde::{Deserialize, Serialize};
use vfs::VfsPath;
use walkdir::WalkDir;

pub(crate) type IniData = IndexMap<String, IndexMap<String, Option<String>>>;

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
    pub data: Vec<Vec<u8>>,
}

#[binread]
#[derive(Serialize, Debug)]
pub(crate) struct Table<const SIZE: u32, T: for<'a> BinRead<Args<'a> = ()> + 'static> {
    num_entries: u32,
    #[br(assert(entry_size==SIZE))]
    entry_size: u32,
    #[br(count=num_entries)]
    pub data: Vec<T>,
}

// impl<T: for<'a> BinRead<Args<'a> = ()>> Serialize for Table<T> where T: Serialize {
//     fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer {
//         self.data.serialize(serializer)
//     }
// }

#[binread]
#[derive(Clone)]
pub(crate) struct Optional<T: for<'a> BinRead<Args<'a> = ()>> {
    #[br(temp)]
    has_value: u32,
    #[br(if(has_value!=0))]
    value: Option<T>,
}

impl<T> Optional<T>
where
    T: for<'a> BinRead<Args<'a> = ()>,
{
    pub(crate) fn get(&self) -> Option<&T> {
        self.value.as_ref()
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

impl std::ops::Deref for PascalString {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        self.string.as_str()
    }
}

impl AsRef<str> for PascalString {
    fn as_ref(&self) -> &str {
        self.string.as_str()
    }
}

impl Serialize for PascalString {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.string.serialize(serializer)
    }
}

impl Display for PascalString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.string, f)
    }
}

impl Debug for PascalString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.string, f)
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

fn parse_ini(data: &str) -> Ini {
    let mut def = IniDefault::default();
    def.comment_symbols = vec![';'];
    def.delimiters = vec!['='];
    def.boolean_values
        .entry(true)
        .or_default()
        .push("si".to_owned());
    let mut ini = Ini::new_from_defaults(def);
    ini.read(data.to_owned()).ok();
    ini
}

impl INI {
    pub(crate) fn data(&self) -> Ini {
        parse_ini(&format!("{self}"))
    }
}

impl std::fmt::Display for INI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for section in &self.sections {
            for line in &section.sections {
                writeln!(f, "{}", line.string.trim_end_matches('\n'))?;
            }
        }
        Ok(())
    }
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

impl RGBA {
    pub(crate) fn as_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a].map(|v| (v as f32) / 255.0)
    }
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
    pub vert_fmt: FVF,
    #[br(assert(vertex_size_from_id(fmt_id).ok()==Some(vert_size)))]
    pub vert_size: u32,
    #[br(temp)]
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

#[binrw]
#[derive(Debug, Serialize)]
pub(crate) struct MD3D_Tris {
    num_tris: u32,
    #[br(assert(tri_size==6,"Invalid MD3D tri size"))]
    tri_size: u32,
    #[br(count=num_tris)]
    pub tris: Vec<[u16; 3]>,
}

#[binrw]
#[derive(Debug, Serialize)]
pub(crate) struct MD3D_TriSeg {
    dist_xor: u32,
    pub normal: [f32; 3],
}

#[binrw]
#[derive(Debug, Serialize)]
pub(crate) struct MD3D_Segment {
    tri_a: i16,
    tri_b: i16,
    vert_a: u16,
    vert_b: u16,
}

#[binrw]
#[derive(Debug, Serialize)]
pub(crate) struct MD3D_Skin {
    pub influence_count: u8,
    pub bone_indices: [u8; 3],
    pub weights: [f32; 3],
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
    pub vert_orig: Table<2, u16>,
    pub tri_seg_start: u32,
    pub tri_seg: Table<0x10, MD3D_TriSeg>,
    pub segments: Table<8, MD3D_Segment>,
    pub vert_orig_pos: Table<0xc, [f32; 3]>,
    pub tri_flags: Table<4, u32>,
    pub skin_ref: u32,
    #[br(if(skin_ref==0))]
    pub skin: Option<Table<0x10, MD3D_Skin>>,
    pub mat_index: i32,
    // Stored as read_int_nonzero/write_int(bool) in CMallaD3D (offsets +0x50/+0x51).
    pub feature_flag_0x50: u32,
    pub feature_flag_0x51: u32,
    #[br(count = 0x18)]
    field_0x54_blob: Vec<u8>,
    #[br(count = 0x18)]
    field_0x6c_blob: Vec<u8>,
    #[br(count = 0xc)]
    field_0x84_blob: Vec<u8>,
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
    TriangleMesh,
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
    scale: [f32; 2],
    name_1: PascalString,
    name_2: PascalString,
    diffuse_mod: RGBA,
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
    angles: [f32; 3],
    origin: [f32; 3],
    target: [f32; 3],
    clip: [f32; 2],
    range: [f32; 2],
    fov: f32,
    phys_aspect_ratio: f32,
    view_aspect_ratio: f32,
    mode: u32,
}

#[binread]
#[br(repr=u32)]
#[repr(u32)]
#[derive(Debug, Serialize)]
pub(crate) enum LightType {
    Point = 5000,
    Spot = 5001,
    Directional = 5002,
}

#[binread]
#[br(magic = b"LUZ\0")]
#[derive(Debug, Serialize)]
pub(crate) struct LUZ {
    size: u32,
    #[br(assert(version==1,"Invalid LUZ version"))]
    version: u32,
    sector: u32,
    pub light_type: LightType,
    pub shadows: u8,
    pub pos: [f32; 3],
    pub dir: [f32; 3],
    pub col: RGBA,
    pub power: f32,
    pub att: [f32; 2],
    pub hotspot: f32,
    pub falloff: f32,
    pub mult: f32,
    pub radcoeff: f32,
    #[br(map = |v: u32| v != 0)]
    pub active: bool,
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
pub(crate) enum NodeFlags {
    ROOT,
    GROUP,
    SELECTED,
    HIDDEN,
    INHERIT_VISIBLE,
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
    MORPH,
    TWOSIDES,
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
    pub object_index: i32,
    pub table_index: i32,
    pub node_xref: i32,
    #[br(map=parse_node_flags)]
    pub flags: BTreeSet<NodeFlags>,
    pub ani_mask: i32,
    pub name: PascalString,
    pub parent: PascalString,
    pub pos_offset: [f32; 3],
    pub rot: [f32; 4],
    pub scale: f32,
    pub transform_world: [[f32; 4]; 4], // 0x40 4x4 Matrix
    pub transform_local: [[f32; 4]; 4], // 0x40 4x4 Matrix
    pub rest_rot: [f32; 4],
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
    pub filter: u8,
    pub max_anisotropy: u8,
    pub is_square: u8,
    pub is_env: u8,
    pub tile: u8,
    pub mirror: u8,
    pub map_type: u8,
    pub displacement: [f32; 2],
    pub scale: [f32; 2],
    pub quantity: f32, // Bumpmap scaling
    #[br(if(version==3))]
    pub uv_matrix: Option<[f32; 2]>,
    #[br(if(version==3))]
    pub angle: Option<f32>,
}

#[binread]
#[br(repr=u32)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Sequence, Serialize, ToPrimitive)]

pub(crate) enum BlendMode {
    Zero = 1,
    One = 2,
    SrcColor = 3,
    InvSrcColor = 4,
    SrcAlpha = 5,
    InvSrcAlpha = 6,
    DestAlpha = 7,
    InvDestAlpha = 8,
    DestColor = 9,
    InvDestColor = 10,
    SrcAlphaSat = 11,
    BothSrcAlpha = 12,
    BothInvSrcAlpha = 13,
}

#[binread]
#[br(repr=u32)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Sequence, Serialize, ToPrimitive)]
pub(crate) enum CmpFunc {
    Never = 1,
    Less = 2,
    SrcColor = 3,
    InvSrcColor = 4,
    SrcAlpha = 5,
    InvSrcAlpha = 6,
    DestAlpha = 7,
    InvDestAlpha = 8,
    DestColor = 9,
    InvDestColor = 10,
    SrcAlphaSat = 11,
    BothSrcAlpha = 12,
    BothInvSrcAlpha = 13,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Sequence, Serialize, ToPrimitive, Clone)]
#[repr(u8)]
pub(crate) enum MatPropAttrib {
    NO_COLLID = 0,
    AGUA = 1,
    FOG_CLOUDS = 2,
    SHAREABLE = 3,
    SORT_ZBIAS = 4,
    ALPHA_TEST = 5,
    HAS_TEXTURE = 6, // ??? (check 0x64d4a0)
    USE_AMBIENT = 7,
    AREA_LIGHT = 8,
    SHADER = 9,
    ZBIAS = 11,
    XSHADOW = 14,
}

fn parse_mat_prop_flags(flags: u16) -> BTreeSet<MatPropAttrib> {
    enum_iterator::all::<MatPropAttrib>()
        .filter_map(|flag| ((flags & (1 << flag.to_u8().unwrap_or(0xff))) != 0).then_some(flag))
        .collect()
}

#[binread]
#[derive(Debug, Serialize, Clone)]
pub(crate) struct MatProps {
    #[br(assert(sub_material==0))]
    pub sub_material: u32,
    pub orig: BlendMode,
    pub dest: BlendMode,
    pub two_sided: u8,
    pub dyn_illum: u8,
    pub dif_alpha: u8,
    pub env_map: u8,
    #[br(map=parse_mat_prop_flags)]
    pub attrib: BTreeSet<MatPropAttrib>,
    pub enable_fog: u8,
    pub z_write: u8,
    pub zfunc: CmpFunc,
}

#[binread]
#[br(magic = b"MAT\0")]
#[derive(Debug, Serialize, Clone)]
pub(crate) struct MAT {
    size: u32,
    #[br(assert((1..=3).contains(&version),"invalid MAT version"))]
    version: u32,
    #[br(if(version>1))]
    pub name: Option<PascalString>,
    pub ambient_override: RGBA,
    pub diffuse_mod: RGBA,
    pub diffuse: RGBA,
    pub specular: RGBA,
    pub glow: RGBA,
    pub spec_power: f32,
    pub spec_mult: f32,
    pub mat_props: MatProps,
    pub maps: [Optional<MAP>; 5], // diffuse, metallic, env, bump, glow
}

#[binread]
#[derive(Debug, Serialize)]
pub(crate) struct LightColor {
    pub color: RGBA,
    pub intensity: f32,
}

#[binread]
#[br(magic = b"SCN\0")]
#[derive(Debug, Serialize)]
pub(crate) struct SCN {
    // 0x650220
    size: u32,
    #[br(temp,assert(version==1))]
    version: u32,
    pub model_name: PascalString,
    pub root_node: PascalString,
    pub node_props: Optional<INI>,
    pub ambient: LightColor,
    pub background: LightColor,
    pub bbox: [[f32; 3]; 2],
    // #[br(assert(collide_mesh_ref==0))]
    collide_mesh_ref: u32,
    pub user_props: Optional<INI>,
    #[br(temp)]
    num_materials: u32,
    #[br(count=num_materials)]
    pub mat: Vec<MAT>,
    #[br(temp,assert(nodes_section_marker==1))]
    nodes_section_marker: u32,
    #[br(temp)]
    num_nodes: u32,
    #[br(count = num_nodes)] // 32
    pub nodes: Vec<Node>,
    pub ani: Optional<ANI>,
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
    fps: f32,
    #[br(count=n_tr)]
    tris: Vec<[u8; 3]>,
}

#[binread]
#[br(magic = b"EVA\0")]
#[derive(Debug, Serialize)]
pub(crate) struct EVA {
    size: u32,
    #[br(assert(version==1,"Invalid EVA version"))]
    version: u32,
    num_verts: u32,
    #[br(count=num_verts)]
    verts: Vec<Optional<VertexAnim>>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Sequence, Serialize, ToPrimitive)]
#[repr(u8)]
pub(crate) enum AniTrackType {
    Position = 0,   // [f32;3]
    Rotation = 1,   // [f32;4]
    FOV = 2,        // f32
    Color = 3,      // [u8;4]
    Intensity = 4,  // f32
    Visibility = 7, // u8
    EVA = 12,       // N/A
}

impl AniTrackType {
    fn size(&self) -> usize {
        match self {
            AniTrackType::Position => 3 * 4,
            AniTrackType::Rotation => 4 * 4,
            AniTrackType::FOV => 1 * 4,
            AniTrackType::Color => 4 * 1,
            AniTrackType::Intensity => 1 * 4,
            AniTrackType::Visibility => 1 * 1,
            AniTrackType::EVA => 0,
        }
    }

    fn mask(&self) -> u32 {
        1u32 << unsafe { *<*const _>::from(self).cast::<u8>() }
    }
}

fn parse_ani_track_type(flags: u32) -> Result<BTreeSet<AniTrackType>> {
    if flags & 0xffffef60 != 0 {
        bail!("unsupported flags detected!");
    }
    Ok(enum_iterator::all::<AniTrackType>()
        .filter(|flag| (flags & (1 << flag.to_u8().unwrap_or(0xff))) != 0)
        .collect())
}

#[derive(Debug, Serialize)]
pub(crate) struct BlockInfo {
    pub size: usize,
    pub elem_size: usize,
    pub stream: bool,
    pub optimized: bool,
    pub track_type: AniTrackType,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct AnimTracks {
    pub pos: Option<Vec<[f32; 3]>>,
    pub rot: Option<Vec<[f32; 4]>>,
    pub fov: Option<Vec<f32>>,
    pub color: Option<Vec<[u8; 4]>>,
    pub intensity: Option<Vec<f32>>,
    pub visibility: Option<Vec<bool>>,
}

fn parse_track_data<T>(data: &[u8]) -> Result<Vec<T>>
where
    T: for<'a> BinRead<Args<'a> = ()>,
{
    let mut reader = Cursor::new(data);
    let mut out = Vec::with_capacity(data.len() / std::mem::size_of::<T>().max(1));
    while (reader.position() as usize) < data.len() {
        out.push(reader.read_le()?);
    }
    Ok(out)
}

#[binrw::parser(reader, endian)]
fn parse_ani_blocks(
    frames: u32,
    cm3_flags: &BTreeSet<AniTrackType>,
    opt_flags: u32,
    stm_flags: u32,
) -> BinResult<Vec<BlockInfo>> {
    let mut blocks = Vec::new();
    for track in cm3_flags.iter() {
        let mut block_info = BlockInfo {
            size: track.size(),
            elem_size: track.size(),
            track_type: *track,
            optimized: (opt_flags & track.mask()) != 0,
            stream: (stm_flags & track.mask()) != 0,
        };
        if block_info.optimized && block_info.stream {
            block_info.size = <u32>::read_options(reader, endian, ())? as usize;
        }
        if !block_info.optimized {
            block_info.size *= frames as usize;
        }
        blocks.push(block_info);
    }
    Ok(blocks)
}

#[binread]
#[derive(Debug)]
struct AniStreamHeader {
    size: u16,
    start_frame: u16,
    num_frames: u16,
}

#[binread]
#[br(magic = b"NAM\0")]
#[derive(Debug, Serialize)]
pub(crate) struct NAM {
    size: u32,
    #[br(assert(version==1))]
    version: u32,
    pub start_frame: u32,
    pub frames: u32,
    #[br(try_map(|v: u32| parse_ani_track_type(v)))]
    pub cm3_flags: BTreeSet<AniTrackType>,
    #[br(assert(opt_flags & 0xfff8 == 0x8000u32))]
    #[br(map(|v: u32| v|0x8000u32))]
    pub opt_flags: u32,
    #[br(assert(stm_flags & 0xfff8 == 0))]
    pub stm_flags: u32,
    #[br(parse_with = parse_ani_blocks, args(frames,&cm3_flags,opt_flags,stm_flags))]
    pub tracks: Vec<BlockInfo>,
    #[br(if(cm3_flags.contains(&AniTrackType::EVA)))]
    pub eva: Option<EVA>,
}

#[binread]
#[br(magic = b"NABK")]
#[derive(Debug, Serialize)]
pub(crate) struct NABK {
    size: u32,
    #[br(count=size)]
    pub data: Vec<u8>,
}

#[binread]
#[br(magic = b"ANI\0")]
#[derive(Debug, Serialize)]
pub(crate) struct ANI {
    size: u32,
    #[br(assert(version==2, "Invalid ANI version"))]
    version: u32,
    pub fps: f32,
    pub first_frame: u32,
    pub last_frame: u32,
    pub num_objects: u32,
    active_flag: u32,
    num_nodes: u32,
    #[br(count=num_nodes, map=|data: Vec<u8>| data.iter().map(|&v| (v!=0).then_some(v-1)).collect())]
    pub track_map: Vec<Option<u8>>,
    #[br(map=|v: NABK| v.data)]
    pub data: Vec<u8>,
    #[br(count = num_objects)]
    pub tracks: Vec<NAM>,
}

impl ANI {
    pub(crate) fn get_track(&self, index: usize) -> Result<Option<AnimTracks>> {
        let Some(track) = self.tracks.get(index) else {
            return Ok(None);
        };
        let mut offset: usize = self
            .tracks
            .iter()
            .take(index)
            .flat_map(|t| t.tracks.iter())
            .map(|b| b.size)
            .sum();
        let mut out = AnimTracks::default();
        for block in &track.tracks {
            let end = offset
                .checked_add(block.size)
                .ok_or_else(|| anyhow!("ANI block size overflow"))?;
            let raw = self
                .data
                .get(offset..end)
                .ok_or_else(|| anyhow!("ANI/NABK block out of bounds"))?;
            offset = end;
            let payload = if block.optimized && block.stream {
                let mut reader = Cursor::new(raw);
                let header: AniStreamHeader = reader.read_le()?;
                if header.size as usize != block.size {
                    bail!(
                        "invalid ANI streamed block size: {} != {}",
                        header.size,
                        block.size
                    );
                }
                &raw[reader.position() as usize..]
            } else {
                raw
            };
            match block.track_type {
                AniTrackType::Position => out.pos = Some(parse_track_data(payload)?),
                AniTrackType::Rotation => out.rot = Some(parse_track_data(payload)?),
                AniTrackType::FOV => out.fov = Some(parse_track_data(payload)?),
                AniTrackType::Color => out.color = Some(parse_track_data(payload)?),
                AniTrackType::Intensity => out.intensity = Some(parse_track_data(payload)?),
                AniTrackType::Visibility => {
                    out.visibility = Some(
                        parse_track_data::<u8>(payload)?
                            .into_iter()
                            .map(|v| v != 0)
                            .collect(),
                    )
                }
                AniTrackType::EVA => (),
            }
        }
        Ok(Some(out))
    }
}

#[binread]
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
            for map in mat.maps.iter().flat_map(|m| m.value.as_ref()) {
                deps.push(map.texture.string.clone())
            }
        }
        deps
    }
}

#[binread]
#[derive(Debug, Serialize)]
pub(crate) struct CM3 {
    size: u32,
    #[br(temp,assert(const_1==0x6515f8,"Invalid timestamp"))]
    const_1: u32,
    #[br(try_map=convert_timestamp)]
    time_1: DateTime<Utc>,
    #[br(try_map=convert_timestamp)]
    time_2: DateTime<Utc>,
    pub scene: SCN,
}

impl CM3 {
    fn dependencies(&self) -> Vec<String> {
        let mut deps = vec![];
        for mat in &self.scene.mat {
            for map in mat.maps.iter().flat_map(|m| m.value.as_ref()) {
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
pub(crate) struct QUAD {
    size: u32,
    #[br(assert(version==1, "Invalid QUAD version"))]
    version: u32,
    mesh: u32,
    table: Table<2,u16>,
    f_4: [f32; 4],
    #[br(temp)]
    num_children: u32,
    #[br(count=num_children)]
    pub children: Vec<QUAD>,
}

#[binread]
#[derive(Debug, Serialize)]
pub(crate) struct CMSH_Tri {
    t: u32,
    pub normal: [f32; 3],
    dist: f32,
    pub idx: [u16; 3],
    flags: u16,
}

#[binread]
#[br(magic = b"CMSH")]
#[derive(Debug, Serialize)]
pub(crate) struct CMSH {
    size: u32,
    #[br(assert(version==2, "Invalid CMSH version"))]
    version: u32,
    #[br(assert(collide_mesh_size==0x34, "Invalid collision mesh size"))]
    collide_mesh_size: u32,
    pub zone_name: PascalString,
    mesh_flags: u16,
    pub sector: u16,
    mesh_uid: u16,
    mesh_id: u8,
    unk_4: u8,
    bbox_1: [[f32; 3]; 2],
    pub verts: Table<0xc, [f32; 3]>,
    pub tris: Table<0x0, CMSH_Tri>,
}

#[binread]
#[br(magic = b"AMC\0")]
#[derive(Debug, Serialize)]
struct EmptyAMC {
    #[br(assert(size==0))]
    size: u32,
}

// TODO: OG game uses version_code==1
#[binread]
#[derive(Debug, Serialize)]
pub(crate) struct AMC {
    size: u32,
    #[br(assert(version==100,"Invalid AMC version"))]
    version: u32,
    #[br(assert(version_code==0, "Invalid AMC version_code: {}", version_code))]
    version_code: u32,
    bbox_1: [[f32; 3]; 2],
    num_tris: u32,
    bbox_2: [[f32; 3]; 2],
    unk: [f32; 3],
    pub cmsh: [CMSH; 2],
    num_sectors: u32,
    #[br(count=num_sectors)]
    pub sector_col: Vec<[CMSH; 2]>,
    grid_size: [u32; 2],
    grid_scale: [f32; 2],
    grid_scale_inv: [f32; 2],
    #[br(temp)]
    num_quads: u32,
    #[br(count=num_quads)]
    pub quads: Vec<QUAD>,
    final_quad: Optional<QUAD>,
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
    pub num_objs: u32,
    #[br(count=num_objs,args{inner: (version,)})]
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
            for map in mat.maps.iter().flat_map(|m| m.value.as_ref()) {
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
    #[br(magic = b"SM3\0")]
    SM3(SM3),
    #[br(magic = b"CM3\0")]
    CM3(CM3),
    #[br(magic = b"DUM\0")]
    DUM(DUM),
    #[br(magic = b"AMC\0")]
    AMC(AMC),
    #[br(magic = b"EMI\0")]
    EMI(EMI),
}

impl Data {
    pub(crate) fn dependencies(&self) -> Vec<String> {
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
    let ret = fh
        .read_le()
        .context(format!("Error parsing {}", path.as_str()))?;
    let pos = fh.stream_position().unwrap_or(0);
    // eprintln!("Read {} bytes from {}", pos, path.as_str());
    let mut buffer = [0u8; 0x1000];
    if let Ok(n) = fh.read(&mut buffer)
        && n != 0
    {
        eprintln!("Rest:\n{}", rhexdumps!(&buffer[..n], pos));
    };
    while let Ok(n) = fh.read(&mut buffer)
        && n != 0
    {
        rest_size += n;
    }
    // eprintln!("+{rest_size} unparsed bytes");
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
    pub collisions: AMC,
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
        let amc_path = map_path.join("map3d.amc")?;
        let config = load_ini(&map_path.join("map3d.ini")?);
        let moredummies = load_ini(&map_path.join("moredummies.ini")?);
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
        let Data::AMC(collisions) = parse_file(&amc_path)? else {
            bail!(
                "Failed to parse AMC at {amc_path}",
                amc_path = amc_path.as_str()
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
                    warn!("Failed to resolve dependency: {}", dep);
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
            collisions,
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

pub(crate) fn resolve_dep(dep: &str, asset_path: &VfsPath, config: &IniData) -> Option<VfsPath> {
    let root = asset_path.root();
    const EXTS: &[&str] = &["png", "bmp", "dds", "tga", "alpha.dds"];
    let tex_path = config
        .get("model")
        .and_then(|config| config.get("texturepath"))
        .cloned()
        .flatten()
        .map(|path| root.join(path.trim_matches('/').to_lowercase()).unwrap())
        .map(|path| ancestors(&path))
        .unwrap_or_default();
    for path in ancestors(asset_path)
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
                if let Ok(path) = root.join(&path)
                    && path.exists().unwrap_or(false)
                {
                    return Some(path);
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
    use std::{
        collections::{BTreeMap, HashMap},
        path::Path,
    };

    use color_eyre::eyre::bail;
    use vfs::{SeekAndRead, VfsPath};

    use super::{ParsedData, PathBuf, Result, Serialize};
    use crate::packed_vfs::MultiPack;

    #[derive(Serialize, Debug)]
    pub(crate) struct Entry {
        pub path: String,
        pub size: u64,
        pub is_file: bool,
    }

    #[derive(Debug)]
    pub(crate) struct MultiPackFS {
        pub fs: VfsPath,
        current: Vec<String>,
    }

    impl MultiPackFS {
        pub(crate) fn new<P: AsRef<Path>>(files: &[P]) -> Result<Self> {
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
                    // println!("File: {}", path.as_str());
                    ParsedData::Data(super::parse_file(&path)?)
                }
                vfs::VfsFileType::Directory => {
                    // println!("Level directory: {}", path.as_str());
                    ParsedData::Level(super::Level::load(&path)?)
                }
            };
            Ok(data)
        }
    }
}
