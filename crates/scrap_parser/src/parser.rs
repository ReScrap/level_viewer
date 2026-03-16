#![allow(clippy::upper_case_acronyms, non_camel_case_types)]
use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::{Debug, Display},
    io::{BufReader, Cursor, Read, Seek, Write},
    ops::{Deref, Index},
    path::{Path, PathBuf},
};

use bilge::prelude::*;
use binrw::{args, helpers::until_exclusive, meta::WriteEndian, prelude::*};
use chrono::{DateTime, Utc, naive::serde::ts_microseconds_option::deserialize};
use color_eyre::eyre::{Context, Result, anyhow, bail};
use configparser::ini::{Ini, IniDefault};
use encoding::{DecoderTrap, EncoderTrap, Encoding, all::WINDOWS_1252};
use enum_iterator::Sequence;
use indexmap::IndexMap;
use log::warn;
use num_derive::ToPrimitive;
use num_traits::ToPrimitive;
use rhexdump::rhexdumps;
use serde::{Deserialize, Serialize};
use vfs::VfsPath;
use walkdir::WalkDir;

pub type IniData = IndexMap<String, IndexMap<String, Option<String>>>;

fn path_len(path: &str) -> Result<u32> {
    Ok(WINDOWS_1252
        .encode(path, EncoderTrap::Strict)
        .map_err(|e| anyhow!("Failed to encode string: {e}"))?
        .len()
        .try_into()?)
}

fn b2s(b: &[u8]) -> Result<String> {
    WINDOWS_1252
        .decode(b, DecoderTrap::Strict)
        .map_err(|e| anyhow!("Failed to decode: {e}"))
}
fn s2b(s: &str) -> Result<Vec<u8>> {
    WINDOWS_1252
        .encode(s, EncoderTrap::Strict)
        .map_err(|e| anyhow!("Failed to encode: {e}"))
}

#[binrw]
#[derive(Serialize, Debug, Clone, Deserialize, facet::Facet)]
pub struct PackedEntry {
    #[br(temp)]
    #[bw(try_calc=path_len(path))]
    pub path_len: u32,
    #[br(count=path_len, try_map=|bytes: Vec<u8>| b2s(&bytes))]
    #[bw(try_map=|s: &String| s2b(s))]
    pub path: String,
    pub size: u32,
    pub offset: u32,
}

#[binrw]
#[brw(magic = b"BFPK")]
#[derive(Serialize, Debug, Deserialize, facet::Facet)]
pub struct PackedHeader {
    #[br(temp,assert(version==0))]
    #[bw(calc = 0u32)]
    pub version: u32,
    #[br(temp)]
    #[bw(try_calc = files.len().try_into())]
    pub num_files: u32,
    #[br(count=num_files)]
    pub files: Vec<PackedEntry>,
}

impl PackedHeader {
    pub fn size(&self) -> usize {
        use binrw::BinWrite;
        let mut cursor = Cursor::new(Vec::new());
        self.write_le(&mut cursor).unwrap();
        cursor.into_inner().len()
    }
}

#[binrw]
#[derive(Serialize, Debug, Deserialize, facet::Facet)]
pub struct Table<
    const SIZE: u32,
    T: for<'a> BinRead<Args<'a> = ()> + for<'a> BinWrite<Args<'a> = ()> + 'static,
> {
    #[bw(try_calc = data.len().try_into())]
    num_entries: u32,
    #[br(assert(entry_size==SIZE))]
    #[bw(calc = SIZE)]
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

#[binrw]
#[bw(import_raw(compute: bool))]
#[derive(Clone, facet::Facet)]
pub struct Optional<T: for<'a> BinRead<Args<'a> = ()> + for<'a> BinWrite<Args<'a> = bool>> {
    #[br(temp)]
    #[bw(calc = u32::from(value.is_some()))]
    has_value: u32,
    #[br(if(has_value!=0))]
    #[bw(if(has_value!=0), args_raw = compute)]
    value: Option<T>,
}

impl<T> Optional<T>
where
    T: for<'a> BinRead<Args<'a> = ()> + for<'a> BinWrite<Args<'a> = bool>,
{
    pub fn get(&self) -> Option<&T> {
        self.value.as_ref()
    }
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.value.as_mut()
    }
}

impl<T> Default for Optional<T>
where
    T: for<'a> BinRead<Args<'a> = ()> + for<'a> BinWrite<Args<'a> = bool>,
{
    fn default() -> Self {
        Self { value: None }
    }
}

impl<T: for<'a> BinRead<Args<'a> = ()> + for<'a> BinWrite<Args<'a> = bool> + Debug> Debug
    for Optional<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value.fmt(f)
    }
}

impl<T: for<'a> BinRead<Args<'a> = ()> + for<'a> BinWrite<Args<'a> = bool> + Serialize> Serialize
    for Optional<T>
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.value.serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for Optional<T>
where
    T: Deserialize<'de> + for<'a> BinRead<Args<'a> = ()> + for<'a> BinWrite<Args<'a> = bool>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Option::<T>::deserialize(deserializer)?;
        Ok(Optional { value })
    }
}

fn encode_pascal_string(string: &str) -> Vec<u8> {
    if string.is_empty() {
        return vec![];
    }
    // These strings are Windows-1252 encoded in packed/game files.
    // Use a single-byte encoding to avoid UTF-8 expansion while preserving names.
    let mut bytes = WINDOWS_1252
        .encode(string, EncoderTrap::Replace)
        .expect("windows-1252 encode with replacement should not fail");
    if !bytes.ends_with(&[0]) {
        bytes.push(0)
    }
    bytes
}

fn decode_pascal_string(bytes: &[u8]) -> String {
    WINDOWS_1252
        .decode(bytes, DecoderTrap::Replace)
        .expect("windows-1252 decode with replacement should not fail")
}

#[cfg(test)]
mod string_encoding_tests {
    use std::io::Cursor;

    use binrw::{BinReaderExt, BinWrite};

    use super::{Data, decode_pascal_string, encode_pascal_string};

    #[test]
    fn pascal_string_roundtrips_single_byte_high_ascii() {
        let raw = [0xF1, 0x00]; // "ñ\0" in game data
        let decoded = decode_pascal_string(&raw);
        let encoded = encode_pascal_string(&decoded);
        assert_eq!(encoded, raw);
    }

    #[test]
    fn pascal_string_encoding_does_not_utf8_expand() {
        let encoded = encode_pascal_string("Señor\0");
        assert_eq!(encoded, b"Se\xF1or\0");
    }

    #[test]
    fn pascal_string_decodes_cp1252_symbols() {
        let raw = [0x80, 0x99, 0x00];
        let decoded = decode_pascal_string(&raw);
        assert_eq!(decoded, "€™\0");
        assert_eq!(encode_pascal_string(&decoded), raw);
    }

    #[test]
    fn empty_pascal_string_encodes_as_single_nul() {
        let encoded = encode_pascal_string("");
        assert_eq!(encoded, Vec::<u8>::new());
    }

    #[test]
    fn debug_roundtrip_local_orig_bin() {
        for file in ["orig.bin", "buffer.bin"] {
            let Ok(bytes) = std::fs::read(file) else {
                continue;
            };
            let mut cur = Cursor::new(bytes);
            let data: Data = cur.read_le().unwrap();
            let json = serde_json::to_string(&data).unwrap();
            if let Err(err) = serde_json::from_str::<Data>(&json) {
                let line = err.line();
                let col = err.column();
                let snippet = json
                    .lines()
                    .nth(line.saturating_sub(1))
                    .map(|l| {
                        let start = col.saturating_sub(80);
                        let end = (col + 80).min(l.len());
                        l.get(start..end).unwrap_or(l)
                    })
                    .unwrap_or("");
                panic!(
                    "{file}: json roundtrip failed at line {line} col {col}: {err}\ncontext: {snippet}"
                );
            }
            let mut out = Cursor::new(Vec::new());
            data.write_le(&mut out).unwrap();
        }
    }

    #[test]
    fn debug_roundtrip_local_mmission_sm3() {
        let file = "mmission.sm3";
        let Ok(bytes) = std::fs::read(file) else {
            return;
        };
        let mut cur = Cursor::new(bytes.clone());
        let data: Data = cur.read_le().unwrap();
        let json = serde_json::to_string(&data).unwrap();
        let data: Data = serde_json::from_str(&json).unwrap();
        let mut out = Cursor::new(Vec::new());
        data.write_le(&mut out).unwrap();
        let out = out.into_inner();
        assert_eq!(out.len(), bytes.len(), "{file}: length mismatch");
        assert_eq!(out, bytes, "{file}: binary mismatch");
    }

    #[test]
    fn md3d_skin_weights_accept_null_and_roundtrip() {
        #[derive(serde::Serialize, serde::Deserialize, facet::Facet)]
        struct Wrap {
            skin: super::MD3D_Skin,
        }

        let json =
            r#"{"skin":{"influence_count":3,"bone_indices":[1,2,3],"weights":[0.5,null,0.5]}}"#;
        let parsed: Wrap = serde_json::from_str(json).unwrap();
        assert!(parsed.skin.weights[1].is_nan());

        let out = serde_json::to_string(&parsed).unwrap();
        let out_v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(out_v["skin"]["weights"][1], serde_json::Value::Null);
    }

    #[test]
    fn tex_coords_accept_null_and_roundtrip() {
        #[derive(serde::Serialize, serde::Deserialize, facet::Facet)]
        struct Wrap {
            tex: super::TexCoords,
        }

        let json = r#"{"tex":[0.25,null]}"#;
        let parsed: Wrap = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.tex.0[0], 0.25);
        assert!(parsed.tex.0[1].is_nan());

        let out = serde_json::to_string(&parsed).unwrap();
        let out_v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(out_v["tex"][1], serde_json::Value::Null);
    }
}

#[binrw]
#[derive(Clone, facet::Facet)]
pub struct PascalString {
    #[br(temp)]
    #[bw(try_calc = encode_pascal_string(string).len().try_into())]
    length: u32,
    #[br(count=length, map=|bytes: Vec<u8>| {
        decode_pascal_string(&bytes)
    })]
    #[bw(map = |value: &String| encode_pascal_string(value))]
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

impl<'de> Deserialize<'de> for PascalString {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let string = String::deserialize(deserializer)?;
        Ok(Self { string })
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

#[binrw]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct IniSection {
    #[br(temp)]
    #[bw(try_calc = sections.len().try_into())]
    num_lines: u32,
    #[br(count=num_lines)]
    pub sections: Vec<PascalString>,
}

/// Configuration data
#[binrw]
#[brw(magic = b"INI\0")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, facet::Facet)]
pub struct INI {
    #[br(temp)]
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(temp)]
    #[bw(try_calc = sections.len().try_into())]
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
    pub fn data(&self) -> Ini {
        parse_ini(&format!("{self}"))
    }
}

impl std::fmt::Display for INI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for section in &self.sections {
            for line in &section.sections {
                writeln!(f, "{}", line.string.trim_end_matches(['\n', '\0']))?;
            }
        }
        Ok(())
    }
}

const EMPTY_LINE_KEY_PREFIX: &str = "\u{0}__EMPTY_LINE__";
const DUPLICATE_KEY_PREFIX: &str = "\u{0}__DUP_KEY__";
const DUPLICATE_KEY_SEPARATOR: char = '\u{0}';

fn make_duplicate_key_marker(duplicate_index: u32, original_key: &str) -> String {
    format!("{DUPLICATE_KEY_PREFIX}{duplicate_index}{DUPLICATE_KEY_SEPARATOR}{original_key}")
}

fn parse_duplicate_key_marker(marker: &str) -> Option<&str> {
    let rest = marker.strip_prefix(DUPLICATE_KEY_PREFIX)?;
    let (_, original_key) = rest.split_once(DUPLICATE_KEY_SEPARATOR)?;
    Some(original_key)
}

impl Serialize for INI {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut out: IniData = IndexMap::new();
        let mut section_name = String::new();
        let mut empty_line_idx = 0u32;
        let mut duplicate_key_count: HashMap<String, u32> = HashMap::new();

        for line in self
            .sections
            .iter()
            .flat_map(|section| section.sections.iter())
        {
            let line = line.string.trim_end_matches(['\r', '\n', '\0']);

            if line.starts_with('[') && line.ends_with(']') && line.len() >= 2 {
                section_name = line[1..line.len() - 1].to_owned();
                out.entry(section_name.clone()).or_default();
                empty_line_idx = 0;
                duplicate_key_count.clear();
                continue;
            }
            if line.is_empty() {
                let section = out.entry(section_name.clone()).or_default();
                section.insert(
                    format!("{EMPTY_LINE_KEY_PREFIX}{empty_line_idx}"),
                    Some(String::new()),
                );
                empty_line_idx += 1;
                continue;
            }

            let section = out.entry(section_name.clone()).or_default();
            if let Some((key, value)) = line.split_once('=') {
                let key = key.to_owned();
                if section.contains_key(&key) {
                    let duplicate_idx = duplicate_key_count.entry(key.clone()).or_insert(0);
                    let marker = make_duplicate_key_marker(*duplicate_idx, &key);
                    *duplicate_idx += 1;
                    section.insert(marker, Some(value.to_owned()));
                } else {
                    section.insert(key, Some(value.to_owned()));
                }
            } else {
                let key = line.to_owned();
                if section.contains_key(&key) {
                    let duplicate_idx = duplicate_key_count.entry(key.clone()).or_insert(0);
                    let marker = make_duplicate_key_marker(*duplicate_idx, &key);
                    *duplicate_idx += 1;
                    section.insert(marker, None);
                } else {
                    section.insert(key, None);
                }
            }
        }

        out.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for INI {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let map = IniData::deserialize(deserializer)?;
        let mut lines = Vec::new();

        for (section, values) in map {
            if !section.is_empty() {
                lines.push(PascalString {
                    string: format!("[{section}]"),
                });
            }

            for (key, value) in values {
                if key.starts_with(EMPTY_LINE_KEY_PREFIX) {
                    lines.push(PascalString {
                        // INI empty lines are encoded as an explicit NUL-terminated empty string.
                        string: "\0".to_owned(),
                    });
                    continue;
                }
                let key = parse_duplicate_key_marker(&key).unwrap_or(&key);
                let string = match value {
                    Some(value) => format!("{key}={value}"),
                    None => key.to_owned(),
                };
                lines.push(PascalString { string });
            }
        }

        Ok(Self {
            sections: vec![IniSection { sections: lines }],
        })
    }
}

#[cfg(test)]
mod ini_roundtrip_tests {
    use super::{
        DUPLICATE_KEY_PREFIX, EMPTY_LINE_KEY_PREFIX, INI, IniSection, PascalString,
        parse_duplicate_key_marker,
    };

    #[test]
    fn ini_serializes_to_hashmap_with_empty_default_section() {
        let ini = INI {
            sections: vec![
                IniSection {
                    sections: vec![PascalString {
                        string: "  MiXeDKey  =  Value  ".to_owned(),
                    }],
                },
                IniSection {
                    sections: vec![
                        PascalString {
                            string: "[MiXeDSection]".to_owned(),
                        },
                        PascalString {
                            string: "  MiXeDKey  =  Other  ".to_owned(),
                        },
                    ],
                },
            ],
        };

        let value = serde_json::to_value(&ini).unwrap();
        let obj = value.as_object().unwrap();

        assert!(obj.contains_key(""));
        assert!(obj.contains_key("MiXeDSection"));
        assert_eq!(obj[""]["  MiXeDKey  "], "  Value  ");
        assert_eq!(obj["MiXeDSection"]["  MiXeDKey  "], "  Other  ");
    }

    #[test]
    fn ini_deserializes_hashmap_with_empty_default_section() {
        let json = r#"{
            "": { "KeyA": "ValA", "FlagOnly": null },
            "SectionB": { "KeyB": "ValB" }
        }"#;
        let ini: INI = serde_json::from_str(json).unwrap();
        let lines: Vec<&str> = ini.sections[0]
            .sections
            .iter()
            .map(|line| line.string.as_str())
            .collect();

        assert!(lines.contains(&"KeyA=ValA"));
        assert!(lines.contains(&"FlagOnly"));
        assert!(lines.contains(&"[SectionB]"));
        assert!(lines.contains(&"KeyB=ValB"));
    }

    #[test]
    fn ini_serialization_preserves_section_and_key_order() {
        let ini = INI {
            sections: vec![IniSection {
                sections: vec![
                    PascalString {
                        string: "k0=v0".to_owned(),
                    },
                    PascalString {
                        string: "[Zeta]".to_owned(),
                    },
                    PascalString {
                        string: "a=A".to_owned(),
                    },
                    PascalString {
                        string: "b=B".to_owned(),
                    },
                    PascalString {
                        string: "[Alpha]".to_owned(),
                    },
                    PascalString {
                        string: "x=X".to_owned(),
                    },
                    PascalString {
                        string: "y=Y".to_owned(),
                    },
                ],
            }],
        };

        let value = serde_json::to_value(&ini).unwrap();
        let obj = value.as_object().unwrap();
        let sections: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
        assert_eq!(sections, vec!["", "Zeta", "Alpha"]);

        let default_keys: Vec<&str> = obj[""]
            .as_object()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect();
        let zeta_keys: Vec<&str> = obj["Zeta"]
            .as_object()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect();
        let alpha_keys: Vec<&str> = obj["Alpha"]
            .as_object()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect();
        assert_eq!(default_keys, vec!["k0"]);
        assert_eq!(zeta_keys, vec!["a", "b"]);
        assert_eq!(alpha_keys, vec!["x", "y"]);
    }

    #[test]
    fn ini_deserialization_preserves_section_and_key_order() {
        let json = r#"{
            "": { "first": "1", "second": "2" },
            "Zeta": { "a": "A", "b": "B" },
            "Alpha": { "x": "X", "y": "Y" }
        }"#;
        let ini: INI = serde_json::from_str(json).unwrap();
        let lines: Vec<&str> = ini.sections[0]
            .sections
            .iter()
            .map(|line| line.string.as_str())
            .collect();

        assert_eq!(
            lines,
            vec![
                "first=1", "second=2", "[Zeta]", "a=A", "b=B", "[Alpha]", "x=X", "y=Y"
            ]
        );
    }

    #[test]
    fn ini_serialization_preserves_empty_lines_with_markers() {
        let ini = INI {
            sections: vec![IniSection {
                sections: vec![
                    PascalString {
                        string: "a=1".to_owned(),
                    },
                    PascalString {
                        string: String::new(),
                    },
                    PascalString {
                        string: "b=2".to_owned(),
                    },
                ],
            }],
        };

        let value = serde_json::to_value(&ini).unwrap();
        let obj = value[""].as_object().unwrap();
        let keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
        assert_eq!(keys, vec!["a", &format!("{EMPTY_LINE_KEY_PREFIX}0"), "b"]);
        assert_eq!(obj[&format!("{EMPTY_LINE_KEY_PREFIX}0")], "");
    }

    #[test]
    fn ini_deserialization_restores_empty_lines_from_markers() {
        let mut section = serde_json::Map::new();
        section.insert("a".to_owned(), serde_json::Value::String("1".to_owned()));
        section.insert(
            format!("{EMPTY_LINE_KEY_PREFIX}0"),
            serde_json::Value::String(String::new()),
        );
        section.insert("b".to_owned(), serde_json::Value::String("2".to_owned()));
        let mut root = serde_json::Map::new();
        root.insert("".to_owned(), serde_json::Value::Object(section));

        let ini: INI = serde_json::from_value(serde_json::Value::Object(root)).unwrap();
        let lines: Vec<&str> = ini.sections[0]
            .sections
            .iter()
            .map(|line| line.string.as_str())
            .collect();

        assert_eq!(lines, vec!["a=1", "\0", "b=2"]);
    }

    #[test]
    fn ini_serialization_preserves_duplicate_keys_with_markers() {
        let ini = INI {
            sections: vec![IniSection {
                sections: vec![
                    PascalString {
                        string: "dup=1".to_owned(),
                    },
                    PascalString {
                        string: "dup=2".to_owned(),
                    },
                ],
            }],
        };

        let value = serde_json::to_value(&ini).unwrap();
        let obj = value[""].as_object().unwrap();
        let keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
        assert_eq!(keys[0], "dup");
        assert!(keys[1].starts_with(DUPLICATE_KEY_PREFIX));
        assert_eq!(parse_duplicate_key_marker(keys[1]), Some("dup"));
        assert_eq!(obj["dup"], "1");
        assert_eq!(obj[keys[1]], "2");
    }

    #[test]
    fn ini_deserialization_restores_duplicate_keys_from_markers() {
        let mut section = serde_json::Map::new();
        section.insert("dup".to_owned(), serde_json::Value::String("1".to_owned()));
        section.insert(
            format!("{DUPLICATE_KEY_PREFIX}0\u{0}dup"),
            serde_json::Value::String("2".to_owned()),
        );
        let mut root = serde_json::Map::new();
        root.insert("".to_owned(), serde_json::Value::Object(section));

        let ini: INI = serde_json::from_value(serde_json::Value::Object(root)).unwrap();
        let lines: Vec<&str> = ini.sections[0]
            .sections
            .iter()
            .map(|line| line.string.as_str())
            .collect();

        assert_eq!(lines, vec!["dup=1", "dup=2"]);
    }
}

#[binrw]
#[derive(Debug, Serialize, Clone, Deserialize, facet::Facet)]
pub struct RGBA {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl RGBA {
    pub fn as_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a].map(|v| (v as f32) / 255.0)
    }
}

#[binrw]
#[derive(Debug, Serialize, Clone, Deserialize, facet::Facet)]
#[br(import(n_dims: usize))]
#[bw(import(n_dims: usize))]
pub struct TexCoords(
    #[serde(with = "nan_f32_vec_serde")]
    #[br(count=n_dims)]
    pub Vec<f32>,
);

mod nan_f32_vec_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S>(value: &Vec<f32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let out: Vec<Option<f32>> = value
            .iter()
            .map(|v| if v.is_finite() { Some(*v) } else { None })
            .collect();
        out.serialize(serializer)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Vec::<Option<f32>>::deserialize(deserializer)?;
        Ok(value.into_iter().map(|v| v.unwrap_or(f32::NAN)).collect())
    }
}

#[binrw]
#[derive(Debug, Serialize, Clone, Deserialize, facet::Facet)]
#[br(import(vert_fmt: FVF))]
#[bw(import(vert_fmt: FVF))]
// https://github.com/elishacloud/dxwrapper/blob/23ffb74c4c93c4c760bb5f1de347a0b039897210/ddraw/IDirect3DDeviceX.cpp#L2642
pub struct Vertex {
    pub xyz: [f32; 3],
    // #[br(if(vert_fmt.pos()==Pos::XYZRHW))] // seems to be unused
    // rhw: Option<f32>,
    #[brw(if(vert_fmt.normal()))]
    pub normal: Option<[f32; 3]>,
    #[brw(if(vert_fmt.point_size()))]
    pub point_size: Option<[f32; 3]>,
    #[brw(if(vert_fmt.diffuse()))]
    pub diffuse: Option<RGBA>,
    #[brw(if(vert_fmt.specular()))]
    pub specular: Option<RGBA>,
    #[brw(if(vert_fmt.tex_count().value()>=1), args (vert_fmt.tex_dims(0),))]
    pub tex_0: Option<TexCoords>,
    #[brw(if(vert_fmt.tex_count().value()>=2), args (vert_fmt.tex_dims(1),))]
    pub tex_1: Option<TexCoords>,
    #[brw(if(vert_fmt.tex_count().value()>=3), args (vert_fmt.tex_dims(2),))]
    pub tex_2: Option<TexCoords>,
    #[brw(if(vert_fmt.tex_count().value()>=4), args (vert_fmt.tex_dims(3),))]
    pub tex_3: Option<TexCoords>,
    #[brw(if(vert_fmt.tex_count().value()>=5), args (vert_fmt.tex_dims(4),))]
    pub tex_4: Option<TexCoords>,
    #[brw(if(vert_fmt.tex_count().value()>=6), args (vert_fmt.tex_dims(5),))]
    pub tex_5: Option<TexCoords>,
    #[brw(if(vert_fmt.tex_count().value()>=7), args (vert_fmt.tex_dims(6),))]
    pub tex_6: Option<TexCoords>,
    #[brw(if(vert_fmt.tex_count().value()>=8), args (vert_fmt.tex_dims(7),))]
    pub tex_7: Option<TexCoords>,
}

#[bitsize(3)]
#[derive(Debug, Serialize, PartialEq, Eq, TryFromBits, Deserialize, facet::Facet)]
#[repr(u8)]
pub enum Pos {
    XYZ,
    XYZRHW,
    XYZB1,
    XYZB2,
    XYZB3,
    XYZB4,
    XYZB5,
}

#[bitsize(32)]
#[derive(
    DebugBits,
    Serialize,
    Copy,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    TryFromBits,
    Deserialize,
    facet::Facet,
)]
pub struct FVF {
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

#[binrw]
#[br(import(fmt_id: u32))]
#[bw(import(fmt_id: u32))]
#[derive(Debug, Serialize, Clone, Deserialize, facet::Facet)]
pub struct LFVFInner {
    #[br(try_map=|v:  u32| vertex_format_from_id(fmt_id,v))]
    #[bw(map = |value: &FVF| u32::from(*value))]
    pub vert_fmt: FVF,
    #[br(assert(vertex_size_from_id(fmt_id).ok()==Some(vert_size)))]
    #[bw(try_calc = vertex_size_from_id(fmt_id))]
    pub vert_size: u32,
    #[br(temp)]
    #[bw(try_calc = data.len().try_into())]
    num_verts: u32,
    #[br(count=num_verts, args {inner: (vert_fmt,)})]
    #[bw(args(self.vert_fmt))]
    pub data: Vec<Vertex>,
}

#[binrw]
#[brw(magic = b"LFVF")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct LFVF {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==1,"invalid LFVF version"))]
    #[bw(calc = 1u32)]
    version: u32,
    #[br(assert((0..=0x11).contains(&fmt_id),"invalid LFVF format_id"))]
    fmt_id: u32,
    #[br(if(fmt_id!=0),args(fmt_id))]
    #[bw(if(*fmt_id!=0),args(*fmt_id))]
    pub inner: Option<LFVFInner>,
}

#[binrw]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct MD3D_Tris {
    #[bw(try_calc = tris.len().try_into())]
    num_tris: u32,
    #[br(assert(tri_size==6,"Invalid MD3D tri size"))]
    #[bw(calc = 6u32)]
    tri_size: u32,
    #[br(count=num_tris)]
    pub tris: Vec<[u16; 3]>,
}

#[binrw]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct MD3D_TriSeg {
    plane_distance_xor: u32,
    pub normal: [f32; 3],
}

#[binrw]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct MD3D_Segment {
    triangle_a_index: i16,
    triangle_b_index: i16,
    vertex_a_index: u16,
    vertex_b_index: u16,
}

#[binrw]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct MD3D_Skin {
    pub influence_count: u8,
    pub bone_indices: [u8; 3],
    #[serde(with = "skin_weights_serde")]
    pub weights: [f32; 3],
}

mod skin_weights_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S>(
        value: &[f32; 3],
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let out: [Option<f32>; 3] = value.map(|v| if v.is_finite() { Some(v) } else { None });
        out.serialize(serializer)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> std::result::Result<[f32; 3], D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = <[Option<f32>; 3]>::deserialize(deserializer)?;
        Ok(value.map(|v| v.unwrap_or(f32::NAN)))
    }
}

mod scale_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S>(value: &f32, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert NaN to null, finite values to themselves
        if value.is_finite() {
            serializer.serialize_f32(*value)
        } else {
            serializer.serialize_none()
        }
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<f32, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Option::<f32>::deserialize(deserializer)?;
        Ok(value.unwrap_or(f32::NAN))
    }
}

#[binrw]
#[brw(magic = b"MD3D")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct MD3D {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==1,"Invalid MD3D version"))]
    #[bw(calc = 1u32)]
    version: u32,
    pub name: PascalString,
    pub tris: MD3D_Tris,
    #[bw(args_raw = compute)]
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
    // Set from conversion flags bits 0x4 and 0x8 respectively; either one switches
    // non-envmap meshes to VERT_TEX1_B1 instead of VERT_TEX1 during D3D conversion.
    pub tex1_b1_flag_from_convert_0x4: u32,
    pub tex1_b1_flag_from_convert_0x8: u32,
    pub subtree_bbox: [[f32; 3]; 2],
    pub local_bbox: [[f32; 3]; 2],
    pub bbox_center: [f32; 3],
    #[bw(calc = u32::from(child.is_some()))]
    has_child: u32,
    #[br(if(has_child!=0))]
    #[bw(if(has_child!=0), args_raw = compute)]
    pub child: Option<Box<MD3D>>,
}

#[binread]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
#[serde(tag = "type")]
#[repr(u32)]
pub enum NodeData {
    #[brw(magic = 0x0u32)]
    Dummy,
    #[brw(magic = 0xa1_00_00_01_u32)]
    TriangleMesh,
    #[brw(magic = 0xa1_00_00_02_u32)]
    D3DMesh(Box<MD3D>),
    #[brw(magic = 0xa2_00_00_04_u32)]
    Camera(CAM),
    #[brw(magic = 0xa3_00_00_08_u32)]
    Light(LUZ),
    #[brw(magic = 0xa4_00_00_10_u32)]
    Ground(SUEL),
    #[brw(magic = 0xa5_00_00_20_u32)]
    SistPart,
    #[brw(magic = 0xa6_00_00_40_u32)]
    Graphic3D(SPR3),
    #[brw(magic = 0xa6_00_00_80_u32)]
    Flare,
    #[brw(magic = 0xa7_00_01_00u32)]
    Portal(PORT),
}

impl BinWrite for NodeData {
    type Args<'a> = bool;

    fn write_options<W: Write + Seek>(
        &self,
        writer: &mut W,
        endian: binrw::Endian,
        compute: Self::Args<'_>,
    ) -> BinResult<()> {
        match self {
            NodeData::Dummy => 0x0u32.write_options(writer, endian, ())?,
            NodeData::TriangleMesh => 0xa1_00_00_01_u32.write_options(writer, endian, ())?,
            NodeData::D3DMesh(data) => {
                0xa1_00_00_02_u32.write_options(writer, endian, ())?;
                data.write_options(writer, endian, compute)?;
            }
            NodeData::Camera(data) => {
                0xa2_00_00_04_u32.write_options(writer, endian, ())?;
                data.write_options(writer, endian, compute)?;
            }
            NodeData::Light(data) => {
                0xa3_00_00_08_u32.write_options(writer, endian, ())?;
                data.write_options(writer, endian, compute)?;
            }
            NodeData::Ground(data) => {
                0xa4_00_00_10_u32.write_options(writer, endian, ())?;
                data.write_options(writer, endian, compute)?;
            }
            NodeData::SistPart => 0xa5_00_00_20_u32.write_options(writer, endian, ())?,
            NodeData::Graphic3D(data) => {
                0xa6_00_00_40_u32.write_options(writer, endian, ())?;
                data.write_options(writer, endian, compute)?;
            }
            NodeData::Flare => 0xa6_00_00_80_u32.write_options(writer, endian, ())?,
            NodeData::Portal(data) => {
                0xa7_00_01_00u32.write_options(writer, endian, ())?;
                data.write_options(writer, endian, compute)?;
            }
        }
        Ok(())
    }
}

#[binrw]
#[brw(magic = b"SPR3")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct SPR3 {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==1,"Invalid SPR3 version"))]
    #[bw(calc = 1u32)]
    version: u32,
    pos: [f32; 3],
    scale: [f32; 2],
    primary_map_name: PascalString,
    secondary_map_name: PascalString,
    diffuse_mod: RGBA,
}

#[binrw]
#[brw(magic = b"SUEL")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct SUEL {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==1,"Invalid SUEL version"))]
    #[bw(calc = 1u32)]
    version: u32,
    bbox: [[f32; 3]; 2],
    dims: [f32; 3],
    grid_size: [u32; 2],
    num_tris: u32,
    collision_bbox: [[f32; 3]; 2],
}

#[binrw]
#[brw(magic = b"CAM\0")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct CAM {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==1,"Invalid CAM version"))]
    #[bw(calc = 1u32)]
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

#[binrw]
#[brw(repr=u32)]
#[repr(u32)]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub enum LightType {
    Point = 5000,
    Spot = 5001,
    Directional = 5002,
}

#[binrw]
#[brw(magic = b"LUZ\0")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct LUZ {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==1,"Invalid LUZ version"))]
    #[bw(calc = 1u32)]
    version: u32,
    sector: u32,
    pub light_type: LightType,
    pub shadows: u8,
    pub pos: [f32; 3],
    pub dir: [f32; 3],
    pub color: RGBA,
    pub power: f32,
    pub attenuation: [f32; 2],
    pub hotspot: f32,
    pub falloff: f32,
    pub mult: f32,
    pub radiosity_coeff: f32,
    #[br(map = |v: u32| v != 0)]
    #[bw(map = |v: &bool| if *v { 1u32 } else { 0u32 })]
    pub active: bool,
}

#[binrw]
#[brw(magic = b"PORT")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct PORT {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==1,"Invalid PORT version"))]
    #[bw(calc = 1u32)]
    version: u32,
    side_node_indices: [i32; 2],
    width: f32,
    height: f32,
}

#[derive(
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Sequence,
    Serialize,
    ToPrimitive,
    Deserialize,
    facet::Facet,
)]
#[repr(u8)]
pub enum NodeFlags {
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
    DUDV_PASS,
}

fn parse_node_flags(flags: u32) -> BTreeSet<NodeFlags> {
    let inv_flag_mask = !enum_iterator::all::<NodeFlags>()
        .fold(0u32, |acc, flag| acc | (1 << flag.to_u8().unwrap_or(0xff)));
    assert_eq!(flags & inv_flag_mask, 0);
    enum_iterator::all::<NodeFlags>()
        .filter_map(|flag| ((flags & (1 << flag.to_u8().unwrap_or(0xff))) != 0).then_some(flag))
        .collect()
}

fn encode_node_flags(flags: &BTreeSet<NodeFlags>) -> u32 {
    flags
        .iter()
        .fold(0u32, |acc, flag| acc | (1 << flag.to_u8().unwrap_or(0xff)))
}

#[binrw]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct Node {
    pub object_index: i32,
    pub table_index: i32,
    pub node_xref: i32,
    #[br(map=parse_node_flags)]
    #[bw(map=encode_node_flags)]
    pub flags: BTreeSet<NodeFlags>,
    pub ani_mask: i32,
    pub name: PascalString,
    pub parent: PascalString,
    pub pos_offset: [f32; 3],
    pub rot: [f32; 4],
    #[serde(with = "scale_serde")]
    pub scale: f32,
    pub transform_world: [[f32; 4]; 4], // 0x40 4x4 Matrix
    pub transform_local: [[f32; 4]; 4], // 0x40 4x4 Matrix
    pub rest_rot: [f32; 4],
    pub axis_scale: [f32; 3],
    #[bw(args_raw = compute)]
    pub info: Optional<INI>,
    #[bw(args_raw = compute)]
    pub content: Optional<NodeData>,
}

#[binrw]
#[brw(magic = b"MAP\0")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Clone, Deserialize, facet::Facet)]
pub struct MAP {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
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
    #[br(map = parse_texture_type)]
    #[bw(map = encode_texture_type)]
    pub texture_type: TextureType,
    pub displacement: [f32; 2],
    pub scale: [f32; 2],
    pub quantity: f32, // Bumpmap scaling
    #[br(if(version==3))]
    #[bw(if(*version==3))]
    pub uv_matrix: Option<[f32; 2]>,
    #[br(if(version==3))]
    #[bw(if(*version==3))]
    pub angle: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, facet::Facet)]
#[repr(u8)]
pub enum TextureType {
    None,
    FxLava,
    FxScroll,
    FxNewsPanel,
    Unknown(u8),
}

fn parse_texture_type(value: u8) -> TextureType {
    match value {
        0 => TextureType::None,
        1 => TextureType::FxLava,
        2 => TextureType::FxScroll,
        3 => TextureType::FxNewsPanel,
        other => TextureType::Unknown(other),
    }
}

fn encode_texture_type(value: &TextureType) -> u8 {
    match value {
        TextureType::None => 0,
        TextureType::FxLava => 1,
        TextureType::FxScroll => 2,
        TextureType::FxNewsPanel => 3,
        TextureType::Unknown(other) => *other,
    }
}

#[binrw]
#[brw(repr=u32)]
#[repr(u32)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Sequence,
    Serialize,
    ToPrimitive,
    Deserialize,
    facet::Facet,
)]

pub enum BlendMode {
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

#[binrw]
#[brw(repr=u32)]
#[repr(u32)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Sequence,
    Serialize,
    ToPrimitive,
    Deserialize,
    facet::Facet,
)]
pub enum CmpFunc {
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

#[derive(
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Sequence,
    Serialize,
    ToPrimitive,
    Clone,
    Deserialize,
    facet::Facet,
)]
#[repr(u8)]
pub enum MatPropAttrib {
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
    NO_ALPHA_BLEND = 10,
    ZBIAS = 11,
    TRANSP_ONEONE = 12,
    UNUSED_13 = 13,
    XSHADOW = 14,
}

fn parse_mat_prop_flags(flags: u16) -> BTreeSet<MatPropAttrib> {
    let inv_flag_mask = !enum_iterator::all::<MatPropAttrib>()
        .fold(0u16, |acc, flag| acc | (1 << flag.to_u8().unwrap_or(0xff)));
    assert_eq!(flags & inv_flag_mask, 0);
    enum_iterator::all::<MatPropAttrib>()
        .filter_map(|flag| ((flags & (1 << flag.to_u8().unwrap_or(0xff))) != 0).then_some(flag))
        .collect()
}

fn encode_mat_prop_flags(flags: &BTreeSet<MatPropAttrib>) -> u16 {
    flags
        .iter()
        .fold(0u16, |acc, flag| acc | (1 << flag.to_u8().unwrap_or(0xff)))
}

#[binrw]
#[derive(Debug, Serialize, Clone, Deserialize, facet::Facet)]
pub struct MatProps {
    #[br(assert(sub_material==0))]
    pub sub_material: u32,
    pub src_blend: BlendMode,
    pub dst_blend: BlendMode,
    pub two_sided: u8,
    pub dyn_illum: u8,
    pub diffuse_alpha: u8,
    pub env_map: u8,
    #[br(map=parse_mat_prop_flags)]
    #[bw(map=encode_mat_prop_flags)]
    pub attrib: BTreeSet<MatPropAttrib>,
    pub enable_fog: u8,
    pub z_write: u8,
    pub zfunc: CmpFunc,
}

#[binrw::parser(reader, endian)]
fn parse_mat_maps(version: u32) -> BinResult<[Optional<MAP>; 5]> {
    let mut maps = std::array::from_fn(|_| Optional::<MAP>::default());
    let map_count = if version < 3 { 3 } else { 5 };
    for map in maps.iter_mut().take(map_count) {
        *map = Optional::<MAP>::read_options(reader, endian, ())?;
    }
    Ok(maps)
}

#[binrw::writer(writer, endian)]
fn write_mat_maps(maps: &[Optional<MAP>; 5], version: u32, compute: bool) -> BinResult<()> {
    let map_count = if version < 3 { 3 } else { 5 };
    for map in maps.iter().take(map_count) {
        map.write_options(writer, endian, compute)?;
    }
    Ok(())
}

#[binrw]
#[brw(magic = b"MAT\0")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Clone, Deserialize, facet::Facet)]
pub struct MAT {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert((1..=3).contains(&version),"invalid MAT version"))]
    version: u32,
    #[br(if(version>1))]
    #[bw(if(*version>1))]
    pub name: Option<PascalString>,
    pub ambient_override: RGBA,
    pub diffuse_mod: RGBA,
    pub diffuse: RGBA,
    pub specular: RGBA,
    pub glow: RGBA,
    pub spec_power: f32,
    pub spec_mult: f32,
    pub mat_props: MatProps,
    #[br(parse_with = parse_mat_maps, args(version))]
    #[bw(write_with = write_mat_maps, args(*version, compute))]
    pub maps: [Optional<MAP>; 5], // diffuse, metallic, env, bump, glow
}

#[binrw]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct LightColor {
    pub color: RGBA,
    pub intensity: f32,
}

#[binrw]
#[brw(magic = b"SCN\0")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct SCN {
    // 0x650220
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(temp,assert(version==1))]
    #[bw(calc = 1u32)]
    version: u32,
    pub model_name: PascalString,
    pub root_node: PascalString,
    #[bw(args_raw = compute)]
    pub node_props: Optional<INI>,
    pub ambient: LightColor,
    pub background: LightColor,
    pub bbox: [[f32; 3]; 2],
    // #[br(assert(collide_mesh_ref==0))]
    collide_mesh_ref: u32,
    #[bw(args_raw = compute)]
    pub user_props: Optional<INI>,
    #[br(temp)]
    #[bw(try_calc = mat.len().try_into())]
    num_materials: u32,
    #[br(count=num_materials)]
    #[bw(args_raw = compute)]
    pub mat: Vec<MAT>,
    #[br(temp,assert(nodes_section_marker==1))]
    #[bw(calc = 1u32)]
    nodes_section_marker: u32,
    #[br(temp)]
    #[bw(try_calc = nodes.len().try_into())]
    num_nodes: u32,
    #[br(count = num_nodes)] // 32
    #[bw(args_raw = compute)]
    pub nodes: Vec<Node>,
    #[bw(args_raw = compute)]
    pub ani: Optional<ANI>,
}

fn convert_timestamp(dt: u32) -> Result<DateTime<Utc>> {
    let Some(dt) = DateTime::from_timestamp(dt.into(), 0) else {
        bail!("Invalid timestamp");
    };
    Ok(dt)
}

#[binrw]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
struct VertexAnim {
    #[bw(try_calc = tris.len().try_into())]
    num_triangles: u32,
    fps: f32,
    #[br(count=num_triangles)]
    tris: Vec<[u8; 3]>,
}

#[binrw]
#[brw(magic = b"EVA\0")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct EVA {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==1,"Invalid EVA version"))]
    #[bw(calc = 1u32)]
    version: u32,
    #[bw(try_calc = verts.len().try_into())]
    num_verts: u32,
    #[br(count=num_verts)]
    #[bw(args_raw = compute)]
    verts: Vec<Optional<VertexAnim>>,
}

#[derive(
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Sequence,
    Serialize,
    ToPrimitive,
    Deserialize,
    facet::Facet,
)]
#[repr(u8)]
pub enum AniTrackType {
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

fn encode_ani_track_type(flags: &BTreeSet<AniTrackType>) -> u32 {
    flags.iter().fold(0u32, |acc, flag| acc | flag.mask())
}

fn encode_track_map(track_map: &[Option<u8>]) -> Vec<u8> {
    track_map
        .iter()
        .map(|value| value.map_or(0, |v| v.saturating_add(1)))
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize, facet::Facet)]
pub struct BlockInfo {
    pub size: usize,
    pub elem_size: usize,
    pub stream: bool,
    pub optimized: bool,
    pub track_type: AniTrackType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_header: Option<AniStreamHeader>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, facet::Facet)]
pub struct AnimFrame {
    pub pos: Option<[f32; 3]>,
    pub rot: Option<[f32; 4]>,
    pub fov: Option<f32>,
    pub color: Option<[u8; 4]>,
    pub intensity: Option<f32>,
    pub visibility: Option<u8>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, facet::Facet)]
pub struct AnimTracks {
    pub pos: Option<Vec<[f32; 3]>>,
    pub rot: Option<Vec<[f32; 4]>>,
    pub fov: Option<Vec<f32>>,
    pub color: Option<Vec<[u8; 4]>>,
    pub intensity: Option<Vec<f32>>,
    pub visibility: Option<Vec<u8>>,
}

impl AnimTracks {
    pub fn get_frame(&self, frame: usize) -> Option<AnimFrame> {
        Some(AnimFrame {
            pos: self.pos.as_ref().and_then(|v| v.get(frame).copied()),
            rot: self.rot.as_ref().and_then(|v| v.get(frame).copied()),
            fov: self.fov.as_ref().and_then(|v| v.get(frame).copied()),
            color: self.color.as_ref().and_then(|v| v.get(frame).copied()),
            intensity: self.intensity.as_ref().and_then(|v| v.get(frame).copied()),
            visibility: self.visibility.as_ref().and_then(|v| v.get(frame).copied()),
        })
    }
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

fn parse_anim_tracks_from_block_data(blocks: &mut [BlockInfo], data: &[u8]) -> Result<AnimTracks> {
    let mut offset = 0usize;
    let mut out = AnimTracks::default();
    for block in blocks {
        let end = offset
            .checked_add(block.size)
            .ok_or_else(|| anyhow!("ANI block size overflow"))?;
        let raw = data
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
            block.stream_header = Some(header);
            &raw[reader.position() as usize..]
        } else {
            block.stream_header = None;
            raw
        };
        match block.track_type {
            AniTrackType::Position => out.pos = Some(parse_track_data(payload)?),
            AniTrackType::Rotation => out.rot = Some(parse_track_data(payload)?),
            AniTrackType::FOV => out.fov = Some(parse_track_data(payload)?),
            AniTrackType::Color => out.color = Some(parse_track_data(payload)?),
            AniTrackType::Intensity => out.intensity = Some(parse_track_data(payload)?),
            AniTrackType::Visibility => out.visibility = Some(parse_track_data(payload)?),
            AniTrackType::EVA => (),
        }
    }
    Ok(out)
}

fn write_block_payload(block: &BlockInfo, nam: &NAM, tracks: &AnimTracks) -> BinResult<Vec<u8>> {
    fn write_values<T>(values: &[T]) -> BinResult<Vec<u8>>
    where
        T: for<'a> BinWrite<Args<'a> = ()>,
    {
        let mut writer = Cursor::new(Vec::new());
        for value in values {
            value.write_le(&mut writer)?;
        }
        Ok(writer.into_inner())
    }

    let payload =
        match block.track_type {
            AniTrackType::Position => {
                write_values(
                    tracks
                        .pos
                        .as_deref()
                        .ok_or_else(|| binrw::Error::AssertFail {
                            pos: 0,
                            message: "missing ANI position track payload".into(),
                        })?,
                )?
            }
            AniTrackType::Rotation => {
                write_values(
                    tracks
                        .rot
                        .as_deref()
                        .ok_or_else(|| binrw::Error::AssertFail {
                            pos: 0,
                            message: "missing ANI rotation track payload".into(),
                        })?,
                )?
            }
            AniTrackType::FOV => {
                write_values(
                    tracks
                        .fov
                        .as_deref()
                        .ok_or_else(|| binrw::Error::AssertFail {
                            pos: 0,
                            message: "missing ANI FOV track payload".into(),
                        })?,
                )?
            }
            AniTrackType::Color => {
                write_values(
                    tracks
                        .color
                        .as_deref()
                        .ok_or_else(|| binrw::Error::AssertFail {
                            pos: 0,
                            message: "missing ANI color track payload".into(),
                        })?,
                )?
            }
            AniTrackType::Intensity => {
                write_values(tracks.intensity.as_deref().ok_or_else(|| {
                    binrw::Error::AssertFail {
                        pos: 0,
                        message: "missing ANI intensity track payload".into(),
                    }
                })?)?
            }
            AniTrackType::Visibility => {
                let values =
                    tracks
                        .visibility
                        .as_deref()
                        .ok_or_else(|| binrw::Error::AssertFail {
                            pos: 0,
                            message: "missing ANI visibility track payload".into(),
                        })?;
                write_values(values)?
            }
            AniTrackType::EVA => Vec::new(),
        };

    let mut out = if block.optimized && block.stream {
        let stream_header = if let Some(header) = block.stream_header {
            header
        } else {
            let start_frame: u16 =
                nam.start_frame
                    .try_into()
                    .map_err(|_| binrw::Error::AssertFail {
                        pos: 0,
                        message: "ANI stream start_frame does not fit in u16".into(),
                    })?;
            let num_frames: u16 = nam
                .num_frames
                .try_into()
                .map_err(|_| binrw::Error::AssertFail {
                    pos: 0,
                    message: "ANI stream frame count does not fit in u16".into(),
                })?;
            AniStreamHeader {
                size: 0,
                start_frame,
                num_frames,
            }
        };
        let size: u16 = (std::mem::size_of::<AniStreamHeader>() + payload.len())
            .try_into()
            .map_err(|_| binrw::Error::AssertFail {
                pos: 0,
                message: "ANI stream block size does not fit in u16".into(),
            })?;
        let mut writer = Cursor::new(Vec::new());
        AniStreamHeader {
            size,
            ..stream_header
        }
        .write_le(&mut writer)?;
        payload.write_le(&mut writer)?;
        writer.into_inner()
    } else {
        payload
    };

    if out.len() != block.size {
        return Err(binrw::Error::AssertFail {
            pos: 0,
            message: format!(
                "ANI block payload size mismatch for {:?}: {} != {}",
                block.track_type,
                out.len(),
                block.size
            ),
        });
    }
    Ok(std::mem::take(&mut out))
}

fn ordered_ani_tracks(
    tracks: &HashMap<u8, (NAM, AnimTracks)>,
) -> BinResult<Vec<(u8, &NAM, &AnimTracks)>> {
    let mut ordered: Vec<(u8, &NAM, &AnimTracks)> = tracks
        .iter()
        .map(|(idx, (nam, track_data))| (*idx, nam, track_data))
        .collect();
    ordered.sort_unstable_by_key(|(idx, _, _)| *idx);
    for (expected, (actual, _, _)) in ordered.iter().enumerate() {
        let expected = u8::try_from(expected).map_err(|_| binrw::Error::AssertFail {
            pos: 0,
            message: "ANI supports at most 256 tracks".into(),
        })?;
        if *actual != expected {
            return Err(binrw::Error::AssertFail {
                pos: 0,
                message: format!(
                    "ANI track map must be contiguous and zero-based (missing index {expected})"
                ),
            });
        }
    }
    Ok(ordered)
}

fn build_ani_nabk_data(tracks: &HashMap<u8, (NAM, AnimTracks)>) -> BinResult<Vec<u8>> {
    let mut out = Vec::new();
    for (_, nam, track_data) in ordered_ani_tracks(tracks)? {
        for block in &nam.tracks {
            out.extend(write_block_payload(block, nam, track_data)?);
        }
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
            stream_header: None,
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

#[binrw::writer(writer, endian)]
fn write_ani_blocks(tracks: &Vec<BlockInfo>) -> BinResult<()> {
    for block in tracks {
        if block.optimized && block.stream {
            let pos = writer.stream_position()?;
            let size: u32 = block
                .size
                .try_into()
                .map_err(|_| binrw::Error::AssertFail {
                    pos,
                    message: "ANI block size does not fit in u32".into(),
                })?;
            size.write_options(writer, endian, ())?;
        }
    }
    Ok(())
}

#[binrw::writer(writer, endian)]
fn write_materials(materials: &Vec<(u32, MAT)>, compute: bool) -> BinResult<()> {
    for (key, mat) in materials {
        key.write_options(writer, endian, ())?;
        mat.write_options(writer, endian, compute)?;
    }
    Ok(())
}

#[binrw::writer(writer, endian)]
fn write_emi_textures(maps: &Vec<EMI_Textures>) -> BinResult<()> {
    for map in maps {
        map.write_options(writer, endian, ())?;
    }
    EMI_Textures { key: 0, data: None }.write_options(writer, endian, ())?;
    Ok(())
}

#[binrw]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, facet::Facet)]
struct AniStreamHeader {
    size: u16,
    start_frame: u16,
    num_frames: u16,
}

#[binrw::parser(reader, endian)]
fn parse_ani_track_entries(
    num_objects: u32,
    data: &Vec<u8>,
) -> BinResult<HashMap<u8, (NAM, AnimTracks)>> {
    let mut tracks = HashMap::new();
    let mut offset = 0usize;
    for idx in 0..num_objects {
        let mut nam = NAM::read_options(reader, endian, ())?;
        let remaining = data.get(offset..).ok_or_else(|| binrw::Error::AssertFail {
            pos: 0,
            message: "ANI/NABK track offset out of bounds".into(),
        })?;
        let track_data =
            parse_anim_tracks_from_block_data(&mut nam.tracks, remaining).map_err(|err| {
                binrw::Error::AssertFail {
                    pos: 0,
                    message: err.to_string(),
                }
            })?;
        let consumed: usize = nam.tracks.iter().map(|b| b.size).sum();
        offset = offset
            .checked_add(consumed)
            .ok_or_else(|| binrw::Error::AssertFail {
                pos: 0,
                message: "ANI data offset overflow".into(),
            })?;
        let key: u8 = idx.try_into().map_err(|_| binrw::Error::AssertFail {
            pos: 0,
            message: "ANI supports at most 256 tracks".into(),
        })?;
        tracks.insert(key, (nam, track_data));
    }
    Ok(tracks)
}

#[cfg(test)]
mod ani_stream_roundtrip_tests {
    use super::{
        AniStreamHeader, AniTrackType, AnimTracks, BlockInfo, NAM,
        parse_anim_tracks_from_block_data, write_block_payload,
    };

    #[test]
    fn streamed_ani_header_survives_roundtrip() {
        let mut blocks = vec![BlockInfo {
            size: 18,
            elem_size: 12,
            stream: true,
            optimized: true,
            track_type: AniTrackType::Position,
            stream_header: None,
        }];
        let mut raw = Vec::new();
        raw.extend_from_slice(&18u16.to_le_bytes());
        raw.extend_from_slice(&3u16.to_le_bytes());
        raw.extend_from_slice(&1u16.to_le_bytes());
        for value in [1.0f32, 2.0, 3.0] {
            raw.extend_from_slice(&value.to_le_bytes());
        }

        let tracks = parse_anim_tracks_from_block_data(&mut blocks, &raw).unwrap();
        assert_eq!(
            blocks[0].stream_header,
            Some(AniStreamHeader {
                size: 18,
                start_frame: 3,
                num_frames: 1,
            })
        );

        let nam = NAM {
            start_frame: 777,
            num_frames: 999,
            cm3_flags: [AniTrackType::Position].into_iter().collect(),
            opt_flags: 1,
            stm_flags: 1,
            tracks: blocks.clone(),
            eva: None,
        };

        let rebuilt = write_block_payload(&blocks[0], &nam, &tracks).unwrap();
        assert_eq!(rebuilt, raw);
    }

    #[test]
    fn streamed_ani_header_fallback_uses_nam_when_missing() {
        let block = BlockInfo {
            size: 18,
            elem_size: 12,
            stream: true,
            optimized: true,
            track_type: AniTrackType::Position,
            stream_header: None,
        };
        let nam = NAM {
            start_frame: 5,
            num_frames: 1,
            cm3_flags: [AniTrackType::Position].into_iter().collect(),
            opt_flags: 1,
            stm_flags: 1,
            tracks: vec![block.clone()],
            eva: None,
        };
        let tracks = AnimTracks {
            pos: Some(vec![[1.0, 2.0, 3.0]]),
            ..Default::default()
        };

        let rebuilt = write_block_payload(&block, &nam, &tracks).unwrap();
        assert_eq!(u16::from_le_bytes([rebuilt[2], rebuilt[3]]), 5);
        assert_eq!(u16::from_le_bytes([rebuilt[4], rebuilt[5]]), 1);
    }
}

#[binrw::writer(writer, endian)]
fn write_ani_track_entries(
    tracks: &HashMap<u8, (NAM, AnimTracks)>,
    compute: bool,
) -> BinResult<()> {
    for (_, nam, _) in ordered_ani_tracks(tracks)? {
        nam.write_options(writer, endian, compute)?;
    }
    Ok(())
}

#[binrw]
#[brw(magic = b"NAM\0")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct NAM {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==1))]
    #[bw(calc = 1u32)]
    version: u32,
    pub start_frame: u32,
    pub num_frames: u32,
    #[br(try_map(|v: u32| parse_ani_track_type(v)))]
    #[bw(map = encode_ani_track_type)]
    pub cm3_flags: BTreeSet<AniTrackType>,
    #[br(assert(opt_flags & 0xfff8 == 0, "invalid NAM opt_flags"))]
    #[bw(assert(*opt_flags & 0xfff8 == 0, "invalid NAM opt_flags"))]
    pub opt_flags: u32,
    #[br(assert(stm_flags & 0xfff8 == 0))]
    pub stm_flags: u32,
    #[br(parse_with = parse_ani_blocks, args(num_frames,&cm3_flags,opt_flags,stm_flags))]
    #[bw(write_with = write_ani_blocks)]
    pub tracks: Vec<BlockInfo>,
    #[br(if(cm3_flags.contains(&AniTrackType::EVA)))]
    #[bw(if(self.cm3_flags.contains(&AniTrackType::EVA)), args_raw = compute)]
    pub eva: Option<EVA>,
}

#[binrw]
#[brw(magic = b"NABK")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct NABK {
    #[br(temp)]
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(count=size)]
    pub data: Vec<u8>,
}

#[binrw]
#[brw(magic = b"ANI\0")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct ANI {
    #[br(temp)]
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==2, "Invalid ANI version"))]
    #[bw(calc = 2u32)]
    version: u32,
    pub fps: f32,
    pub first_frame: u32,
    pub last_frame: u32,
    #[bw(try_calc = tracks.len().try_into())]
    pub num_objects: u32,
    is_active: u32,
    #[bw(try_calc = track_map.len().try_into())]
    num_nodes: u32,
    #[br(count=num_nodes, map=|data: Vec<u8>| data.iter().map(|&v| (v!=0).then(|| v-1)).collect())]
    #[bw(map = |value: &Vec<Option<u8>>| encode_track_map(value))]
    pub track_map: Vec<Option<u8>>,
    #[bw(try_map = |_: &NABK| build_ani_nabk_data(tracks).map(|data| NABK { data }))]
    #[bw(args_raw = compute)]
    data: NABK,
    #[br(parse_with = parse_ani_track_entries, args(num_objects, &data.data))]
    #[bw(write_with = write_ani_track_entries, args(compute))]
    pub tracks: HashMap<u8, (NAM, AnimTracks)>,
}

impl ANI {
    pub fn get_track(&self, index: usize) -> Result<Option<AnimTracks>> {
        let Some(index) = u8::try_from(index).ok() else {
            return Ok(None);
        };
        Ok(self
            .tracks
            .get(&index)
            .map(|(_, track_data)| track_data.clone()))
    }
}

#[binrw]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct SM3 {
    #[bw(try_calc = compute_size(self, 4, compute)?.try_into())]
    size: u32,
    #[br(temp,assert(timestamp_magic==0x6515f8,"Invalid timestamp"))]
    #[bw(calc = 0x6515f8u32)]
    timestamp_magic: u32,
    #[br(try_map=convert_timestamp)]
    #[bw(map = |value: &DateTime<Utc>| value.timestamp() as u32)]
    dependency_timestamp_a: DateTime<Utc>,
    #[br(try_map=convert_timestamp)]
    #[bw(map = |value: &DateTime<Utc>| value.timestamp() as u32)]
    dependency_timestamp_b: DateTime<Utc>,
    #[bw(args_raw = compute)]
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

#[binrw]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct CM3 {
    #[bw(try_calc = compute_size(self, 4, compute)?.try_into())]
    size: u32,
    #[br(temp,assert(timestamp_magic==0x6515f8,"Invalid timestamp"))]
    #[bw(calc = 0x6515f8u32)]
    timestamp_magic: u32,
    #[br(try_map=convert_timestamp)]
    #[bw(map = |value: &DateTime<Utc>| value.timestamp() as u32)]
    dependency_timestamp_a: DateTime<Utc>,
    #[br(try_map=convert_timestamp)]
    #[bw(map = |value: &DateTime<Utc>| value.timestamp() as u32)]
    dependency_timestamp_b: DateTime<Utc>,
    #[bw(args_raw = compute)]
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

#[binrw]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct Dummy {
    has_next: u32,
    pub name: PascalString,
    pub pos: [f32; 3],
    pub rot: [f32; 3],
    #[bw(args_raw = compute)]
    pub info: Optional<INI>,
}

#[binrw]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct DUM {
    #[bw(try_calc = compute_size(self, 4, compute)?.try_into())]
    size: u32,
    #[br(assert(version==1, "Invalid DUM version"))]
    #[bw(calc = 1u32)]
    version: u32,
    #[bw(try_calc = dummies.len().try_into())]
    num_dummies: u32,
    #[br(count=num_dummies)]
    #[bw(args_raw = compute)]
    pub dummies: Vec<Dummy>,
    #[br(assert(end_marker==0))]
    #[bw(calc = 0u32)]
    end_marker: u32,
}

#[binrw]
#[brw(magic = b"QUAD")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct QUAD {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==1, "Invalid QUAD version"))]
    #[bw(calc = 1u32)]
    version: u32,
    mesh_id: u32,
    triangle_indices: Table<2, u16>,
    base_height_range: [f32; 2],
    active_height_range: [f32; 2],
    #[bw(args_raw = compute)]
    pub child: Optional<Box<QUAD>>,
}

#[binrw]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct CMSH_Tri {
    cache_stamp: u32,
    pub normal: [f32; 3],
    plane_distance: f32,
    pub idx: [u16; 3],
    collision_flags: u16,
}

#[binrw]
#[brw(magic = b"CMSH")]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct CMSH {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==2, "Invalid CMSH version"))]
    #[bw(calc = 2u32)]
    version: u32,
    #[br(assert(collide_mesh_size==0x34, "Invalid collision mesh size"))]
    #[bw(calc = 0x34u32)]
    collide_mesh_size: u32,
    pub zone_name: PascalString,
    triangle_mesh_flags: u16,
    pub sector: u16,
    mesh_unique_id: u16,
    load_mesh_slot: u8,
    is_out_sector: u8,
    bbox: [[f32; 3]; 2],
    pub verts: Table<0xc, [f32; 3]>,
    pub tris: Table<0x1c, CMSH_Tri>,
}

#[binrw]
#[brw(magic = b"AMC\0")]
#[derive(Debug, Serialize, Default, Deserialize, facet::Facet)]
struct EmptyAMC {
    #[br(assert(size==0))]
    #[bw(calc = 0u32)]
    size: u32,
}

// TODO: OG game uses version_code==1
#[binrw]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct AMC {
    // subtract 8 for empty AMC block at the end
    #[bw(try_calc = compute_size(self, 4+8, compute)?.try_into())]
    size: u32,
    #[br(assert(version==100,"Invalid AMC version"))]
    #[bw(calc = 100u32)]
    version: u32,
    #[br(assert(amc_version_code==0, "Invalid AMC version_code: {}", amc_version_code))]
    amc_version_code: u32,
    collision_bbox: [[f32; 3]; 2],
    total_triangles: u32,
    quad_grid_bbox: [[f32; 3]; 2],
    quad_grid_center: [f32; 3],
    #[bw(args_raw = compute)]
    pub cmsh: [CMSH; 2],
    #[bw(try_calc = sector_col.len().try_into())]
    num_sectors: u32,
    #[br(count=num_sectors)]
    #[bw(args_raw = compute)]
    pub sector_col: Vec<[CMSH; 2]>,
    grid_size: [u32; 2],
    grid_scale: [f32; 2],
    grid_scale_inv: [f32; 2],
    #[br(temp)]
    #[bw(try_calc = quads.len().try_into())]
    num_quads: u32,
    #[br(count=num_quads)]
    #[bw(args_raw = compute)]
    pub quads: Vec<QUAD>,
    #[bw(args_raw = compute)]
    final_quad: Optional<QUAD>,
    #[br(temp)]
    #[bw(calc = EmptyAMC::default())]
    _empty_amc: EmptyAMC,
}

#[binrw]
#[br(import(version: u32))]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct TriV104 {
    #[br(if(version>=0x69))]
    #[bw(if(zone_name.is_some()))]
    pub zone_name: Option<PascalString>,
    pub mat_key: u32,
    pub map_key: u32,
    #[bw(try_calc = tris.len().try_into())]
    num_tris: u32,
    #[br(count=num_tris)]
    pub tris: Vec<[u16; 3]>,
    #[bw(args_raw = compute)]
    pub geometry_verts: LFVF,
    #[bw(args_raw = compute)]
    pub lightmap_verts: LFVF,
}

#[binrw]
#[brw(magic = b"TRI\0")]
#[br(import(version: u32))]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct TRI {
    #[bw(try_calc = compute_size(self, 8, compute)?.try_into())]
    size: u32,
    pub flags: u32,
    pub name: PascalString,
    pub zone_index: u32,
    #[br(args(version))]
    #[bw(args_raw = compute)]
    pub data: TriV104,
}

#[binrw]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct EMI_Textures {
    pub key: u32,
    #[br(if(key!=0))]
    #[bw(if(*key!=0))]
    pub data: Option<(PascalString, u32, PascalString)>,
}

#[binrw]
#[bw(import_raw(compute: bool))]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
pub struct EMI {
    #[bw(try_calc = compute_size(self, 4, compute)?.try_into())]
    size: u32,
    #[br(assert((103..=105).contains(&version)))]
    pub version: u32,
    #[bw(try_calc = materials.len().try_into())]
    pub num_materials: u32,
    #[br(count=num_materials)]
    #[bw(write_with = write_materials, args(compute))]
    pub materials: Vec<(u32, MAT)>,
    #[br(parse_with = until_exclusive(|v: &EMI_Textures| v.key==0))]
    #[bw(write_with = write_emi_textures)]
    pub maps: Vec<EMI_Textures>,
    #[bw(try_calc = tri.len().try_into())]
    pub num_objs: u32,
    #[br(count=num_objs,args{inner: (version,)})]
    #[bw(args_raw = compute)]
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

#[binrw]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
#[serde(tag = "type")]
#[repr(u8)]
pub enum Data {
    #[brw(magic = b"SM3\0")]
    SM3(SM3),
    #[brw(magic = b"CM3\0")]
    CM3(CM3),
    #[brw(magic = b"DUM\0")]
    DUM(DUM),
    #[brw(magic = b"AMC\0")]
    AMC(AMC),
    #[brw(magic = b"EMI\0")]
    EMI(EMI),
}

impl Data {
    pub fn dependencies(&self) -> Vec<String> {
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
        eprintln!("{} Rest:\n{}", path.as_str(), rhexdumps!(&buffer[..n], pos));
    };
    while let Ok(n) = fh.read(&mut buffer)
        && n != 0
    {
        rest_size += n;
    }
    assert_eq!(rest_size, 0);
    // eprintln!("+{rest_size} unparsed bytes");
    Ok(ret)
}

fn load_ini(path: &VfsPath) -> IniData {
    let Ok(data) = path.read_to_string() else {
        return IniData::default();
    };
    Ini::new().read(data).unwrap_or_default()
}

#[derive(Serialize, Debug, Deserialize, facet::Facet)]
pub struct Level {
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
    pub fn load(path: &VfsPath) -> Result<Self> {
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

pub fn resolve_dep(dep: &str, asset_path: &VfsPath, config: &IniData) -> Option<VfsPath> {
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

pub fn find_packed<P: AsRef<Path>>(root: P) -> Result<Vec<PathBuf>> {
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
pub enum ParsedData {
    Level(Level),
    Data(Data),
}

pub mod multi_pack_fs {
    use std::{
        collections::{BTreeMap, HashMap},
        path::Path,
        sync::Arc,
    };

    use color_eyre::eyre::bail;
    use vfs::{SeekAndRead, VfsPath};

    use super::{Deserialize, ParsedData, PathBuf, Result, Serialize};
    use crate::packed_vfs::{MultiPack, MultiPackTransformer};

    #[derive(Serialize, Debug, Deserialize, facet::Facet)]
    pub struct Entry {
        pub path: String,
        pub size: u64,
        pub is_file: bool,
    }

    #[derive(Debug, Clone)]
    pub struct MultiPackFS {
        pub root: VfsPath,
        pack: MultiPack,
        current: Vec<String>,
    }

    impl MultiPackFS {
        pub fn new<P: AsRef<Path>>(files: &[P]) -> Result<Self> {
            let pack = MultiPack::load_all(files)?;
            Ok(MultiPackFS {
                root: pack.clone().into(),
                pack,
                current: vec![],
            })
        }

        pub fn transform(&self) -> MultiPackTransformer {
            // self.fs.
            MultiPackTransformer::new(self.pack.clone())
        }

        pub fn for_each_file(&self, func: fn(&str, &[u8]) -> Result<()>) -> Result<()> {
            self.pack.for_each_file(func)
        }

        pub fn exists(&self, path: &str) -> Result<bool> {
            Ok(self
                .root
                .root()
                .join(path)
                .and_then(|p| p.metadata())
                .is_ok())
        }

        pub fn is_level(&self, path: Option<String>) -> Result<bool> {
            let mut root = self.root.root();
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

        pub fn ls(&self) -> Result<Vec<Entry>> {
            let mut ret = vec![];
            let mut root = self.root.root();
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

        pub fn entries(&self) -> Result<Vec<Entry>> {
            let mut entries = vec![];
            for res in self.root.walk_dir()? {
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

        pub fn pwd(&self) -> Result<String> {
            if self.current.is_empty() {
                return Ok("/".to_owned());
            }
            let mut root = self.root.root();
            for entry in &self.current {
                root = root.join(entry)?
            }
            Ok(root.as_str().to_owned())
        }

        pub fn cd(&mut self, path: &str) -> Result<()> {
            self.current = self.current.drain(..).filter(|v| !v.is_empty()).collect();
            if path == ".." {
                self.current.pop();
                return Ok(());
            }
            let mut root = self.root.root();
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

        pub fn dependencies(&self, path: &str) -> Result<BTreeMap<String, String>> {
            let mut root = self.root.root();
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

        pub fn open_file(&self, path: &str) -> Result<Box<dyn SeekAndRead>> {
            let mut root = self.root.root();
            for entry in &self.current {
                root = root.join(entry)?
            }
            let path = root.join(path)?;

            if path.is_dir()? {
                bail!("{path} is a directory", path = path.as_str());
            }
            Ok(path.open_file()?)
        }

        pub fn parse_file(&self, path: &str) -> Result<ParsedData> {
            let mut root = self.root.root();
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

// Test Zone :)

fn compute_size<B>(s: B, header_size: u64, compute: bool) -> BinResult<u64>
where
    B: for<'a> BinWrite<Args<'a> = bool>,
{
    let mut buffer = Cursor::new(Vec::new());
    if !compute {
        s.write_le_args(&mut buffer, true)?;
    }
    Ok(buffer.position().saturating_sub(header_size))
}

#[binrw]
#[derive(Debug, Serialize, Deserialize, facet::Facet)]
#[brw(magic = b"TEST")]
#[bw(import_raw(compute: bool))]
pub struct Test {
    #[bw(try_calc=compute_size(self, 8, compute)?.try_into())]
    pub size: u32,
    #[bw(try_calc=data.len().try_into())]
    pub n: u32,
    #[br(count=n)]
    pub data: Vec<[f32; 3]>,
}
