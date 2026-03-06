use std::{
    collections::{HashMap, VecDeque},
    io::{BufWriter, Cursor, Read, Seek, SeekFrom, Write},
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use binrw::{BinReaderExt, io::BufReader};
use color_eyre::eyre::{Context, Result, anyhow, bail};
use fs_err as fs;
use futures_lite::{AsyncRead, AsyncSeek};
use memmap2::Mmap;
use serde::Serialize;
use vfs::{FileSystem, SeekAndWrite, VfsMetadata, error::VfsErrorKind};

use crate::parser::{PackedEntry, PackedHeader};

#[derive(Debug, Clone)]
pub enum PackedTransformAction {
    Keep,
    Rename(String),
    Delete,
    Rewrite { path: Option<String> },
}

pub trait WriteSeek: Write + Seek {}

impl<T: Write + Seek + ?Sized> WriteSeek for T {}

fn encode_packed_path(path: &str) -> Vec<u8> {
    path.chars()
        .map(|c| u8::try_from(c as u32).unwrap_or(b'?'))
        .collect()
}

fn packed_header_len(entries: &[PackedEntry]) -> Result<u64> {
    let mut len = 12u64;
    for entry in entries {
        let path_len = encode_packed_path(&entry.path.string).len() as u64;
        len = len
            .checked_add(4)
            .and_then(|v| v.checked_add(path_len))
            .and_then(|v| v.checked_add(8))
            .ok_or_else(|| anyhow!("Packed header size overflow"))?;
    }
    Ok(len)
}

fn write_packed_header<W: Write>(writer: &mut W, entries: &[PackedEntry]) -> Result<()> {
    writer.write_all(b"BFPK")?;
    writer.write_all(&0u32.to_le_bytes())?;
    writer.write_all(
        &u32::try_from(entries.len())
            .context("Too many packed entries")?
            .to_le_bytes(),
    )?;
    for entry in entries {
        let path = encode_packed_path(&entry.path.string);
        writer.write_all(
            &u32::try_from(path.len())
                .context("Packed path is too long")?
                .to_le_bytes(),
        )?;
        writer.write_all(&path)?;
        writer.write_all(&entry.size.to_le_bytes())?;
        writer.write_all(&entry.offset.to_le_bytes())?;
    }
    Ok(())
}

pub fn transform_packed<PIn, POut, F>(input: PIn, output: POut, mut transform: F) -> Result<()>
where
    PIn: AsRef<Path>,
    POut: AsRef<Path>,
    F: FnMut(&PackedEntry, &mut dyn Read, &mut dyn WriteSeek) -> Result<PackedTransformAction>,
{
    let input_path = input.as_ref();
    let output_path = output.as_ref();

    let mut in_file = fs::File::open(input_path)
        .with_context(|| format!("Failed to open {}", input_path.display()))?;
    let mut header_reader = BufReader::new(fs::File::open(input_path)?);
    let input_header = header_reader.read_le::<PackedHeader>()?;

    let parent = output_path
        .parent()
        .ok_or_else(|| anyhow!("Invalid output path"))?;
    let mut temp_path = parent.to_path_buf();
    temp_path.push(format!(
        ".{}.tmp",
        output_path
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("packed")
    ));

    let mut temp_payload = BufWriter::new(fs::File::create(&temp_path)?);
    let mut out_entries: Vec<PackedEntry> = Vec::with_capacity(input_header.files.len());
    let mut payload_offsets: Vec<u64> = Vec::with_capacity(input_header.files.len());

    for entry in &input_header.files {
        in_file.seek(SeekFrom::Start(entry.offset as u64))?;
        let mut source = std::io::Read::take(&mut in_file, entry.size as u64);
        let payload_offset = temp_payload.stream_position()?;
        let action = transform(entry, &mut source, &mut temp_payload)?;

        match action {
            PackedTransformAction::Keep => {
                std::io::copy(&mut source, &mut temp_payload)?;
                let end_pos = temp_payload.stream_position()?;
                let size = end_pos
                    .checked_sub(payload_offset)
                    .ok_or_else(|| anyhow!("Transformed entry size underflow"))?;
                let size = u32::try_from(size).context("Packed entry larger than 4 GiB")?;
                out_entries.push(PackedEntry {
                    path: entry.path.clone(),
                    size,
                    offset: 0,
                });
                payload_offsets.push(payload_offset);
            }
            PackedTransformAction::Rename(path) => {
                std::io::copy(&mut source, &mut temp_payload)?;
                let end_pos = temp_payload.stream_position()?;
                let size = end_pos
                    .checked_sub(payload_offset)
                    .ok_or_else(|| anyhow!("Transformed entry size underflow"))?;
                let size = u32::try_from(size).context("Packed entry larger than 4 GiB")?;
                out_entries.push(PackedEntry {
                    path: crate::parser::PascalString { string: path },
                    size,
                    offset: 0,
                });
                payload_offsets.push(payload_offset);
            }
            PackedTransformAction::Delete => {
                if temp_payload.stream_position()? != payload_offset {
                    bail!(
                        "Transformer wrote bytes for deleted entry {}",
                        entry.path.string
                    );
                }
            }
            PackedTransformAction::Rewrite { path } => {
                let end_pos = temp_payload.stream_position()?;
                let size = end_pos
                    .checked_sub(payload_offset)
                    .ok_or_else(|| anyhow!("Transformed entry size underflow"))?;
                let size = u32::try_from(size).context("Packed entry larger than 4 GiB")?;
                out_entries.push(PackedEntry {
                    path: crate::parser::PascalString {
                        string: path.unwrap_or_else(|| entry.path.string.clone()),
                    },
                    size,
                    offset: 0,
                });
                payload_offsets.push(payload_offset);
            }
        }
    }

    temp_payload.flush()?;
    drop(temp_payload);

    let header_size = packed_header_len(&out_entries)?;

    for (entry, payload_offset) in out_entries.iter_mut().zip(payload_offsets.into_iter()) {
        let offset = header_size
            .checked_add(payload_offset)
            .ok_or_else(|| anyhow!("Packed file offset overflow"))?;
        entry.offset = u32::try_from(offset).context("Packed file exceeds 4 GiB limit")?;
    }

    let mut out_file = BufWriter::new(fs::File::create(output_path)?);
    write_packed_header(&mut out_file, &out_entries)?;
    let mut tmp_read = fs::File::open(&temp_path)?;
    std::io::copy(&mut tmp_read, &mut out_file)?;
    out_file.flush()?;

    fs::remove_file(&temp_path)?;
    Ok(())
}

#[derive(Debug)]
pub struct PackedFile {
    _fh: fs::File,
    mm: Arc<Mmap>,
    _path: PathBuf,
}

#[derive(Debug)]
pub struct MultiPack {
    files: Vec<PackedFile>,
    pub tree: DirectoryTree,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum DirectoryTree {
    File {
        data: Range<usize>,
        file_index: usize,
    },
    Directory {
        entries: HashMap<String, DirectoryTree>,
    },
}

impl Default for DirectoryTree {
    fn default() -> Self {
        Self::Directory {
            entries: Default::default(),
        }
    }
}

impl MultiPack {
    pub fn load_all<P: AsRef<Path>>(files: &[P]) -> Result<Self> {
        let mut tree = DirectoryTree::default();
        let mut packed_files = vec![];
        for (file_index, file) in files.iter().enumerate() {
            let mut fh = BufReader::new(fs::File::open(file.as_ref())?);
            let header = fh.read_le::<PackedHeader>()?;
            println!(
                "Found {} files in {}",
                header.files.len(),
                file.as_ref().display()
            );
            // for file in &header.files {
            //     println!("{}", file.path.string);
            // }
            tree.merge(&header.files, file_index);
            let fh = fh.into_inner();
            packed_files.push(PackedFile {
                mm: Arc::new(unsafe { Mmap::map(&fh)? }),
                _path: file.as_ref().to_owned(),
                _fh: fh,
            });
        }
        Ok(Self {
            tree,
            files: packed_files,
        })
    }

    pub fn get_file(&self, path: &str) -> vfs::VfsResult<FileHandle> {
        match self.tree.get_entry(path)? {
            DirectoryTree::File { data, file_index } => {
                let Some(file) = self.files.get(*file_index) else {
                    return Err(VfsErrorKind::FileNotFound.into());
                };
                let mm = Arc::clone(&file.mm);
                let cursor = Cursor::new(Arc::from(&mm[data.clone()]));
                Ok(FileHandle {
                    cursor,
                    _mm: mm,
                    data: data.clone(),
                })
            }
            DirectoryTree::Directory { .. } => Err(VfsErrorKind::NotSupported.into()),
        }
    }

    // pub fn add<P: AsRef<Path>>(&mut self, file: &P) -> Result<()> {
    //     let file = file.as_ref();
    //     for packed in &self.files {
    //         if packed.path == file {
    //             bail!("File already loaded!");
    //         }
    //     }
    //     let mut fh = BufReader::new(fs::File::open(file)?);
    //     let header = fh.read_le::<PackedHeader>()?;
    //     println!("Found {} files in {}", header.files.len(), file.display());
    //     self.tree.merge(&header.files, self.files.len());
    //     let fh = fh.into_inner();
    //     self.files.push(PackedFile {
    //         mm: Arc::new(unsafe { Mmap::map(&fh)? }),
    //         path: file.to_owned(),
    //         _fh: fh,
    //     });
    //     Ok(())
    // }
}

impl DirectoryTree {
    fn add_child(&mut self, name: &str, node: Self) -> &mut Self {
        match self {
            Self::File { .. } => panic!("Can't add child to file!"),
            Self::Directory { entries } => entries.entry(name.to_ascii_lowercase()).or_insert(node),
        }
    }

    fn merge(&mut self, files: &[PackedEntry], file_index: usize) {
        for file in files {
            let mut folder = &mut *self;
            let path: Vec<_> = file.path.string.split('/').collect();
            if let Some((filename, path)) = path.as_slice().split_last() {
                for part in path {
                    let DirectoryTree::Directory { entries } = folder else {
                        unreachable!();
                    };
                    folder = entries.entry(part.to_ascii_lowercase()).or_default();
                }
                let offset = file.offset as usize;
                let size = file.size as usize;
                folder.add_child(
                    filename,
                    DirectoryTree::File {
                        data: offset..(offset + size),
                        file_index,
                    },
                );
            }
        }
    }

    pub fn get_entry(&self, path: &str) -> vfs::VfsResult<&Self> {
        let mut path = path.to_ascii_lowercase();
        if !path.starts_with('/') {
            path = "/".to_owned() + &path;
        }
        let mut path: VecDeque<&str> = match path.as_str() {
            "/" => VecDeque::new(),
            path => path.split('/').collect(),
        };
        if path.front() == Some(&"") {
            path.pop_front();
        }
        let mut tree = self;
        while let Some(part) = path.pop_front() {
            match tree {
                DirectoryTree::File { .. } => {
                    if !path.is_empty() {
                        return Err(VfsErrorKind::InvalidPath.into());
                    }
                }
                DirectoryTree::Directory { entries } => {
                    if let Some(entry) = entries.get(part) {
                        tree = entry;
                    } else {
                        return Err(VfsErrorKind::FileNotFound.into());
                    }
                }
            };
        }
        Ok(tree)
    }
}

#[derive(Debug)]
pub struct FileHandle {
    _mm: Arc<Mmap>,
    cursor: Cursor<Arc<[u8]>>,
    data: Range<usize>,
}

impl FileHandle {
    pub fn get<'a>(&'a self) -> &'a [u8] {
        let b = self.cursor.get_ref();
        &b[self.data.clone()]
    }
}

impl Seek for FileHandle {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.cursor.seek(pos)
    }
}

impl Read for FileHandle {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl FileHandle {
    fn buffer(&self) -> &[u8] {
        let idx = self.data.clone();
        &self.cursor.get_ref()[idx]
    }
}

impl AsyncRead for FileHandle {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &mut [u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        let fh = self.get_mut();
        std::task::Poll::Ready(fh.cursor.read(buf))
    }
}

impl AsyncSeek for FileHandle {
    fn poll_seek(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        pos: SeekFrom,
    ) -> std::task::Poll<std::io::Result<u64>> {
        let fh = self.get_mut();
        std::task::Poll::Ready(fh.cursor.seek(pos))
    }
}

impl FileSystem for MultiPack {
    fn read_dir(&self, path: &str) -> vfs::VfsResult<Box<dyn Iterator<Item = String> + Send>> {
        match self.tree.get_entry(path)? {
            DirectoryTree::File { .. } => Err(VfsErrorKind::NotSupported.into()),
            DirectoryTree::Directory { entries } => {
                let keys: Vec<String> = entries.keys().cloned().collect();
                Ok(Box::new(keys.into_iter()))
            }
        }
    }

    fn create_dir(&self, _: &str) -> vfs::VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn open_file(&self, path: &str) -> vfs::VfsResult<Box<dyn vfs::SeekAndRead + Send>> {
        Ok(Box::new(self.get_file(path)?))
    }

    fn create_file(&self, _: &str) -> vfs::VfsResult<Box<dyn SeekAndWrite + Send>> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn append_file(&self, _: &str) -> vfs::VfsResult<Box<dyn SeekAndWrite + Send>> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn metadata(&self, path: &str) -> vfs::VfsResult<vfs::VfsMetadata> {
        Ok(match self.tree.get_entry(path)? {
            DirectoryTree::File {
                data,
                file_index: _,
            } => VfsMetadata {
                file_type: vfs::VfsFileType::File,
                len: data
                    .len()
                    .try_into()
                    .map_err(|e| VfsErrorKind::Other(format!("{e}")))?,
                created: None,
                modified: None,
                accessed: None,
            },
            DirectoryTree::Directory { entries: _ } => VfsMetadata {
                file_type: vfs::VfsFileType::Directory,
                len: 0,
                created: None,
                modified: None,
                accessed: None,
            },
        })
    }

    fn exists(&self, path: &str) -> vfs::VfsResult<bool> {
        self.tree.get_entry(path).map(|_| true)
    }

    fn remove_file(&self, _: &str) -> vfs::VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn remove_dir(&self, _: &str) -> vfs::VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        io::{Cursor, Read, Seek, SeekFrom},
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use binrw::{BinReaderExt, BinWrite};

    use super::{PackedTransformAction, transform_packed};
    use crate::parser::{Data, PackedHeader};

    const SUPPORTED_EXTS: &[&str] = &["cm3", "sm3", "emi", "dum", "amc"];

    fn workspace_data_packed() -> Option<PathBuf> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data.packed");
        path.is_file().then_some(path)
    }

    fn normalized_ext(path: &str) -> Option<String> {
        let clean = path.trim_end_matches('\0');
        let ext = PathBuf::from(clean)
            .extension()
            .and_then(|v| v.to_str())?
            .to_ascii_lowercase();
        Some(ext)
    }

    fn read_entry_bytes(file: &mut std::fs::File, offset: u32, size: u32) -> Vec<u8> {
        let mut bytes = vec![0u8; size as usize];
        file.seek(SeekFrom::Start(offset as u64)).unwrap();
        file.read_exact(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn packed_roundtrip_preserves_supported_entries() {
        let Some(input_path) = workspace_data_packed() else {
            return;
        };

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let output_path = std::env::temp_dir().join(format!("scrap_roundtrip_{unique}.packed"));

        transform_packed(
            &input_path,
            &output_path,
            |entry, source, target| match normalized_ext(&entry.path.string).as_deref() {
                Some(ext) if SUPPORTED_EXTS.contains(&ext) => {
                    let mut bytes = Vec::new();
                    source.read_to_end(&mut bytes)?;
                    let mut cur = Cursor::new(&bytes);
                    if let Ok(parsed) = cur.read_le::<Data>() {
                        let mut rewritten = Cursor::new(Vec::new());
                        parsed.write_le(&mut rewritten)?;
                        target.write_all(rewritten.get_ref())?;
                    } else {
                        target.write_all(&bytes)?;
                    }
                    Ok(PackedTransformAction::Rewrite { path: None })
                }
                _ => Ok(PackedTransformAction::Keep),
            },
        )
        .unwrap();

        let mut input_header_reader =
            binrw::io::BufReader::new(std::fs::File::open(&input_path).unwrap());
        let input_header = input_header_reader.read_le::<PackedHeader>().unwrap();
        let mut output_header_reader =
            binrw::io::BufReader::new(std::fs::File::open(&output_path).unwrap());
        let output_header = output_header_reader.read_le::<PackedHeader>().unwrap();

        let mut output_by_path: HashMap<&str, (u32, u32)> = HashMap::new();
        for entry in &output_header.files {
            output_by_path.insert(&entry.path.string, (entry.offset, entry.size));
        }

        let mut input_file = std::fs::File::open(&input_path).unwrap();
        let mut output_file = std::fs::File::open(&output_path).unwrap();

        for entry in &input_header.files {
            let Some(ext) = normalized_ext(&entry.path.string) else {
                continue;
            };
            if !SUPPORTED_EXTS.contains(&ext.as_str()) {
                continue;
            }
            let Some((out_offset, out_size)) = output_by_path.get(entry.path.string.as_str())
            else {
                panic!("Missing entry after roundtrip: {}", entry.path.string);
            };

            let in_bytes = read_entry_bytes(&mut input_file, entry.offset, entry.size);
            let out_bytes = read_entry_bytes(&mut output_file, *out_offset, *out_size);
            assert_eq!(
                out_bytes, in_bytes,
                "Binary mismatch after roundtrip for {}",
                entry.path.string
            );
        }

        std::fs::remove_file(output_path).unwrap();
    }
}
