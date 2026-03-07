use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    io::{Cursor, Read, Seek, SeekFrom, Write},
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::parser::{PackedEntry, PackedHeader, PascalString};
use binrw::{io::BufReader, BinReaderExt};
use color_eyre::eyre::{anyhow, bail, Context, Result};
use fs_err as fs;
use futures_lite::{AsyncRead, AsyncSeek};
use glob::Pattern;
use memmap2::Mmap;
use serde::Serialize;
use vfs::{error::VfsErrorKind, FileSystem, SeekAndWrite, VfsMetadata};

#[derive(Debug)]
pub struct PackedFile {
    _fh: fs::File,
    mm: Arc<Mmap>,
    _path: PathBuf,
    pub header: PackedHeader,
}

impl PackedFile {
    fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut fh = BufReader::new(fs::File::open(path.as_ref())?);
        let header = fh.read_le::<PackedHeader>()?;
        println!(
            "Found {} files in {}",
            header.files.len(),
            path.as_ref().display()
        );
        let fh = fh.into_inner();
        Ok(Self {
            mm: Arc::new(unsafe { Mmap::map(&fh)? }),
            _path: path.as_ref().to_owned(),
            _fh: fh,
            header,
        })
    }
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
            let packed = PackedFile::load(file)?;
            tree.merge(&packed.header.files, file_index);
            packed_files.push(packed);
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

enum PackedOp {
    Delete(Pattern),
    Rename(Pattern, fn(&str) -> String),
    Patch(Pattern, fn(&str, &[u8]) -> Vec<u8>),
    Add(String, fn() -> Vec<u8>),
}

pub struct PackedTransformer {
    packed: PackedFile,
    ops: Vec<PackedOp>,
}

impl PackedTransformer {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self {
            packed: PackedFile::load(path)?,
            ops: vec![],
        })
    }

    pub fn delete(mut self, pattern: &str) -> Result<Self> {
        self.ops
            .push(PackedOp::Delete(glob::Pattern::new(pattern)?));
        Ok(self)
    }

    pub fn rename(mut self, pattern: &str, func: fn(&str) -> String) -> Result<Self> {
        self.ops
            .push(PackedOp::Rename(glob::Pattern::new(pattern)?, func));
        Ok(self)
    }

    pub fn patch(mut self, pattern: &str, func: fn(&str, &[u8]) -> Vec<u8>) -> Result<Self> {
        self.ops
            .push(PackedOp::Patch(glob::Pattern::new(pattern)?, func));
        Ok(self)
    }
    pub fn add(mut self, path: &str, func: fn() -> Vec<u8>) -> Result<Self> {
        self.ops.push(PackedOp::Add(path.to_owned(), func));
        Ok(self)
    }

    pub fn write<P: AsRef<Path>>(self, output_path: P) -> Result<()> {
        use binrw::BinWrite;

        let mut output_file = File::create(output_path)?;

        let (entries, new_files) = self.process_ops()?;

        let dummy_header = PackedHeader {
            files: entries
                .iter()
                .map(|e| PackedEntry {
                    path: e.path.clone(),
                    size: e.size,
                    offset: 0,
                })
                .collect(),
        };
        let header_size = dummy_header.size();

        let mut data_offset = header_size;

        let mut entries_with_offsets = Vec::with_capacity(entries.len());
        for entry in &entries {
            let offset = data_offset;
            data_offset += entry.size as usize;
            entries_with_offsets.push((entry, offset as u32));
        }

        output_file.seek(SeekFrom::Start(header_size as u64))?;

        for (entry, _offset) in &entries_with_offsets {
            if let Some(data) = new_files.get(&entry.path.string) {
                output_file.write_all(data)?;
            } else {
                let data = self.get_file_data(&entry.path.string)?;
                output_file.write_all(&data)?;
            }
        }

        output_file.seek(SeekFrom::Start(0))?;

        let header = PackedHeader {
            files: entries_with_offsets
                .into_iter()
                .map(|(e, offset)| {
                    let mut e = e.clone();
                    e.offset = offset;
                    e
                })
                .collect(),
        };
        header.write_le(&mut output_file)?;

        Ok(())
    }

    fn process_ops(&self) -> Result<(Vec<PackedEntry>, HashMap<String, Vec<u8>>)> {
        let mut result: Vec<PackedEntry> = self.packed.header.files.clone();
        let mut new_files: HashMap<String, Vec<u8>> = HashMap::new();

        for op in &self.ops {
            match op {
                PackedOp::Delete(pattern) => {
                    result.retain(|e| !pattern.matches(&e.path.string));
                }
                PackedOp::Rename(pattern, func) => {
                    for e in &mut result {
                        if pattern.matches(&e.path.string) {
                            e.path.string = func(&e.path.string);
                        }
                    }
                }
                PackedOp::Patch(pattern, func) => {
                    for e in &mut result {
                        if pattern.matches(&e.path.string) {
                            let data = self.get_file_data(&e.path.string)?;
                            let patched = func(&e.path.string, &data);
                            e.size = patched.len() as u32;
                        }
                    }
                }
                PackedOp::Add(path, generator) => {
                    let data = generator();
                    let size = data.len() as u32;
                    new_files.insert(path.clone(), data);
                    result.push(PackedEntry {
                        path: PascalString {
                            string: path.clone(),
                        },
                        size,
                        offset: 0,
                    });
                }
            }
        }

        Ok((result, new_files))
    }

    fn get_file_data(&self, path: &str) -> Result<Vec<u8>> {
        let file = self
            .packed
            .header
            .files
            .iter()
            .find(|e| e.path.string == path)
            .ok_or_else(|| anyhow!("File not found: {}", path))?;

        let mm = &self.packed.mm;
        let data_start = file.offset as usize;
        let data_end = data_start + file.size as usize;
        Ok(mm[data_start..data_end].to_vec())
    }
}
