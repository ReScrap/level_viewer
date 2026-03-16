use std::{
    borrow::Cow,
    collections::{HashMap, HashSet, VecDeque},
    fs::File,
    io::{Cursor, Read, Seek, SeekFrom, Write},
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use binrw::{BinReaderExt, io::BufReader};
use color_eyre::eyre::{Context, Result, anyhow, bail};
use fs_err as fs;
use futures_lite::{AsyncRead, AsyncSeek};
use glob::Pattern;
use memmap2::Mmap;
use serde::Serialize;
use vfs::{FileSystem, SeekAndWrite, VfsMetadata, error::VfsErrorKind};

use crate::parser::{PackedEntry, PackedHeader, PascalString};

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

#[derive(Debug, Clone)]
pub struct MultiPack {
    files: Arc<[PackedFile]>,
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
            files: packed_files.into(),
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

    pub fn for_each_file(&self, callback: fn(&str, &[u8]) -> Result<()>) -> Result<()> {
        for packed in self.files.iter() {
            for entry in &packed.header.files {
                let data_start = entry.offset as usize;
                let data_end = data_start
                    .checked_add(entry.size as usize)
                    .ok_or_else(|| anyhow!("Invalid entry range for {}", entry.path))?;
                if data_end > packed.mm.len() {
                    bail!("Entry out of bounds: {}", entry.path);
                }
                callback(&entry.path, &packed.mm[data_start..data_end])?;
            }
        }
        Ok(())
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
            let path: Vec<_> = file.path.split('/').collect();
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
    pub fn get(&self) -> &[u8] {
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

type RenameFunc = fn(&str) -> String;
type PatchFunc = fn(&str, &mut Cow<[u8]>) -> Result<()>;
type AddFunc = fn() -> Vec<u8>;

enum PackedOp {
    Delete(Pattern),
    Rename(Pattern, RenameFunc),
    Patch(Pattern, PatchFunc),
    Add(String, AddFunc),
}

struct PatchStep {
    path: String,
    func: PatchFunc,
}

enum PackedDataSource {
    Existing(usize),
    Added(AddFunc),
}

struct ProcessedEntry {
    path: String,
    source: PackedDataSource,
    patches: Vec<PatchStep>,
}

#[derive(Clone, Copy)]
enum ModSource {
    Existing {
        pack_index: usize,
        file_index: usize,
    },
    Added(AddFunc),
}

#[derive(Clone)]
struct ModEntry {
    new_path: String,
    source: ModSource,
    patches: Vec<PatchFunc>,
}

pub struct MultiPackTransformer {
    packs: Arc<[PackedFile]>,
    ops: Vec<PackedOp>,
}

impl MultiPackTransformer {
    pub fn new(packs: MultiPack) -> Self {
        Self {
            packs: packs.files,
            ops: vec![],
        }
    }

    pub fn delete(mut self, pattern: &str) -> Result<Self> {
        self.ops
            .push(PackedOp::Delete(glob::Pattern::new(pattern)?));
        Ok(self)
    }

    pub fn rename(mut self, pattern: &str, func: RenameFunc) -> Result<Self> {
        self.ops
            .push(PackedOp::Rename(glob::Pattern::new(pattern)?, func));
        Ok(self)
    }

    pub fn patch(mut self, pattern: &str, func: PatchFunc) -> Result<Self> {
        self.ops
            .push(PackedOp::Patch(glob::Pattern::new(pattern)?, func));
        Ok(self)
    }

    pub fn add(mut self, path: &str, func: AddFunc) -> Result<Self> {
        self.ops.push(PackedOp::Add(path.to_owned(), func));
        Ok(self)
    }

    pub fn write<P: AsRef<Path>>(self, output_dir: P) -> Result<()> {
        fs::create_dir_all(output_dir.as_ref()).with_context(|| {
            format!(
                "Failed to create output directory {}",
                output_dir.as_ref().display()
            )
        })?;

        for packed in self.packs.iter() {
            let file_name = packed
                ._path
                .file_name()
                .ok_or_else(|| anyhow!("Invalid packed file path {}", packed._path.display()))?;
            let output_path = output_dir.as_ref().join(file_name);
            Self::write_pack(packed, &self.ops, &output_path)?;
        }

        Ok(())
    }

    pub fn write_mod<P: AsRef<Path>>(self, output_path: P) -> Result<()> {
        use binrw::BinWrite;

        const MAX_PACKED_SIZE: u32 = 0x7fffffff;

        let changed = Self::collect_changed_entries(&self.packs, &self.ops)?;

        if changed.is_empty() {
            return Ok(());
        }

        let mut final_entries: Vec<(ModEntry, usize)> = Vec::with_capacity(changed.len());

        for entry in &changed {
            let mut data: Cow<[u8]> = match &entry.source {
                ModSource::Existing {
                    pack_index,
                    file_index,
                } => {
                    let packed = self
                        .packs
                        .get(*pack_index)
                        .ok_or_else(|| anyhow!("Pack index out of bounds"))?;
                    let file = packed
                        .header
                        .files
                        .get(*file_index)
                        .ok_or_else(|| anyhow!("File index out of bounds"))?;
                    Cow::Borrowed(Self::get_entry_data(packed, file)?)
                }
                ModSource::Added(generator) => Cow::Owned(generator()),
            };

            let mut should_include = false;

            for patch_func in &entry.patches {
                let data_prev = data.clone();
                (patch_func)(&entry.new_path, &mut data)
                    .context(format!("Error patching {}", entry.new_path))?;
                should_include |= data_prev != data;
            }

            should_include |= matches!(&data, Cow::Owned(_));
            if should_include {
                final_entries.push((entry.clone(), data.len()));
            }
        }

        if final_entries.is_empty() {
            return Ok(());
        }

        let output_dir = output_path.as_ref().parent().unwrap_or(Path::new("."));
        let output_stem = output_path
            .as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("mod");
        let output_ext = output_path
            .as_ref()
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("packed");

        let mut split_index = 0;
        let mut current_entries: Vec<(ModEntry, usize)> = Vec::new();
        let mut current_data_size = 0usize;
        let mut current_header_size = 0usize;

        fn write_split(
            output_dir: &Path,
            output_stem: &str,
            output_ext: &str,
            split_index: usize,
            entries: &[(ModEntry, usize)],
            packs: &[PackedFile],
        ) -> Result<()> {
            use binrw::BinWrite;

            let filename = if split_index == 0 {
                format!("{}.{}", output_stem, output_ext)
            } else {
                format!("{}.{:03}.{}", output_stem, split_index, output_ext)
            };
            let output_path = output_dir.join(filename);
            println!(
                "Writing {} entries to {}",
                entries.len(),
                output_path.display()
            );
            let header_size = PackedHeader {
                files: entries
                    .iter()
                    .map(|(entry, _)| PackedEntry {
                        path: entry.new_path.clone(),
                        size: 0,
                        offset: 0,
                    })
                    .collect(),
            }
            .size();

            let mut output_file = File::create(&output_path)
                .with_context(|| format!("Failed to create {}", output_path.display()))?;

            output_file.seek(SeekFrom::Start(header_size as u64))?;

            let mut data_offset = header_size;
            let mut final_header_entries = Vec::with_capacity(entries.len());

            for (entry, data_len) in entries {
                let offset: u32 = data_offset
                    .try_into()
                    .map_err(|_| anyhow!("Offset overflow for {}", entry.new_path))?;

                let mut data: Cow<[u8]> = match entry.source {
                    ModSource::Existing {
                        pack_index,
                        file_index,
                    } => {
                        let packed = packs
                            .get(pack_index)
                            .ok_or_else(|| anyhow!("Pack index out of bounds"))?;
                        let file = packed
                            .header
                            .files
                            .get(file_index)
                            .ok_or_else(|| anyhow!("File index out of bounds"))?;
                        Cow::Borrowed(MultiPackTransformer::get_entry_data(packed, file)?)
                    }
                    ModSource::Added(generator) => Cow::Owned(generator()),
                };

                for patch_func in &entry.patches {
                    (patch_func)(&entry.new_path, &mut data)
                        .context(format!("Error patching {}", entry.new_path))?;
                }

                output_file.write_all(data.as_ref())?;
                data_offset = data_offset
                    .checked_add(*data_len)
                    .ok_or_else(|| anyhow!("Packed file too large"))?;
                final_header_entries.push(PackedEntry {
                    path: entry.new_path.clone(),
                    size: *data_len as u32,
                    offset,
                });
            }

            output_file.seek(SeekFrom::Start(0))?;
            PackedHeader {
                files: final_header_entries,
            }
            .write_le(&mut output_file)?;

            Ok(())
        }

        for (entry, data_len) in final_entries {
            let entry_header_size = PackedHeader {
                files: vec![PackedEntry {
                    path: entry.new_path.clone(),
                    size: 0,
                    offset: 0,
                }],
            }
            .size();

            let would_exceed = (current_data_size as u32)
                .checked_add(data_len as u32)
                .map(|s| s > MAX_PACKED_SIZE)
                .unwrap_or(true)
                || current_header_size
                    .checked_add(entry_header_size)
                    .map(|s| s as u32 > MAX_PACKED_SIZE)
                    .unwrap_or(true);

            if would_exceed && !current_entries.is_empty() {
                write_split(
                    output_dir,
                    output_stem,
                    output_ext,
                    split_index,
                    &current_entries,
                    &self.packs,
                )?;
                split_index += 1;
                current_entries.clear();
                current_data_size = 0;
                current_header_size = 0;
            }

            current_entries.push((entry, data_len));
            current_data_size += data_len;
            current_header_size += entry_header_size;
        }

        if !current_entries.is_empty() {
            write_split(
                output_dir,
                output_stem,
                output_ext,
                split_index,
                &current_entries,
                &self.packs,
            )?;
        }

        Ok(())
    }

    fn collect_changed_entries(packs: &[PackedFile], ops: &[PackedOp]) -> Result<Vec<ModEntry>> {
        let mut result = Vec::new();

        for (pack_index, packed) in packs.iter().enumerate() {
            for (file_index, file_entry) in packed.header.files.iter().enumerate() {
                let mut current_path = file_entry.path.clone();
                let mut is_changed = false;
                let mut patches = Vec::new();

                for op in ops {
                    match op {
                        PackedOp::Delete(pattern) => {
                            if pattern.matches(&current_path) {
                                is_changed = true;
                                break;
                            }
                        }
                        PackedOp::Rename(pattern, func) => {
                            if pattern.matches(&current_path) {
                                current_path = func(&current_path);
                                is_changed = true;
                            }
                        }
                        PackedOp::Patch(pattern, func) => {
                            if pattern.matches(&current_path) {
                                patches.push(*func);
                                is_changed = true;
                            }
                        }
                        PackedOp::Add(_, _) => {}
                    }
                }

                if is_changed {
                    result.push(ModEntry {
                        new_path: current_path,
                        source: ModSource::Existing {
                            pack_index,
                            file_index,
                        },
                        patches,
                    });
                }
            }
        }

        for op in ops {
            if let PackedOp::Add(path, generator) = op {
                result.push(ModEntry {
                    new_path: path.clone(),
                    source: ModSource::Added(*generator),
                    patches: Vec::new(),
                });
            }
        }

        let mut seen = HashSet::with_capacity(result.len());
        for entry in &result {
            let key = entry.new_path.to_ascii_lowercase();
            if !seen.insert(key) {
                bail!("Duplicate path in mod: {}", entry.new_path);
            }
        }

        Ok(result)
    }

    fn write_pack(packed: &PackedFile, ops: &[PackedOp], output_path: &Path) -> Result<()> {
        use binrw::BinWrite;

        let mut output_file = File::create(output_path)
            .with_context(|| format!("Failed to create {}", output_path.display()))?;
        let processed_entries = Self::process_ops(packed, ops)?;

        let header_size = PackedHeader {
            files: processed_entries
                .iter()
                .map(|entry| PackedEntry {
                    path: entry.path.clone(),
                    size: 0,
                    offset: 0,
                })
                .collect(),
        }
        .size();

        output_file.seek(SeekFrom::Start(header_size as u64))?;

        let mut data_offset = header_size;
        let mut final_entries = Vec::with_capacity(processed_entries.len());

        for entry in processed_entries {
            let offset: u32 = data_offset
                .try_into()
                .map_err(|_| anyhow!("Packed offset overflow for {}", entry.path))?;

            let mut data: Cow<[u8]> = match entry.source {
                PackedDataSource::Existing(index) => {
                    let original =
                        packed.header.files.get(index).ok_or_else(|| {
                            anyhow!("Packed source index out of bounds: {}", index)
                        })?;
                    Cow::Borrowed(Self::get_entry_data(packed, original)?)
                }
                PackedDataSource::Added(generator) => Cow::Owned(generator()),
            };

            for patch in entry.patches {
                (patch.func)(&patch.path, &mut data)
                    .context(format!("Error patching {}", patch.path))?;
            }

            output_file.write_all(data.as_ref())?;
            let len = data.len();

            data_offset = data_offset
                .checked_add(len)
                .ok_or_else(|| anyhow!("Packed file is too large"))?;

            let size: u32 = len
                .try_into()
                .map_err(|_| anyhow!("Packed entry too large for {}", entry.path))?;
            final_entries.push(PackedEntry {
                path: entry.path.clone(),
                size,
                offset,
            });
        }

        output_file.seek(SeekFrom::Start(0))?;
        PackedHeader {
            files: final_entries,
        }
        .write_le(&mut output_file)?;

        Ok(())
    }

    fn process_ops(packed: &PackedFile, ops: &[PackedOp]) -> Result<Vec<ProcessedEntry>> {
        let mut result: Vec<ProcessedEntry> = packed
            .header
            .files
            .iter()
            .enumerate()
            .map(|(index, entry)| ProcessedEntry {
                path: entry.path.clone(),
                source: PackedDataSource::Existing(index),
                patches: Vec::new(),
            })
            .collect();

        for op in ops {
            match op {
                PackedOp::Delete(pattern) => {
                    result.retain(|entry| !pattern.matches(&entry.path));
                }
                PackedOp::Rename(pattern, func) => {
                    for entry in &mut result {
                        if pattern.matches(&entry.path) {
                            entry.path = func(&entry.path);
                        }
                    }
                }
                PackedOp::Patch(pattern, func) => {
                    for entry in &mut result {
                        if pattern.matches(&entry.path) {
                            entry.patches.push(PatchStep {
                                path: entry.path.clone(),
                                func: *func,
                            });
                        }
                    }
                }
                PackedOp::Add(path, generator) => {
                    result.push(ProcessedEntry {
                        path: path.clone(),
                        source: PackedDataSource::Added(*generator),
                        patches: Vec::new(),
                    });
                }
            }
        }

        let mut seen = HashSet::with_capacity(result.len());
        for entry in &result {
            let key = entry.path.to_ascii_lowercase();
            if !seen.insert(key) {
                bail!("Duplicate path after transform: {}", entry.path);
            }
        }

        Ok(result)
    }

    fn get_entry_data<'s>(packed: &'s PackedFile, entry: &PackedEntry) -> Result<&'s [u8]> {
        let data_start = entry.offset as usize;
        let data_end = data_start
            .checked_add(entry.size as usize)
            .ok_or_else(|| anyhow!("Invalid entry range for {}", entry.path))?;
        if data_end > packed.mm.len() {
            bail!("Entry out of bounds: {}", entry.path);
        }
        Ok(&packed.mm[data_start..data_end])
    }
}
