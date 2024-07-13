use std::{
    collections::{HashMap, VecDeque},
    io::{Cursor, Read, Seek},
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
use binrw::{io::BufReader, BinReaderExt};
use fs_err as fs;
use memmap2::Mmap;
use serde::Serialize;
use vfs::{error::VfsErrorKind, FileSystem, SeekAndWrite, VfsMetadata};

use crate::parser::{PackedEntry, PackedHeader};

#[derive(Debug)]
pub(crate) struct PackedFile {
    _fh: fs::File,
    mm: Arc<Mmap>,
    _path: PathBuf,
}

#[derive(Debug)]
pub(crate) struct MultiPack {
    files: Vec<PackedFile>,
    pub(crate) tree: DirectoryTree,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub(crate) enum DirectoryTree {
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

    fn get_file(&self, path: &str) -> vfs::VfsResult<FileHandle> {
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

    pub(crate) fn get_entry(&self, path: &str) -> vfs::VfsResult<&Self> {
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
struct FileHandle {
    _mm: Arc<Mmap>,
    cursor: Cursor<Arc<[u8]>>,
    data: Range<usize>,
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
