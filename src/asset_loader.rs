use std::{path::Path, sync::Arc};

use color_eyre::eyre::{Context, Result, anyhow, bail};
use bevy::{
    asset::{
        AssetLoader, AsyncReadExt, VisitAssetDependencies,
        io::{
            AssetReader, AssetReaderError, AssetReaderFuture, AssetSource, AssetSourceId, Reader,
            SliceReader,
        },
    },
    prelude::*,
};
use vfs::{FileSystem, SeekAndRead, error::VfsErrorKind};

use crate::{
    packed_vfs::{DirectoryTree, FileHandle, MultiPack},
    parser::multi_pack_fs::MultiPackFS,
};

#[derive(Clone, Deref, DerefMut, Resource)]
pub(crate) struct PackedAssetRepository(pub(crate) Arc<MultiPack>);

pub(crate) struct PackedAssetRepositoryPlugin {
    repository: PackedAssetRepository,
}

impl PackedAssetRepositoryPlugin {
    pub(crate) fn new(mp: MultiPack) -> Self {
        PackedAssetRepositoryPlugin {
            repository: PackedAssetRepository(Arc::new(mp)),
        }
    }
}

impl Plugin for PackedAssetRepositoryPlugin {
    fn build(&self, app: &mut App) {
        let repository_1 = self.repository.clone();
        let repository_2 = self.repository.clone();
        app.insert_resource(self.repository.clone());
        app.register_asset_source(
            AssetSourceId::Name("packed".into()),
            AssetSource::build()
                .with_reader(move || Box::new(repository_1.clone()))
                .with_processed_reader(move || Box::new(repository_2.clone())),
        );
    }
}

impl AssetReader for PackedAssetRepository {
    async fn read<'a>(
        &'a self,
        path: &'a Path,
    ) -> Result<impl bevy::asset::io::Reader + 'a, AssetReaderError> {
        let path = format!("{}", path.display());
        println!("[READ] {path}");
        let res = self
            .get_file(&path)
            .map_err(|e| AssetReaderError::Io(Arc::new(std::io::Error::other(e))))?;
        Ok(res)
    }

    async fn read_meta<'a>(
        &'a self,
        path: &'a Path,
    ) -> Result<impl bevy::asset::io::Reader + 'a, AssetReaderError> {
        let path_str = format!("{}", path.display());
        println!("[READ_META] {path_str}");
        Result::<Box<dyn bevy::asset::io::Reader>, AssetReaderError>::Err(
            AssetReaderError::NotFound(path.to_owned()),
        )
    }

    async fn read_directory<'a>(
        &'a self,
        path: &'a Path,
    ) -> std::result::Result<Box<bevy::asset::io::PathStream>, AssetReaderError> {
        let path = format!("{}", path.display());
        println!("[READ_DIR] {path}");
        let res = self
            .tree
            .get_entry(&path)
            .map_err(|e| AssetReaderError::Io(Arc::new(std::io::Error::other(e))))?;
        match res {
            DirectoryTree::File { .. } => Err(AssetReaderError::Io(Arc::new(std::io::Error::new(
                std::io::ErrorKind::NotADirectory,
                path,
            )))),
            DirectoryTree::Directory { entries } => {
                let keys = entries
                    .keys()
                    .map(std::path::PathBuf::from)
                    .collect::<Vec<_>>();
                let stream = bevy::tasks::futures_lite::stream::iter(keys);
                let stream: Box<bevy::asset::io::PathStream> = Box::new(stream);
                Ok(stream)
            }
        }
    }

    async fn is_directory<'a>(&'a self, path: &'a Path) -> Result<bool, AssetReaderError> {
        let path = format!("{}", path.display());
        println!("[IS_DIR] {path}");
        let res = self
            .0
            .tree
            .get_entry(&path)
            .map_err(|e| AssetReaderError::Io(Arc::new(std::io::Error::other(e))))?;
        Ok(matches!(res, DirectoryTree::Directory { .. }))
    }
}

#[derive(Debug, Asset)]
pub(crate) struct TestAsset(Vec<u8>);

impl TypePath for TestAsset {
    fn type_path() -> &'static str {
        "scrap/test"
    }

    fn short_type_path() -> &'static str {
        "scrap/test"
    }
}

pub(crate) struct TestLoader;

impl AssetLoader for TestLoader {
    type Asset = TestAsset;

    type Settings = ();

    type Error = color_eyre::eyre::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &Self::Settings,
        load_context: &mut bevy::asset::LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        println!("[LOAD_ASSET] {path:?}!", path = load_context.asset_path());
        let mut data = vec![];
        reader.read_to_end(&mut data).await?;
        Ok(TestAsset(data[..0x10].to_vec()))
    }

    fn extensions(&self) -> &[&str] {
        &["emi"]
    }
}
