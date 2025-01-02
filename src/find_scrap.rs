use std::path::PathBuf;
use anyhow::{anyhow, Result};
use steamlocate::SteamDir;
const APP_ID: u32 = 897610;

pub(crate) fn get_path() -> Result<PathBuf> {
    let steam = SteamDir::locate()?;
    let (app,lib) = steam.find_app(APP_ID)?.ok_or_else(|| anyhow!("Scrapland (App ID {APP_ID}) not found in steam library"))?;
    Ok(lib.resolve_app_dir(&app))
}
