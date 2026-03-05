use std::path::PathBuf;

use color_eyre::eyre::{Context, Result, anyhow, bail};
use log::error;
use steamlocate::SteamDir;
const APP_ID: u32 = 897610;

pub(crate) fn get_steam_path() -> Result<PathBuf> {
    if let Ok(scrap_path) = std::env::var("SCRAPLAND_DIR") {
        return Ok(PathBuf::from(scrap_path));
    }
    let steam = SteamDir::locate()?;
    let (app, lib) = steam
        .find_app(APP_ID)?
        .ok_or_else(|| anyhow!("Scrapland (App ID {APP_ID}) not found in steam library"))?;
    Ok(lib.resolve_app_dir(&app))
}

pub(crate) fn get_path() -> PathBuf {
    let err = match get_steam_path() {
        Ok(path) => {
            return path;
        }
        Err(err) => err,
    };
    error!("{err}");
    let err_msg = format!("{err}");
    rfd::MessageDialog::new()
        .set_title(err_msg)
        .set_description("Please locate the Scrapland installation folder manually")
        .set_buttons(rfd::MessageButtons::Ok)
        .set_level(rfd::MessageLevel::Warning)
        .show();

    let Some(folder) = rfd::FileDialog::new()
        .set_title("Scrapland installation folder")
        .pick_folder()
    else {
        std::process::exit(1);
    };
    folder
}
