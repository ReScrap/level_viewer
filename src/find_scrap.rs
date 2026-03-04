use std::path::PathBuf;

use steamlocate::SteamDir;
const APP_ID: u32 = 897610;

pub(crate) fn get_path() -> Option<PathBuf> {
    let mut steam = SteamDir::locate()?;
    let app = steam.app(&APP_ID)?;
    Some(app.path.clone())
}
