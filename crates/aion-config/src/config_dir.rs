use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// Resolve the macOS config root, moving the legacy directory on first use.
pub fn app_config_dir() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let legacy = dirs::config_dir().map(|dir| dir.join("aionrs"));
    Some(resolve_config_dir(&home, legacy))
}

fn is_missing(path: &Path) -> bool {
    matches!(fs::symlink_metadata(path), Err(err) if err.kind() == ErrorKind::NotFound)
}

fn resolve_config_dir(home: &Path, legacy: Option<PathBuf>) -> PathBuf {
    let preferred = home.join(".aionrs");
    let Some(legacy) = legacy else {
        return preferred;
    };
    if !is_missing(&preferred) {
        return preferred;
    }

    // Resolve legacy symlinks before moving so relative targets stay valid.
    let moved = match fs::symlink_metadata(&legacy) {
        Ok(metadata) if metadata.is_symlink() => {
            fs::canonicalize(&legacy).and_then(|target| std::os::unix::fs::symlink(target, &preferred))
        }
        Ok(metadata) if metadata.is_dir() => fs::rename(&legacy, &preferred),
        Ok(_) => return preferred,
        Err(error) if error.kind() == ErrorKind::NotFound => return preferred,
        Err(error) => Err(error),
    };

    if let Err(error) = moved
        && is_missing(&preferred)
    {
        tracing::warn!(target: "aion_config", from = %legacy.display(), to = %preferred.display(), %error, "could not migrate aionrs config directory");
        return legacy;
    }
    preferred
}

#[cfg(test)]
#[path = "config_dir_test.rs"]
mod config_dir_test;
