// ref: internal/pluginhost/platform.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: discovers process executables through an injected filesystem authority
// License: MIT (upstream); modifications AGPL-3.0-only

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::config::normalize_version;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginFileInfo {
    pub id: String,
    pub path: PathBuf,
    pub version: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginPlatform {
    pub os: String,
    pub arch: String,
    pub executable_suffix: String,
}

impl PluginPlatform {
    pub fn process(os: impl Into<String>, arch: impl Into<String>) -> Self {
        let os = os.into();
        let executable_suffix = if os == "windows" {
            ".ctox-plugin.exe"
        } else {
            ".ctox-plugin"
        };
        Self {
            os,
            arch: arch.into(),
            executable_suffix: executable_suffix.to_owned(),
        }
    }
}

pub trait PluginDiscoveryFilesystem: Send + Sync {
    fn regular_file_names(&self, directory: &Path) -> Result<Vec<String>, PlatformError>;
    fn remove_file(&self, path: &Path) -> Result<(), PlatformError>;
}

pub fn validate_plugin_id(id: &str) -> bool {
    let bytes = id.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 128
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(byte))
}

pub fn plugin_file_from_name(directory: &Path, name: &str, suffix: &str) -> Option<PluginFileInfo> {
    if name.contains(['/', '\\'])
        || !name
            .to_ascii_lowercase()
            .ends_with(&suffix.to_ascii_lowercase())
    {
        return None;
    }
    let stem = &name[..name.len().checked_sub(suffix.len())?];
    let (id, version) = stem
        .rfind("-v")
        .and_then(|index| {
            let id = &stem[..index];
            let version = normalize_version(&stem[index + 2..])?;
            validate_plugin_id(id).then_some((id, Some(version)))
        })
        .unwrap_or((stem, None));
    if !validate_plugin_id(id) {
        return None;
    }
    Some(PluginFileInfo {
        id: id.to_owned(),
        path: directory.join(name),
        version,
    })
}

pub fn discover_plugin_files(
    filesystem: &dyn PluginDiscoveryFilesystem,
    root: &Path,
    platform: &PluginPlatform,
    desired: &BTreeMap<String, String>,
) -> Result<(Vec<PluginFileInfo>, Vec<PluginFileInfo>), PlatformError> {
    if !root.is_absolute() || platform.os.trim().is_empty() || platform.arch.trim().is_empty() {
        return Err(PlatformError::InvalidConfig);
    }
    let directories = [
        root.join(&platform.os).join(&platform.arch),
        root.to_path_buf(),
    ];
    let mut all = Vec::new();
    let mut selected = BTreeMap::<String, PluginFileInfo>::new();
    let mut order = Vec::new();
    for directory in directories {
        let mut names = filesystem.regular_file_names(&directory)?;
        names.sort();
        for name in names {
            let Some(candidate) =
                plugin_file_from_name(&directory, &name, &platform.executable_suffix)
            else {
                continue;
            };
            all.push(candidate.clone());
            match selected.get(&candidate.id) {
                Some(current)
                    if !plugin_file_preferred(
                        &candidate,
                        current,
                        desired.get(&candidate.id).map(String::as_str),
                    ) => {}
                Some(_) => {
                    selected.insert(candidate.id.clone(), candidate);
                }
                None => {
                    order.push(candidate.id.clone());
                    selected.insert(candidate.id.clone(), candidate);
                }
            }
        }
    }
    let selected = order
        .into_iter()
        .filter_map(|id| {
            let candidate = selected.remove(&id)?;
            let matches = desired
                .get(&id)
                .is_none_or(|version| candidate.version.as_ref() == Some(version));
            matches.then_some(candidate)
        })
        .collect();
    Ok((selected, all))
}

pub fn cleanup_unselected_plugin_files(
    filesystem: &dyn PluginDiscoveryFilesystem,
    selected: &[PluginFileInfo],
    all: &[PluginFileInfo],
) -> Result<(), PlatformError> {
    let selected: BTreeSet<_> = selected.iter().map(|file| file.path.clone()).collect();
    let loaded_ids: BTreeSet<_> = selected
        .iter()
        .filter_map(|path| all.iter().find(|file| &file.path == path))
        .map(|file| file.id.clone())
        .collect();
    for candidate in all {
        if loaded_ids.contains(&candidate.id) && !selected.contains(&candidate.path) {
            filesystem.remove_file(&candidate.path)?;
        }
    }
    Ok(())
}

fn plugin_file_preferred(
    candidate: &PluginFileInfo,
    current: &PluginFileInfo,
    desired: Option<&str>,
) -> bool {
    if let Some(desired) = desired {
        let candidate_matches = candidate.version.as_deref() == Some(desired);
        let current_matches = current.version.as_deref() == Some(desired);
        if candidate_matches != current_matches {
            return candidate_matches;
        }
    }
    match (&candidate.version, &current.version) {
        (Some(_), None) => true,
        (None, _) => false,
        (Some(candidate), Some(current)) => {
            compare_versions(candidate, current) == Ordering::Greater
        }
    }
}

fn compare_versions(left: &str, right: &str) -> Ordering {
    let left_parts: Option<Vec<u64>> = left.split('.').map(|part| part.parse().ok()).collect();
    let right_parts: Option<Vec<u64>> = right.split('.').map(|part| part.parse().ok()).collect();
    match (left_parts, right_parts) {
        (Some(left), Some(right)) => {
            let length = left.len().max(right.len());
            (0..length)
                .map(|index| {
                    left.get(index)
                        .copied()
                        .unwrap_or(0)
                        .cmp(&right.get(index).copied().unwrap_or(0))
                })
                .find(|ordering| *ordering != Ordering::Equal)
                .unwrap_or(Ordering::Equal)
        }
        _ => left.cmp(right),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlatformError {
    InvalidConfig,
    Filesystem(String),
}

impl std::fmt::Display for PlatformError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig => formatter.write_str("plugin platform config is invalid"),
            Self::Filesystem(message) => write!(formatter, "plugin discovery failed: {message}"),
        }
    }
}

impl std::error::Error for PlatformError {}
