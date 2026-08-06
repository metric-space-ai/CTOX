// ref: internal/pluginhost/platform_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: process-executable discovery uses an injected filesystem
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::platform::{
    cleanup_unselected_plugin_files, discover_plugin_files, plugin_file_from_name,
    validate_plugin_id, PlatformError, PluginDiscoveryFilesystem, PluginPlatform,
};

#[derive(Default)]
struct Filesystem {
    directories: BTreeMap<PathBuf, Vec<String>>,
    removed: Mutex<Vec<PathBuf>>,
}

impl PluginDiscoveryFilesystem for Filesystem {
    fn regular_file_names(&self, directory: &Path) -> Result<Vec<String>, PlatformError> {
        Ok(self.directories.get(directory).cloned().unwrap_or_default())
    }

    fn remove_file(&self, path: &Path) -> Result<(), PlatformError> {
        self.removed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(path.to_path_buf());
        Ok(())
    }
}

#[test]
fn identifiers_and_process_artifact_names_are_strict() {
    assert!(validate_plugin_id("provider.one-2"));
    assert!(!validate_plugin_id("../provider"));
    assert!(!validate_plugin_id(" provider"));
    let file = plugin_file_from_name(
        Path::new("/plugins"),
        "provider-v1.2.3.ctox-plugin",
        ".ctox-plugin",
    )
    .unwrap();
    assert_eq!(file.id, "provider");
    assert_eq!(file.version.as_deref(), Some("1.2.3"));
    assert!(plugin_file_from_name(
        Path::new("/plugins"),
        "../provider.ctox-plugin",
        ".ctox-plugin"
    )
    .is_none());
}

#[test]
fn platform_directory_precedes_root_and_desired_version_wins() {
    let root = PathBuf::from("/plugins");
    let platform = PluginPlatform::process("linux", "x86_64");
    let filesystem = Filesystem {
        directories: BTreeMap::from([
            (
                root.join("linux/x86_64"),
                vec![
                    "alpha-v1.0.0.ctox-plugin".to_owned(),
                    "beta.ctox-plugin".to_owned(),
                ],
            ),
            (
                root.clone(),
                vec![
                    "alpha-v2.0.0.ctox-plugin".to_owned(),
                    "alpha-v3.0.0.ctox-plugin".to_owned(),
                ],
            ),
        ]),
        ..Filesystem::default()
    };
    let desired = BTreeMap::from([("alpha".to_owned(), "2.0.0".to_owned())]);
    let (selected, all) = discover_plugin_files(&filesystem, &root, &platform, &desired).unwrap();
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].id, "alpha");
    assert_eq!(selected[0].version.as_deref(), Some("2.0.0"));
    assert_eq!(selected[1].id, "beta");

    cleanup_unselected_plugin_files(&filesystem, &selected, &all).unwrap();
    let removed: BTreeSet<_> = filesystem
        .removed
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .cloned()
        .collect();
    assert!(removed.contains(&root.join("linux/x86_64/alpha-v1.0.0.ctox-plugin")));
    assert!(removed.contains(&root.join("alpha-v3.0.0.ctox-plugin")));
    assert!(!removed.contains(&root.join("linux/x86_64/beta.ctox-plugin")));
}

#[test]
fn unresolved_root_fails_before_filesystem_authority() {
    let error = discover_plugin_files(
        &Filesystem::default(),
        Path::new("plugins"),
        &PluginPlatform::process("linux", "arm64"),
        &BTreeMap::new(),
    )
    .unwrap_err();
    assert_eq!(error, PlatformError::InvalidConfig);
}
