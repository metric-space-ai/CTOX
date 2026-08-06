// ref: internal/pluginhost/config_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: validates normalized typed config without ambient directory resolution
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::internal::config::config_normalization::{PluginInstanceConfig, PluginsConfig};

use super::config::{
    desired_plugin_versions, normalize_version, runtime_config_from_config, ConfigError,
};

#[test]
fn disabled_config_has_no_runtime_items_or_path_authority() {
    let config = runtime_config_from_config(&PluginsConfig::default()).unwrap();
    assert!(!config.enabled);
    assert!(config.items.is_empty());
    assert_eq!(config.directory, PathBuf::from("plugins"));
}

#[test]
fn enabled_config_requires_pre_resolved_absolute_directory() {
    let config = PluginsConfig {
        enabled: true,
        dir: "plugins".to_owned(),
        ..PluginsConfig::default()
    };
    assert_eq!(
        runtime_config_from_config(&config),
        Err(ConfigError::UnresolvedPluginDirectory)
    );
}

#[test]
fn items_are_sorted_preserve_raw_values_and_normalize_store_version() {
    let raw = serde_yaml::from_str(
        "custom: keep\nstore:\n  release-tag: V1.2.3\nenabled: true\npriority: 7\n",
    )
    .unwrap();
    let config = PluginsConfig {
        enabled: true,
        dir: "/ctox/plugins".to_owned(),
        configs: BTreeMap::from([
            ("zeta".to_owned(), PluginInstanceConfig::default()),
            (
                "alpha".to_owned(),
                PluginInstanceConfig {
                    enabled: Some(true),
                    priority: 7,
                    raw,
                },
            ),
        ]),
        ..PluginsConfig::default()
    };
    let runtime = runtime_config_from_config(&config).unwrap();
    assert_eq!(
        runtime.items.keys().next().map(String::as_str),
        Some("alpha")
    );
    assert!(String::from_utf8_lossy(&runtime.items["alpha"].config_yaml).contains("custom: keep"));
    assert_eq!(
        desired_plugin_versions(&runtime),
        BTreeMap::from([("alpha".to_owned(), "1.2.3".to_owned())])
    );
}

#[test]
fn version_normalization_is_fail_closed() {
    assert_eq!(normalize_version(" v2.0.1 "), Some("2.0.1".to_owned()));
    assert_eq!(normalize_version("latest"), None);
    assert_eq!(normalize_version("1/../../bad"), None);
}
