// ref: internal/config/plugin_config_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::path::Path;

use super::config_load::{load_config, FileConfigDocument};
use super::config_yaml::save_config_preserve_comments;
use super::parse::{parse_provider_compat_config, parse_provider_compat_config_with_root};

#[test]
fn plugin_defaults_and_paths_use_injected_data_root() {
    let default = parse_provider_compat_config(b"plugins: {}").unwrap();
    assert!(!default.plugins.enabled);
    assert_eq!(default.plugins.dir, "plugins");
    assert!(default.plugins.configs.is_empty());
    let rooted = parse_provider_compat_config_with_root(
        b"plugins:\n  dir: ~/.cli-proxy-api/plugins\n",
        Path::new("/typed/data"),
    )
    .unwrap();
    assert_eq!(rooted.plugins.dir, "/typed/data/.cli-proxy-api/plugins");
}

#[test]
fn sources_auth_and_raw_instance_are_normalized_without_env_authority() {
    let config = parse_provider_compat_config(
        br#"plugins:
  enabled: true
  store-sources: [" https://community.example/registry.json ", ""]
  auth-revision: 42
  store-auth:
    - match: " https://plugins.example.com/ "
      apply-to: [registry, artifact, registry]
      type: bearer
      token-secret: {scope: plugin-store, name: community-token}
  configs:
    sample:
      enabled: false
      priority: 7
      config1: value1
      config2: {nested: value2}
"#,
    )
    .unwrap();
    assert_eq!(
        config.plugins.store_sources,
        ["https://community.example/registry.json"]
    );
    assert_eq!(config.plugins.auth_revision, 42);
    let auth = &config.plugins.store_auth[0];
    assert_eq!(auth.match_url, "https://plugins.example.com/");
    assert_eq!(auth.apply_to, ["registry", "artifact"]);
    assert_eq!(auth.token_secret.as_ref().unwrap().name, "community-token");
    let plugin = &config.plugins.configs["sample"];
    assert_eq!(plugin.enabled, Some(false));
    assert_eq!(plugin.priority, 7);
    let raw = serde_yaml::to_string(&plugin.raw).unwrap();
    assert!(raw.contains("config1: value1"));
    assert!(raw.contains("nested: value2"));
    assert!(serde_yaml::from_str::<super::ProviderCompatConfig>(
        "plugins:\n  store-auth:\n    - match: https://x\n      token-env: TOKEN\n"
    )
    .is_err());
}

#[test]
fn typed_file_source_and_save_are_explicit_and_prune_default_plugins() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.yaml");
    std::fs::write(&path, "# retained\nplugins: {}\n").unwrap();
    let document = FileConfigDocument::new(&path);
    let config = load_config(&document, directory.path(), false).unwrap();
    save_config_preserve_comments(&document, &config).unwrap();
    let saved = std::fs::read_to_string(&path).unwrap();
    assert!(saved.starts_with("# retained\n"));
    assert!(!saved.contains("plugins:"));
}
