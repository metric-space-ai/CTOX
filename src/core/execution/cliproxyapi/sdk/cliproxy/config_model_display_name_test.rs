// ref: sdk/cliproxy/config_model_display_name_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::service_models::{build_codex_config_models, build_config_models, ConfiguredModel};
use crate::internal::registry::embedded_models_catalog;

#[test]
fn configured_display_name_and_name_fallback_are_preserved() {
    let explicit = ConfiguredModel {
        name: "claude-upstream".into(),
        alias: "claude-catalog".into(),
        display_name: "Claude Catalog Name".into(),
        ..ConfiguredModel::default()
    };
    assert_eq!(
        build_config_models(&[explicit], "anthropic", "claude")[0].display_name,
        "Claude Catalog Name"
    );
    let fallback = ConfiguredModel {
        name: "claude-upstream".into(),
        alias: "claude-catalog".into(),
        ..ConfiguredModel::default()
    };
    assert_eq!(
        build_config_models(&[fallback], "anthropic", "claude")[0].display_name,
        "claude-upstream"
    );
}

#[test]
fn codex_configured_models_replace_defaults_and_empty_uses_defaults() {
    let catalog = embedded_models_catalog().unwrap();
    let configured = build_codex_config_models(
        &[ConfiguredModel {
            name: "upstream".into(),
            alias: "configured-codex".into(),
            ..ConfiguredModel::default()
        }],
        &catalog,
    );
    assert_eq!(configured.len(), 1);
    assert_eq!(configured[0].id, "configured-codex");
    let defaults = build_codex_config_models(&[], &catalog);
    assert!(defaults.iter().any(|model| model.id == "gpt-image-1.5"));
    assert!(defaults.iter().any(|model| model.id == "gpt-image-2"));
}
