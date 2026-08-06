// ref: sdk/cliproxy/service_oauth_model_alias_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth::AuthKind;
use super::service_models::{
    apply_oauth_model_alias_for_auth, OAuthModelAlias, ServiceModelConfig,
};
use crate::internal::registry::RegistryModelInfo;
use std::collections::BTreeMap;

fn model() -> RegistryModelInfo {
    RegistryModelInfo {
        id: "gpt-5".into(),
        name: "models/gpt-5".into(),
        display_name: "Upstream GPT Five".into(),
        ..RegistryModelInfo::default()
    }
}

#[test]
fn oauth_alias_rename_rewrites_id_name_and_display_name() {
    let mut config = ServiceModelConfig::default();
    config.oauth_model_alias.insert(
        "codex".into(),
        vec![OAuthModelAlias {
            name: "gpt-5".into(),
            alias: "g5".into(),
            display_name: "Configured GPT Five".into(),
            fork: false,
        }],
    );
    let out = apply_oauth_model_alias_for_auth(
        &config,
        "codex",
        Some(AuthKind::OAuth),
        &BTreeMap::new(),
        vec![model()],
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, "g5");
    assert_eq!(out[0].name, "models/g5");
    assert_eq!(out[0].display_name, "Configured GPT Five");
}

#[test]
fn fork_keeps_original_and_adds_multiple_aliases() {
    let mut config = ServiceModelConfig::default();
    config.oauth_model_alias.insert(
        "codex".into(),
        vec![
            OAuthModelAlias {
                name: "gpt-5".into(),
                alias: "g5".into(),
                fork: true,
                ..OAuthModelAlias::default()
            },
            OAuthModelAlias {
                name: "gpt-5".into(),
                alias: "g5-2".into(),
                fork: true,
                ..OAuthModelAlias::default()
            },
        ],
    );
    let out = apply_oauth_model_alias_for_auth(
        &config,
        "codex",
        Some(AuthKind::OAuth),
        &BTreeMap::new(),
        vec![model()],
    );
    assert_eq!(
        out.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        ["gpt-5", "g5", "g5-2"]
    );
}

#[test]
fn api_key_skips_alias_and_per_auth_alias_precedes_global() {
    let mut config = ServiceModelConfig::default();
    config.oauth_model_alias.insert(
        "codex".into(),
        vec![OAuthModelAlias {
            name: "gpt-5".into(),
            alias: "global".into(),
            ..OAuthModelAlias::default()
        }],
    );
    let unchanged = apply_oauth_model_alias_for_auth(
        &config,
        "codex",
        Some(AuthKind::ApiKey),
        &BTreeMap::new(),
        vec![model()],
    );
    assert_eq!(unchanged[0].id, "gpt-5");
    let attrs = BTreeMap::from([(
        "model_aliases".into(),
        r#"[{"name":"gpt-5","alias":"g5","display-name":"Configured"}]"#.into(),
    )]);
    let out = apply_oauth_model_alias_for_auth(
        &config,
        "codex",
        Some(AuthKind::OAuth),
        &attrs,
        vec![model()],
    );
    assert_eq!(out[0].id, "g5");
    assert_eq!(out[0].display_name, "Configured");
}
