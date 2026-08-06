// ref: sdk/cliproxy/auth/api_key_model_alias_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::config::{CodexKey, CodexModel, ProviderCompatConfig};

use super::api_key_model_capabilities_test::{auth, manager, register};

#[test]
fn lookup_preserves_suffix_and_config_suffix_priority() {
    let manager = manager();
    manager.set_provider_config(&ProviderCompatConfig {
        gemini_api_key: vec![CodexKey {
            api_key: "gemini-secret".into(),
            models: vec![
                CodexModel {
                    name: "gemini-pro".into(),
                    alias: "g25p".into(),
                    ..CodexModel::default()
                },
                CodexModel {
                    name: "gemini-flash(low)".into(),
                    alias: "g25f".into(),
                    ..CodexModel::default()
                },
            ],
            ..CodexKey::default()
        }],
        ..ProviderCompatConfig::default()
    });
    let mut auth = auth("gemini-auth", "gemini-secret");
    auth.provider = "gemini".into();
    register(&manager, auth);
    assert_eq!(
        manager.lookup_api_key_upstream_model("gemini-auth", "G25P(8192)"),
        "gemini-pro(8192)"
    );
    assert_eq!(
        manager.lookup_api_key_upstream_model("gemini-auth", "g25f(high)"),
        "gemini-flash(low)"
    );
    assert!(manager
        .lookup_api_key_upstream_model("missing", "g25p")
        .is_empty());
}

#[test]
fn hot_reload_replaces_new_lookups() {
    let manager = manager();
    let config = |name: &str| ProviderCompatConfig {
        gemini_api_key: vec![CodexKey {
            api_key: "reload-secret".into(),
            models: vec![CodexModel {
                name: name.into(),
                alias: "public".into(),
                ..CodexModel::default()
            }],
            ..CodexKey::default()
        }],
        ..ProviderCompatConfig::default()
    };
    manager.set_provider_config(&config("old"));
    let mut auth = auth("reload-alias", "reload-secret");
    auth.provider = "gemini".into();
    register(&manager, auth);
    assert_eq!(
        manager.lookup_api_key_upstream_model("reload-alias", "public"),
        "old"
    );
    manager.set_provider_config(&config("new"));
    assert_eq!(
        manager.lookup_api_key_upstream_model("reload-alias", "public"),
        "new"
    );
}

#[test]
fn force_mapping_uses_config_alias_and_direct_name_is_passthrough() {
    let manager = manager();
    manager.set_provider_config(&ProviderCompatConfig {
        claude_api_key: vec![CodexKey {
            api_key: "force-secret".into(),
            models: vec![CodexModel {
                name: "glm-5.2".into(),
                alias: "claude-sonnet-latest".into(),
                force_mapping: true,
                ..CodexModel::default()
            }],
            ..CodexKey::default()
        }],
        ..ProviderCompatConfig::default()
    });
    let auth = register(&manager, auth("force", "force-secret"));
    let result =
        manager.resolve_api_key_model_alias_with_result(&auth, "claude-sonnet-latest(high)");
    assert_eq!(result.upstream_model, "glm-5.2(high)");
    assert!(result.force_mapping);
    assert_eq!(result.original_alias, "claude-sonnet-latest");
    let direct = manager.resolve_api_key_model_alias_with_result(&auth, "glm-5.2");
    assert_eq!(direct.upstream_model, "glm-5.2");
    assert!(!direct.force_mapping);
}

#[test]
fn same_base_force_mapping_preserves_requested_suffix() {
    let manager = manager();
    manager.set_provider_config(&ProviderCompatConfig {
        gemini_api_key: vec![CodexKey {
            api_key: "same-base-secret".into(),
            models: vec![CodexModel {
                name: "gemini-pro".into(),
                alias: "gemini-pro(8192)".into(),
                force_mapping: true,
                ..CodexModel::default()
            }],
            ..CodexKey::default()
        }],
        ..ProviderCompatConfig::default()
    });
    let mut auth = auth("same-base", "same-base-secret");
    auth.provider = "gemini".into();
    let auth = register(&manager, auth);
    let result = manager.resolve_api_key_model_alias_with_result(&auth, "gemini-pro(8192)");
    assert_eq!(result.upstream_model, "gemini-pro(8192)");
}
