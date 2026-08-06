// ref: internal/api/handlers/management/config_codex_alpha_search_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::config::{CodexKey, ProviderCompatConfig};

use super::{patch_management_provider_key, ManagementProviderKeyKind, ManagementProviderKeyPatch};

#[test]
fn patch_codex_key_updates_alpha_search_without_replacing_secret_fields() {
    let mut config = ProviderCompatConfig {
        codex_api_key: vec![CodexKey {
            api_key: "codex-key".into(),
            base_url: "https://codex.example.com".into(),
            ..CodexKey::default()
        }],
        ..ProviderCompatConfig::default()
    };
    patch_management_provider_key(
        &mut config,
        ManagementProviderKeyKind::Codex,
        0,
        ManagementProviderKeyPatch {
            alpha_search: Some(true),
            ..ManagementProviderKeyPatch::default()
        },
    )
    .unwrap();
    assert!(config.codex_api_key[0].alpha_search);
    assert_eq!(config.codex_api_key[0].api_key, "codex-key");
}
