// ref: internal/api/handlers/management/config_xai_key_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::config::{CodexKey, ProviderCompatConfig};

use super::{patch_management_provider_key, ManagementProviderKeyKind, ManagementProviderKeyPatch};

#[test]
fn patch_xai_key_updates_execution_fields() {
    let mut config = ProviderCompatConfig {
        xai_api_key: vec![CodexKey {
            api_key: "xai-key".into(),
            priority: 1,
            base_url: "https://api.x.ai/v1".into(),
            websockets: true,
            ..CodexKey::default()
        }],
        ..ProviderCompatConfig::default()
    };
    patch_management_provider_key(
        &mut config,
        ManagementProviderKeyKind::Xai,
        0,
        ManagementProviderKeyPatch {
            priority: Some(7),
            websockets: Some(false),
            disable_cooling: Some(true),
            alpha_search: None,
        },
    )
    .unwrap();
    let entry = &config.xai_api_key[0];
    assert_eq!(entry.priority, 7);
    assert!(!entry.websockets);
    assert!(entry.disable_cooling);
}
