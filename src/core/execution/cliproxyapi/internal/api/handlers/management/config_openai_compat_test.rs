// ref: internal/api/handlers/management/config_openai_compat_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::config::{
    CodexModel, OpenAiCompatibility, OpenAiCompatibilityApiKey, ProviderCompatConfig,
};

use super::management_openai_compatibility_views;

#[test]
fn openai_compat_view_includes_execution_flags_without_api_keys() {
    let mut config = ProviderCompatConfig {
        openai_compatibility: vec![OpenAiCompatibility {
            name: "Mimo CN".into(),
            base_url: "https://user:secret@example.invalid/v1".into(),
            api_key_entries: vec![OpenAiCompatibilityApiKey {
                api_key: "test-key".into(),
                ..OpenAiCompatibilityApiKey::default()
            }],
            models: vec![CodexModel {
                name: "mimo-v2.5".into(),
                ..CodexModel::default()
            }],
            support_prompt_cache_key: true,
            disable_cooling: true,
            ..OpenAiCompatibility::default()
        }],
        ..ProviderCompatConfig::default()
    };
    config.sanitize();
    assert_eq!(
        config.openai_compatibility[0].api_key_entries[0].api_key,
        "test-key"
    );
    assert!(config.openai_compatibility[0].support_prompt_cache_key);
    assert!(config.openai_compatibility[0].disable_cooling);
    let views = management_openai_compatibility_views(&config);
    assert!(views[0].support_prompt_cache_key);
    assert!(views[0].disable_cooling);
    assert_eq!(views[0].credential_count, 1);
    let serialized = serde_json::to_string(&views).unwrap();
    assert!(!serialized.contains("test-key"));
    assert!(!serialized.contains("user:secret"));
}
