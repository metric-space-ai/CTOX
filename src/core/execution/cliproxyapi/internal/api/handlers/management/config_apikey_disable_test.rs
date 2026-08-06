// ref: internal/api/handlers/management/config_apikey_disable_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::config::{CodexKey, ProviderCompatConfig};
use crate::internal::watcher::synthesizer::helpers::StableIdGenerator;

use super::{set_config_api_key_excluded_all, toggle_config_api_key_excluded_all};

#[test]
fn excluded_all_is_normalized_and_reversible() {
    assert_eq!(
        set_config_api_key_excluded_all(&[" GPT-5 ".into()], true),
        ["gpt-5", "*"]
    );
    assert_eq!(
        set_config_api_key_excluded_all(&["gpt-5".into(), " * ".into()], false),
        ["gpt-5"]
    );
}

#[test]
fn xai_and_codex_toggle_only_the_publicly_selected_credential() {
    let mut config = ProviderCompatConfig {
        xai_api_key: vec![CodexKey {
            api_key: "xai-test".into(),
            base_url: "https://api.x.ai/v1".into(),
            ..CodexKey::default()
        }],
        codex_api_key: vec![CodexKey {
            api_key: "sk-test".into(),
            base_url: "https://example.com/v1".into(),
            ..CodexKey::default()
        }],
        ..ProviderCompatConfig::default()
    };
    let mut ids = StableIdGenerator::default();
    let (xai_id, _) = ids.next("xai:apikey", &["xai-test", "https://api.x.ai/v1"]);
    assert!(toggle_config_api_key_excluded_all(&mut config, &xai_id, true).unwrap());
    assert_eq!(config.xai_api_key[0].excluded_models, ["*"]);
    assert!(config.codex_api_key[0].excluded_models.is_empty());

    let mut ids = StableIdGenerator::default();
    let (codex_id, _) = ids.next("codex:apikey", &["sk-test", "https://example.com/v1"]);
    assert!(toggle_config_api_key_excluded_all(&mut config, &codex_id, true).unwrap());
    assert!(toggle_config_api_key_excluded_all(&mut config, &codex_id, false).unwrap());
    assert!(config.codex_api_key[0].excluded_models.is_empty());
}
