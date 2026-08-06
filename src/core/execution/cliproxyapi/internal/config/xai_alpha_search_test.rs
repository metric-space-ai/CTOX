// ref: internal/config/xai_alpha_search_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::config_normalization::{sanitize_xai_keys, CodexKey};

#[test]
fn sanitize_xai_keys_clears_codex_alpha_search_capability() {
    let mut keys = vec![CodexKey {
        api_key: "xai-key".into(),
        base_url: "https://api.x.ai/v1".into(),
        alpha_search: true,
        ..CodexKey::default()
    }];
    sanitize_xai_keys(&mut keys);
    assert_eq!(keys.len(), 1);
    assert!(!keys[0].alpha_search);
}
