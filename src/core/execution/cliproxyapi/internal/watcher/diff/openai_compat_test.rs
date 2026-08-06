// ref: internal/watcher/diff/openai_compat_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::openai_compat::{diff_openai_compatibility, openai_compat_key, openai_compat_signature};
use crate::internal::watcher::config_reload::{ModelRoute, OpenAiCompatibility};
#[test]
fn compatibility_diff_handles_duplicate_names_and_redacts_endpoint_credentials() {
    let entry = OpenAiCompatibility {
        name: "same".into(),
        base_url: "https://user:secret@example.com/v1".into(),
        api_keys: vec!["secret-key".into()],
        models: vec![ModelRoute {
            name: "gpt".into(),
            ..ModelRoute::default()
        }],
        ..OpenAiCompatibility::default()
    };
    let mut changed = entry.clone();
    changed.prompt_cache_key = "new".into();
    let changes =
        diff_openai_compatibility(&[entry.clone(), entry.clone()], &[entry.clone(), changed]);
    assert_eq!(changes.len(), 1);
    assert!(!changes[0].contains("secret"));
    assert_eq!(openai_compat_key(&entry, 0), "same");
    assert!(!openai_compat_signature(&entry).is_empty());
}
