// ref: internal/watcher/diff/config_diff_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::config_diff::{build_config_change_details, display_optional_value, trim_strings};
use crate::internal::watcher::config_reload::{ApiKeyConfig, WatcherConfig};

#[test]
fn config_diff_is_nil_safe_counted_and_secret_redacted() {
    let old = WatcherConfig::default();
    let mut new = WatcherConfig::default();
    new.providers.insert(
        "claude".into(),
        vec![ApiKeyConfig {
            api_key: "top-secret".into(),
            ..ApiKeyConfig::default()
        }],
    );
    let changes = build_config_change_details(&old, &new);
    assert_eq!(changes.len(), 1);
    assert!(changes[0].contains("credentials 0 -> 1"));
    assert!(!changes[0].contains("top-secret"));
    assert!(build_config_change_details(&new, &new).is_empty());
}
#[test]
fn config_diff_helpers_normalize_without_disclosing_values() {
    assert_eq!(
        trim_strings(&[" a ".into(), " ".into(), "b".into()]),
        vec!["a", "b"]
    );
    assert_eq!(display_optional_value(""), "<unset>");
    assert_eq!(display_optional_value("secret"), "<set>");
}
