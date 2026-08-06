// ref: internal/watcher/diff/oauth_excluded_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::oauth_excluded::{
    diff_oauth_excluded_model_changes, summarize_excluded_models, summarize_oauth_excluded_models,
};
use std::collections::BTreeMap;
#[test]
fn excluded_models_normalize_and_diff_affected_providers() {
    let summary = summarize_excluded_models(&[" GPT-4 ".into(), "gpt-4".into(), "Claude".into()]);
    assert_eq!(summary.count, 2);
    let old = BTreeMap::from([(" CLAUDE ".into(), vec!["one".into()])]);
    let new = BTreeMap::from([("claude".into(), vec!["two".into()])]);
    assert!(summarize_oauth_excluded_models(&old).contains_key("claude"));
    let (changes, affected) = diff_oauth_excluded_model_changes(&old, &new);
    assert_eq!(changes.len(), 1);
    assert_eq!(affected, vec!["claude"]);
}
