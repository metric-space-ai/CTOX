// ref: internal/watcher/diff/oauth_model_alias_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::oauth_model_alias::diff_oauth_model_alias_changes;
use crate::internal::watcher::synthesizer::context::ModelAlias;
use std::collections::BTreeMap;
#[test]
fn alias_diff_includes_display_name_changes() {
    let alias = |display: &str| ModelAlias {
        name: "model".into(),
        alias: "alias".into(),
        display_name: display.into(),
        fork: false,
    };
    let old = BTreeMap::from([("claude".into(), vec![alias("Old")])]);
    let new = BTreeMap::from([("claude".into(), vec![alias("New")])]);
    let (changes, affected) = diff_oauth_model_alias_changes(&old, &new);
    assert_eq!(changes.len(), 1);
    assert_eq!(affected, vec!["claude"]);
}
