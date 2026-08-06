// ref: internal/translator/codex/gemini/noop_optimization_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::json;

use super::codex_gemini_request::{build_short_name_map, clean_parameters};

#[test]
fn canonical_schema_is_preserved_and_missing_closed_shape_is_added() {
    let canonical = json!({"type":"object","additionalProperties":false});
    assert_eq!(clean_parameters(&canonical), canonical);
    assert_eq!(
        clean_parameters(&json!({"$schema":"draft","type":"object"})),
        json!({"type":"object","additionalProperties":false})
    );
}

#[test]
fn short_name_collisions_remain_unique_and_bounded() {
    let prefix = "x".repeat(70);
    let names = vec![format!("{prefix}a"), format!("{prefix}b")];
    let map = build_short_name_map(&names);
    assert_ne!(map[&names[0]], map[&names[1]]);
    assert!(map.values().all(|name| name.len() <= 64));
}
