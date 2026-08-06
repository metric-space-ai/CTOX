// ref: internal/client/claude/models/models_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Map, Value};

use super::{
    build_response, ensure_claude_model_id_prefix, resolve_claude_model_id_prefix, ClaudeModel,
};

fn model(value: Value) -> ClaudeModel {
    value.as_object().cloned().unwrap_or_else(Map::new)
}

#[test]
fn build_response_clones_cloaks_and_sorts_models() {
    let available = vec![
        model(json!({"id":"claude-z","display_name":"Zebra","max_tokens":64000})),
        model(json!({"id":"gpt-4o","display_name":"Alpha"})),
        model(json!({"id":"claude-c","display_name":"Alpha"})),
        model(json!({"id":"claude-b","display_name":"Beta"})),
    ];

    let response = build_response(&available, false);
    let ids = response
        .data
        .iter()
        .map(|entry| entry["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        [
            "claude-c",
            "claude-fable-5-dd-o4-tpg",
            "claude-b",
            "claude-z"
        ]
    );
    assert_eq!(response.data[3]["max_tokens"], 64_000);
    assert!(!response.has_more);
    assert_eq!(response.first_id, "claude-c");
    assert_eq!(response.last_id, "claude-z");
    assert_eq!(available[1]["id"], "gpt-4o");
    assert_eq!(available[0]["id"], "claude-z");
}

#[test]
fn build_response_can_disable_cloaking_and_handles_empty_catalog() {
    let available = vec![model(json!({"id":"gpt-4o","display_name":"GPT-4o"}))];
    let response = build_response(&available, true);
    assert_eq!(response.data[0]["id"], "gpt-4o");
    assert_eq!(response.first_id, "gpt-4o");
    assert_eq!(response.last_id, "gpt-4o");

    let empty = build_response(&[], false);
    assert!(empty.data.is_empty());
    assert!(empty.first_id.is_empty());
    assert!(empty.last_id.is_empty());
}

#[test]
fn ensure_prefix_matches_pinned_unicode_safe_encoding() {
    let cases = [
        ("", ""),
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("my-claude-custom", "claude-fable-5-dd-motsuc-edualc-ym"),
        ("Claude-Opus-4", "claude-fable-5-dd-4-supO-edualC"),
        ("gpt-4o", "claude-fable-5-dd-o4-tpg"),
        ("gemini-2.5-pro", "claude-fable-5-dd-orp-5.2-inimeg"),
        ("mødel-猫", "claude-fable-5-dd-猫-ledøm"),
    ];
    for (input, expected) in cases {
        assert_eq!(ensure_claude_model_id_prefix(input), expected);
    }
}

#[test]
fn resolve_prefix_preserves_thinking_suffix_and_round_trips() {
    let cases = [
        ("", ""),
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("gpt-4o", "gpt-4o"),
        ("claude-fable-5-dd-o4-tpg", "gpt-4o"),
        ("claude-fable-5-dd-orp-5.2-inimeg", "gemini-2.5-pro"),
        ("claude-fable-5-dd-", "claude-fable-5-dd-"),
        ("claude-fable-5-dd-o4-tpg(high)", "gpt-4o(high)"),
    ];
    for (input, expected) in cases {
        assert_eq!(resolve_claude_model_id_prefix(input), expected);
    }
    let encoded = ensure_claude_model_id_prefix("custom-model-x");
    assert_eq!(resolve_claude_model_id_prefix(&encoded), "custom-model-x");
}
