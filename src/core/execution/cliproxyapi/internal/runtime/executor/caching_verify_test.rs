// ref: internal/runtime/executor/caching_verify_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};

use super::claude_executor_cloaking::ensure_claude_cache_control;

fn cached(value: &Value) -> bool {
    value
        .get("cache_control")
        .and_then(|cache| cache.get("type"))
        .and_then(Value::as_str)
        == Some("ephemeral")
}

#[test]
fn cache_control_covers_string_array_tools_and_independent_sections() {
    let output = ensure_claude_cache_control(
        br#"{"model":"claude-3-5-sonnet","tools":[{"name":"first"},{"name":"last"}],"system":"long prompt","messages":[]}"#,
    );
    let value: Value = serde_json::from_slice(&output).unwrap();
    assert!(!cached(&value["tools"][0]));
    assert!(cached(&value["tools"][1]));
    assert_eq!(value["system"][0]["text"], "long prompt");
    assert!(cached(&value["system"][0]));

    let output = ensure_claude_cache_control(
        br#"{"tools":[{"name":"tool","cache_control":{"type":"ephemeral"}}],"system":[{"type":"text","text":"one"},{"type":"text","text":"two"}],"messages":[]}"#,
    );
    let value: Value = serde_json::from_slice(&output).unwrap();
    assert!(cached(&value["tools"][0]));
    assert!(!cached(&value["system"][0]));
    assert!(cached(&value["system"][1]));
}

#[test]
fn cache_control_handles_empty_and_many_tool_lists() {
    let tools = (0..50)
        .map(|index| json!({"name": format!("tool{index}")}))
        .collect::<Vec<_>>();
    let input = serde_json::to_vec(&json!({
        "tools": tools,
        "system": [{"type":"text", "text":"Claude Code"}],
        "messages": [{"role":"user", "content":"hello"}]
    }))
    .unwrap();
    let value: Value = serde_json::from_slice(&ensure_claude_cache_control(&input)).unwrap();
    assert!(value["tools"]
        .as_array()
        .unwrap()
        .iter()
        .take(49)
        .all(|tool| !cached(tool)));
    assert!(cached(&value["tools"][49]));
    assert!(cached(&value["system"][0]));

    let empty = ensure_claude_cache_control(
        br#"{"tools":[],"system":"test","messages":[{"role":"user","content":"hi"}]}"#,
    );
    let empty: Value = serde_json::from_slice(&empty).unwrap();
    assert!(cached(&empty["system"][0]));
}

#[test]
fn message_cache_uses_second_last_user_and_preserves_existing_breakpoint() {
    let output = ensure_claude_cache_control(
        br#"{"messages":[{"role":"user","content":"first"},{"role":"assistant","content":"reply"},{"role":"user","content":"second"},{"role":"assistant","content":"reply 2"},{"role":"user","content":"third"}]}"#,
    );
    let value: Value = serde_json::from_slice(&output).unwrap();
    assert!(cached(&value["messages"][2]["content"][0]));
    assert!(!cached(&value["messages"][4]["content"][0]));

    let output = ensure_claude_cache_control(
        br#"{"messages":[{"role":"user","content":[{"type":"text","text":"first"}]},{"role":"assistant","content":[{"type":"text","text":"reply","cache_control":{"type":"ephemeral"}}]},{"role":"user","content":[{"type":"text","text":"second"}]}]}"#,
    );
    let value: Value = serde_json::from_slice(&output).unwrap();
    assert!(!cached(&value["messages"][0]["content"][0]));
    assert!(cached(&value["messages"][1]["content"][0]));
}

#[test]
fn deferred_tools_are_never_selected_as_cache_breakpoints() {
    for (tools, expected) in [
        (
            json!([{"name":"resident"},{"name":"deferred","defer_loading":true}]),
            Some(0),
        ),
        (
            json!([{"name":"resident"},{"name":"d1","defer_loading":true},{"name":"d2","defer_loading":true}]),
            Some(0),
        ),
        (
            json!([{"name":"r1"},{"name":"deferred","defer_loading":true},{"name":"r2"}]),
            Some(2),
        ),
        (
            json!([{"name":"d1","defer_loading":true},{"name":"d2","defer_loading":true}]),
            None,
        ),
    ] {
        let input = serde_json::to_vec(&json!({"tools": tools})).unwrap();
        let value: Value = serde_json::from_slice(&ensure_claude_cache_control(&input)).unwrap();
        for (index, tool) in value["tools"].as_array().unwrap().iter().enumerate() {
            assert_eq!(cached(tool), expected == Some(index));
        }
    }

    let existing = ensure_claude_cache_control(
        br#"{"tools":[{"name":"resident","cache_control":{"type":"ephemeral","ttl":"1h"}},{"name":"other"}]}"#,
    );
    let existing: Value = serde_json::from_slice(&existing).unwrap();
    assert_eq!(existing["tools"][0]["cache_control"]["ttl"], "1h");
    assert!(!cached(&existing["tools"][1]));
}
