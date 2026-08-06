// ref: internal/runtime/executor/claude_executor_request_remap_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::remap_claude_oauth_tool_names_with_secret;

#[test]
fn declared_tool_and_history_receive_same_mcp_alias() {
    let body = br#"{"tools":[{"name":"fetch_url"}],"messages":[{"role":"assistant","content":[{"type":"tool_use","name":"fetch_url"}]}]}"#;
    let (mapped, reverse) = remap_claude_oauth_tool_names_with_secret(body, "caller-a");
    let root: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
    let alias = root
        .pointer("/tools/0/name")
        .and_then(serde_json::Value::as_str)
        .unwrap();
    assert!(alias.starts_with("mcp__"));
    assert_eq!(
        root.pointer("/messages/0/content/0/name")
            .and_then(serde_json::Value::as_str),
        Some(alias)
    );
    assert_eq!(reverse.get(alias).map(String::as_str), Some("fetch_url"));
}

#[test]
fn typed_custom_tools_are_aliased_but_server_tools_remain_native() {
    let body = br#"{"tools":[{"name":"custom","type":"custom"},{"name":"native","type":"web_search_20250305"}],"messages":[{"role":"assistant","content":[{"type":"tool_use","name":"custom"},{"type":"tool_use","name":"native"}]}]}"#;
    let (mapped, reverse) = remap_claude_oauth_tool_names_with_secret(body, "caller-a");
    let root: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
    let alias = root
        .pointer("/tools/0/name")
        .and_then(serde_json::Value::as_str)
        .unwrap();
    assert!(alias.starts_with("mcp__"));
    assert_eq!(root["tools"][1]["name"], "native");
    assert_eq!(root["messages"][0]["content"][0]["name"], alias);
    assert_eq!(root["messages"][0]["content"][1]["name"], "native");
    assert_eq!(reverse.get(alias).map(String::as_str), Some("custom"));
    assert!(!reverse.values().any(|name| name == "native"));
}

#[test]
fn server_tool_name_protects_ambiguous_history_reference() {
    let body = br#"{"tools":[{"name":"shared","type":"custom"},{"name":"shared","type":"web_search_20250305"}],"messages":[{"role":"assistant","content":[{"type":"tool_use","name":"shared"}]}]}"#;
    let (mapped, reverse) = remap_claude_oauth_tool_names_with_secret(body, "caller-a");
    let root: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
    assert_ne!(root["tools"][0]["name"], "shared");
    assert_eq!(root["tools"][1]["name"], "shared");
    assert_eq!(root["messages"][0]["content"][0]["name"], "shared");
    assert_eq!(reverse.len(), 1);
}
