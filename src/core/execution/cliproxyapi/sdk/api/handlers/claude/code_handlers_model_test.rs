// ref: sdk/api/handlers/claude/code_handlers_model_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Map, Value};

use super::{claude_models_response, rewrite_claude_dd_model_in_body};
use crate::internal::client::claude::models::ClaudeModel;

fn model(value: Value) -> ClaudeModel {
    value.as_object().cloned().unwrap_or_else(Map::new)
}

#[test]
fn claude_models_response_uses_configured_display_name() {
    let catalog = vec![model(json!({
        "id":"claude-display-name-catalog-test",
        "object":"model",
        "owned_by":"test",
        "display_name":"Configured Claude Name"
    }))];

    let response = claude_models_response(&catalog, false);
    assert_eq!(response.status(), 200);
    let payload: Value = serde_json::from_slice(response.body()).unwrap();
    let configured = payload["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["id"] == "claude-display-name-catalog-test")
        .expect("configured model");
    assert_eq!(configured["display_name"], "Configured Claude Name");
}

#[test]
fn claude_models_response_disables_model_list_cloaking() {
    let catalog = vec![model(json!({
        "id":"gpt-disable-model-list-cloaking-test",
        "object":"model",
        "owned_by":"test"
    }))];

    let response = claude_models_response(&catalog, true);
    let payload: Value = serde_json::from_slice(response.body()).unwrap();
    assert!(payload["data"]
        .as_array()
        .unwrap()
        .iter()
        .any(|entry| entry["id"] == "gpt-disable-model-list-cloaking-test"));
}

#[test]
fn rewrite_claude_dd_model_in_body_matches_pinned_alias_contract() {
    let cases = [
        (
            br#"{"model":"claude-fable-5-dd-o4-tpg","messages":[]}"#.as_slice(),
            "gpt-4o",
        ),
        (
            br#"{"model":"claude-sonnet-4-6","messages":[]}"#.as_slice(),
            "claude-sonnet-4-6",
        ),
        (
            br#"{"model":"claude-fable-5-dd-o4-tpg(high)","stream":true}"#.as_slice(),
            "gpt-4o(high)",
        ),
        (br#"{"messages":[]}"#.as_slice(), ""),
    ];

    for (body, expected) in cases {
        let rewritten = rewrite_claude_dd_model_in_body(body);
        let parsed: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(
            parsed
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            expected
        );
    }
}

#[test]
fn rewrite_keeps_invalid_and_plain_json_byte_identical() {
    for body in [
        br#"not-json"#.as_slice(),
        br#"{ "model": "claude-sonnet-4-6", "messages": [] }"#.as_slice(),
        br#"{ "messages": [] }"#.as_slice(),
    ] {
        assert_eq!(rewrite_claude_dd_model_in_body(body), body);
    }
}
