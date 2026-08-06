// ref: sdk/api/handlers/openai/openai_responses_websocket_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};

use super::*;

#[test]
fn close_reason_is_utf8_safe_and_policy_mapped() {
    let reason = "ä".repeat(100);
    let truncated = truncate_websocket_close_reason(&reason, 123);
    assert!(truncated.len() <= 123);
    assert!(websocket_close_payload_for_upstream_error(429, "limited").is_some());
    assert!(websocket_close_payload_for_upstream_error(200, "ok").is_none());
}

#[test]
fn native_passthrough_requires_mode_transport_and_pinned_auth_match() {
    assert!(responses_websocket_native_passthrough_allowed(
        "native", true, "auth-1", "auth-1"
    ));
    assert!(!responses_websocket_native_passthrough_allowed(
        "native", true, "auth-1", "auth-2"
    ));
    assert!(!responses_websocket_native_passthrough_allowed(
        "sse", true, "", "auth-2"
    ));
}

#[test]
fn websocket_chunks_extract_json_from_raw_and_sse_frames() {
    assert_eq!(
        websocket_json_payloads_from_chunk(br#"{"type":"x"}"#).len(),
        1
    );
    assert_eq!(
        websocket_json_payloads_from_chunk(b"event: x\ndata: {\"type\":\"x\"}\n\ndata: [DONE]\n")
            .len(),
        1
    );
    assert_eq!(
        sorted_string_set(&BTreeSet::from(["b".to_owned(), "a".to_owned()])),
        ["a", "b"]
    );
}

#[test]
fn prewarm_and_request_normalization_are_fail_closed() {
    let payloads =
        synthetic_responses_websocket_prewarm_payloads(br#"{"request_id":"r1"}"#).unwrap();
    assert_eq!(payloads.len(), 2);
    assert!(String::from_utf8_lossy(&payloads[1]).contains("response.completed"));
    let request: serde_json::Value = serde_json::from_slice(
        &normalize_responses_websocket_passthrough_request(br#"{"input":[]}"#, "gpt-5").unwrap(),
    )
    .unwrap();
    assert_eq!(request["model"], "gpt-5");
    assert_eq!(request["stream"], true);
    assert!(normalize_responses_websocket_passthrough_request(b"[]", "gpt-5").is_err());
}

#[test]
fn input_deduplication_preserves_first_item_order() {
    let output: serde_json::Value =
        serde_json::from_slice(&dedupe_responses_websocket_input_items_by_id(
            br#"{"input":[{"id":"a","v":1},{"id":"a","v":2},{"id":"b"}]}"#,
        ))
        .unwrap();
    assert_eq!(output["input"].as_array().unwrap().len(), 2);
    assert_eq!(output["input"][0]["v"], 1);
}

#[test]
fn session_capability_and_model_resolution_are_typed() {
    let attributes = BTreeMap::from([(
        "responses_websocket_incremental_input".to_owned(),
        "true".to_owned(),
    )]);
    assert!(websocket_upstream_supports_incremental_input(
        &attributes,
        &BTreeMap::new()
    ));
    assert_eq!(
        responses_websocket_resolved_model_name("gpt-5(high)"),
        "gpt-5"
    );
}

#[test]
fn timeline_is_instance_owned_and_disabled_by_default() {
    let mut disabled = WebsocketTimeline::default();
    disabled.append("request", b"{}", 1);
    assert!(disabled.body().is_empty());
    let mut enabled = WebsocketTimeline::new(true);
    enabled.append("request", br#"{"type":"response.create"}"#, 1);
    assert!(String::from_utf8(enabled.body())
        .unwrap()
        .contains("response.create"));
    assert_eq!(
        websocket_payload_event_type(br#"{"type":"response.done"}"#),
        "response.done"
    );
}

#[test]
fn tool_cache_is_bounded_per_session_and_repairs_known_output() {
    let cache = WebsocketToolOutputCache::new(2);
    cache.record(
        "s",
        "a",
        serde_json::json!({"type":"function_call","call_id":"a"}),
    );
    cache.record(
        "s",
        "b",
        serde_json::json!({"type":"function_call","call_id":"b"}),
    );
    cache.record(
        "s",
        "c",
        serde_json::json!({"type":"function_call","call_id":"c"}),
    );
    assert!(cache.get("s", "a").is_none());
    let repaired = repair_responses_websocket_tool_calls(
        &cache,
        "s",
        br#"{"output":[{"type":"function_call_output","call_id":"c","output":"ok"}]}"#,
    );
    assert!(String::from_utf8_lossy(&repaired).contains("function_call"));
    cache.delete_session("s");
    assert!(cache.get("s", "c").is_none());
}
