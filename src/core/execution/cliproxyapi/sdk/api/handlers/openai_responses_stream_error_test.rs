// ref: sdk/api/handlers/openai_responses_stream_error_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::build_openai_responses_stream_error_chunk;

#[test]
fn builds_top_level_responses_stream_error_chunk() {
    let chunk = build_openai_responses_stream_error_chunk(500, "unexpected EOF", 0);
    let payload: Value = serde_json::from_slice(&chunk).unwrap();
    assert_eq!(payload["type"], "error");
    assert_eq!(payload["code"], "internal_server_error");
    assert_eq!(payload["message"], "unexpected EOF");
    assert_eq!(payload["sequence_number"], 0);
    assert!(payload.get("error").is_none());
}

#[test]
fn extracts_nested_http_error_body() {
    let chunk = build_openai_responses_stream_error_chunk(
        500,
        r#"{"error":{"message":"oops","type":"server_error","code":"internal_server_error"}}"#,
        0,
    );
    let payload: Value = serde_json::from_slice(&chunk).unwrap();
    assert_eq!(payload["type"], "error");
    assert_eq!(payload["code"], "internal_server_error");
    assert_eq!(payload["message"], "oops");
}

#[test]
fn extracts_top_level_error_sequence_and_scalar_code() {
    let chunk = build_openai_responses_stream_error_chunk(
        429,
        r#"{"type":"error","message":"slow down","code":429,"sequence_number":7}"#,
        0,
    );
    let payload: Value = serde_json::from_slice(&chunk).unwrap();
    assert_eq!(payload["code"], "429");
    assert_eq!(payload["message"], "slow down");
    assert_eq!(payload["sequence_number"], 7);
}
