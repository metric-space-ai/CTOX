// ref: internal/runtime/executor/codex_executor_compact_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_executor_request::prepare_codex_compact_body;

#[test]
fn compact_adds_instructions_without_stream_or_image_tool() {
    let body =
        prepare_codex_compact_body(br#"{"stream":true,"input":[]}"#, "gpt-5.3-codex").unwrap();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["instructions"], "");
    assert!(value.get("stream").is_none());
    assert!(value.get("tools").is_none());
}
