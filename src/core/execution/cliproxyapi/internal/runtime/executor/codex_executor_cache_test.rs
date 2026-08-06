// ref: internal/runtime/executor/codex_executor_cache_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_websockets_request::{
    apply_codex_prompt_cache_headers, codex_session_header_value, CodexWebsocketHeaders,
};

#[test]
fn prompt_cache_key_and_session_header_share_one_identity() {
    let (body, headers) = apply_codex_prompt_cache_headers(
        br#"{"prompt_cache_key":"session-a","input":[]}"#,
        &CodexWebsocketHeaders::new(),
    );
    assert_eq!(codex_session_header_value(&headers), Some("session-a"));
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).unwrap()["prompt_cache_key"],
        "session-a"
    );
}
