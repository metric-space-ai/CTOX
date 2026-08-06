// ref: internal/translator/claude/openai/chat-completions/noop_optimization_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::convert_claude_response_to_openai_chat_non_stream;
use serde_json::Value;

#[test]
fn maps_all_upstream_finish_reason_cases() {
    for (reason, expected) in [
        ("", "stop"),
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
    ] {
        let raw = format!(
            "data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"{reason}\"}}}}"
        );
        let value: Value = serde_json::from_slice(
            &convert_claude_response_to_openai_chat_non_stream(b"", b"", raw.as_bytes()),
        )
        .unwrap();
        assert_eq!(value["choices"][0]["finish_reason"], expected);
    }
}
