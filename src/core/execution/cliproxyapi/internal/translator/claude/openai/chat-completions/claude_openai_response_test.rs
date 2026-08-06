// ref: internal/translator/claude/openai/chat-completions/claude_openai_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{
    convert_claude_response_to_openai_chat_non_stream,
    convert_claude_response_to_openai_chat_stream, ClaudeToChatStreamState,
};
use serde_json::Value;

fn assert_usage(value: &Value) {
    assert_eq!(value["usage"]["prompt_tokens"], 22_044);
    assert_eq!(value["usage"]["completion_tokens"], 4);
    assert_eq!(value["usage"]["total_tokens"], 22_048);
    assert_eq!(
        value["usage"]["prompt_tokens_details"]["cached_tokens"],
        22_000
    );
    assert_eq!(
        value["usage"]["prompt_tokens_details"]["cached_creation_tokens"],
        31
    );
}

#[test]
fn stream_usage_includes_and_merges_cached_tokens() {
    let mut state = ClaudeToChatStreamState::default();
    convert_claude_response_to_openai_chat_stream(
        "claude-opus-4-6", b"", b"",
        br#"data: {"type":"message_start","message":{"usage":{"input_tokens":13,"output_tokens":1,"cache_read_input_tokens":22000,"cache_creation_input_tokens":31}}}"#,
        &mut state,
    );
    let chunks = convert_claude_response_to_openai_chat_stream(
        "claude-opus-4-6", b"", b"",
        br#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}"#,
        &mut state,
    );
    assert_usage(&serde_json::from_slice(&chunks[0]).unwrap());
}

#[test]
fn non_stream_usage_includes_and_merges_cached_tokens() {
    let raw = br#"data: {"type":"message_start","message":{"id":"msg_123","model":"claude-opus-4-6","usage":{"input_tokens":13,"output_tokens":1,"cache_read_input_tokens":22000,"cache_creation_input_tokens":31}}}
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}
"#;
    let value: Value = serde_json::from_slice(&convert_claude_response_to_openai_chat_non_stream(
        b"", b"", raw,
    ))
    .unwrap();
    assert_usage(&value);
}
