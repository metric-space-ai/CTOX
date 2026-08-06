// ref: internal/translator/gemini/openai/chat-completions/noop_optimization_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{
    convert_gemini_response_to_openai_chat_non_stream,
    convert_gemini_response_to_openai_chat_stream, convert_openai_chat_request_to_gemini,
    GeminiToChatStreamState,
};

#[test]
fn normalizes_non_string_tool_name_and_removes_strict() {
    let output: Value = serde_json::from_slice(&convert_openai_chat_request_to_gemini(
        "gemini-test",
        br#"{"messages":[],"tools":[{"type":"function","function":{"name":true,"strict":true,"parameters":{"type":"object"}}}]}"#,
        false,
    ))
    .unwrap();
    assert_eq!(
        output["tools"][0]["functionDeclarations"][0]["name"],
        "true"
    );
    assert!(output["tools"][0]["functionDeclarations"][0]
        .get("strict")
        .is_none());
}

#[test]
fn responses_keep_assistant_role_for_nonstream_and_mixed_stream_payloads() {
    let nonstream: Value = serde_json::from_slice(
        &convert_gemini_response_to_openai_chat_non_stream(
            b"",
            b"",
            br#"{"candidates":[{"index":0,"content":{"parts":[{"text":"hello"}]},"finishReason":"STOP"}]}"#,
        ),
    )
    .unwrap();
    assert_eq!(nonstream["choices"][0]["message"]["role"], "assistant");

    let chunks = convert_gemini_response_to_openai_chat_stream(
        "",
        b"",
        b"",
        br#"{"candidates":[{"index":0,"content":{"parts":[{"text":"hello"},{"functionCall":{"name":"lookup","args":{}}},{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}]}}]}"#,
        &mut GeminiToChatStreamState::with_call_id_authority(7, 0),
    );
    let stream: Value = serde_json::from_slice(&chunks[0]).unwrap();
    assert_eq!(stream["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(stream["choices"][0]["delta"]["content"], "hello");
    assert!(stream["choices"][0]["delta"]["tool_calls"][0].is_object());
    assert!(stream["choices"][0]["delta"]["images"][0].is_object());
}
