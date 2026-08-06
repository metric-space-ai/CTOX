// ref: internal/translator/gemini/openai/chat-completions/gemini_openai_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{
    convert_gemini_response_to_openai_chat_non_stream,
    convert_gemini_response_to_openai_chat_stream, GeminiToChatStreamState,
};

#[test]
fn includes_zero_completion_tokens_when_missing() {
    let raw = br#"{"usageMetadata":{"promptTokenCount":16,"thoughtsTokenCount":42,"totalTokenCount":58}}"#;
    let output: Value = serde_json::from_slice(&convert_gemini_response_to_openai_chat_non_stream(
        b"", b"", raw,
    ))
    .unwrap();
    assert_eq!(output["usage"]["completion_tokens"], 0);

    let chunks = convert_gemini_response_to_openai_chat_stream(
        "model",
        b"",
        b"",
        raw,
        &mut GeminiToChatStreamState::default(),
    );
    let output: Value = serde_json::from_slice(&chunks[0]).unwrap();
    assert_eq!(output["usage"]["completion_tokens"], 0);
}

#[test]
fn finish_reason_appears_only_on_final_usage_chunk() {
    let mut state = GeminiToChatStreamState::with_call_id_authority(123, 0);
    let first = convert_gemini_response_to_openai_chat_stream(
        "model",
        b"",
        b"",
        br#"{"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"list_dir","args":{}}}]}}],"usageMetadata":{"trafficType":"ON_DEMAND"}}"#,
        &mut state,
    );
    let first: Value = serde_json::from_slice(&first[0]).unwrap();
    assert!(first["choices"][0]["finish_reason"].is_null());
    assert_eq!(
        first["choices"][0]["delta"]["tool_calls"][0]["id"],
        "list_dir-123-1"
    );

    let final_chunk = convert_gemini_response_to_openai_chat_stream(
        "model",
        b"",
        b"",
        br#"{"candidates":[{"index":0,"content":{"parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10}}"#,
        &mut state,
    );
    let final_chunk: Value = serde_json::from_slice(&final_chunk[0]).unwrap();
    assert_eq!(final_chunk["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(final_chunk["choices"][0]["native_finish_reason"], "stop");
}
