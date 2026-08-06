// ref: internal/translator/antigravity/openai/chat-completions/antigravity_openai_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{
    convert_antigravity_response_to_openai_chat_non_stream,
    convert_antigravity_response_to_openai_chat_stream, AntigravityToChatStreamState,
};
use crate::internal::util::sanitized_function_name_map;

fn chunk(state: &mut AntigravityToChatStreamState, raw: &[u8]) -> Value {
    let output = convert_antigravity_response_to_openai_chat_stream("model", &[], &[], raw, state);
    serde_json::from_slice(&output[0]).unwrap()
}

#[test]
fn finish_reason_preserves_tool_call_priority() {
    let mut state = AntigravityToChatStreamState::default();
    let first = chunk(
        &mut state,
        br#"{"response":{"candidates":[{"content":{"parts":[{"functionCall":{"name":"list_files","args":{"path":"."}}}]}}]}}"#,
    );
    assert!(first
        .pointer("/choices/0/finish_reason")
        .is_none_or(Value::is_null));
    let final_chunk = chunk(
        &mut state,
        br#"{"response":{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30}}}"#,
    );
    assert_eq!(
        final_chunk.pointer("/choices/0/finish_reason"),
        Some(&Value::String("tool_calls".into()))
    );
    assert_eq!(
        final_chunk.pointer("/choices/0/native_finish_reason"),
        Some(&Value::String("stop".into()))
    );

    let mut state = AntigravityToChatStreamState::default();
    let _ = chunk(&mut state, br#"{"response":{"candidates":[{"content":{"parts":[{"functionCall":{"name":"test","args":{}}}]}}]}}"#);
    let final_chunk = chunk(&mut state, br#"{"response":{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"totalTokenCount":110}}}"#);
    assert_eq!(
        final_chunk.pointer("/choices/0/finish_reason"),
        Some(&Value::String("tool_calls".into()))
    );
}

#[test]
fn finish_reason_maps_stop_and_max_tokens_but_not_intermediate_chunks() {
    for (native, expected) in [("STOP", "stop"), ("MAX_TOKENS", "max_tokens")] {
        let mut state = AntigravityToChatStreamState::default();
        let intermediate = chunk(
            &mut state,
            br#"{"response":{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}}"#,
        );
        assert!(intermediate
            .pointer("/choices/0/finish_reason")
            .is_none_or(Value::is_null));
        let final_raw = format!(
            r#"{{"response":{{"candidates":[{{"finishReason":"{native}"}}],"usageMetadata":{{"totalTokenCount":15}}}}}}"#
        );
        let final_chunk = chunk(&mut state, final_raw.as_bytes());
        assert_eq!(
            final_chunk.pointer("/choices/0/finish_reason"),
            Some(&Value::String(expected.into()))
        );
    }
}

#[test]
fn stream_usage_includes_zero_completion_tokens_when_missing() {
    let mut state = AntigravityToChatStreamState::default();
    let output = chunk(
        &mut state,
        br#"{"response":{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"totalTokenCount":10}}}"#,
    );
    assert_eq!(
        output
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_i64),
        Some(0)
    );
}

#[test]
fn non_stream_restores_disambiguated_name_and_reasoning_content() {
    let first = "mcp__plugin_cloudflare_cloudflare-builds__workers_builds_get_build";
    let second = "mcp__plugin_cloudflare_cloudflare-builds__workers_builds_get_build_logs";
    let original = format!(
        r#"{{"tools":[{{"type":"function","function":{{"name":"{first}"}}}},{{"type":"function","function":{{"name":"{second}"}}}}]}}"#
    );
    let mapped = sanitized_function_name_map(original.as_bytes())[second].clone();
    let raw = format!(
        r#"{{"response":{{"candidates":[{{"content":{{"parts":[{{"text":"thinking","thought":true}},{{"text":"answer"}},{{"functionCall":{{"name":"{mapped}","args":{{}}}}}}]}}}}]}}}}"#
    );
    let output: Value =
        serde_json::from_slice(&convert_antigravity_response_to_openai_chat_non_stream(
            original.as_bytes(),
            &[],
            raw.as_bytes(),
        ))
        .unwrap();
    assert_eq!(
        output.pointer("/choices/0/message/tool_calls/0/function/name"),
        Some(&Value::String(second.into()))
    );
    assert_eq!(
        output.pointer("/choices/0/message/reasoning_content"),
        Some(&Value::String("thinking".into()))
    );
    assert_eq!(
        output.pointer("/choices/0/message/content"),
        Some(&Value::String("answer".into()))
    );
}
