// ref: internal/translator/gemini/claude/gemini_claude_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{
    convert_gemini_response_to_claude_non_stream, convert_gemini_response_to_claude_stream,
    GeminiToClaudeStreamState,
};

#[test]
fn nonstream_maps_thinking_text_tools_and_usage() {
    let output: Value = serde_json::from_slice(&convert_gemini_response_to_claude_non_stream(
        br#"{"tools":[{"name":"read file","input_schema":{"type":"object"}}]}"#,
        br#"{"model":"gemini-3"}"#,
        br#"{"responseId":"r1","modelVersion":"gemini-3","usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"thoughtsTokenCount":1},"candidates":[{"finishReason":"STOP","content":{"parts":[{"thought":true,"text":"why"},{"text":"answer"},{"functionCall":{"name":"read_file","args":{}}}]}}]}"#,
    ))
    .unwrap();
    assert_eq!(output["id"], "r1");
    assert_eq!(output["content"][0]["type"], "thinking");
    assert_eq!(output["content"][1]["text"], "answer");
    assert_eq!(output["content"][2]["type"], "tool_use");
    assert_eq!(output["stop_reason"], "tool_use");
    assert_eq!(output["usage"]["output_tokens"], 4);
}

#[test]
fn signature_only_stream_part_does_not_open_empty_text_block() {
    let mut state = GeminiToClaudeStreamState::default();
    let events = convert_gemini_response_to_claude_stream(
        b"{}",
        br#"{"model":"gemini-3"}"#,
        br#"{"responseId":"r","candidates":[{"content":{"parts":[{"thought":true,"text":"thinking"},{"thoughtSignature":"signature-only"}]}}]}"#,
        &mut state,
    );
    let joined = String::from_utf8(events.concat()).unwrap();
    assert!(joined.contains("thinking_delta"));
    assert!(!joined.contains(r#"\"type\":\"text\",\"text\":\"\""#));
}
