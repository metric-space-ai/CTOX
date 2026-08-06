// ref: internal/translator/codex/claude/codex_claude_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::sdk::translator::TranslationContext;

use super::{
    claude_token_count, convert_codex_response_to_claude_non_stream,
    convert_codex_response_to_claude_stream, CodexToClaudeStreamState,
};

#[test]
fn aggregate_maps_text_thinking_tools_usage_and_web_search() {
    let output: Value = serde_json::from_slice(&convert_codex_response_to_claude_non_stream(
        &TranslationContext::default(),
        "gpt-5",
        b"{}",
        b"{}",
        br#"{"id":"resp","model":"gpt-5","output":[{"type":"reasoning","encrypted_content":"sig","summary":[{"type":"summary_text","text":"why"}]},{"type":"message","content":[{"type":"output_text","text":"answer"}]},{"type":"function_call","call_id":"c1","name":"Read","arguments":"{}"},{"type":"web_search_call","id":"ws1","action":{"query":"rust"},"results":[{"url":"https://example.com","title":"Example"}]}],"usage":{"input_tokens":3,"output_tokens":4,"input_tokens_details":{"cached_tokens":1}}}"#,
    ))
    .unwrap();
    assert_eq!(output["type"], "message");
    assert_eq!(output["usage"]["input_tokens"], 3);
    assert!(output["content"]
        .as_array()
        .unwrap()
        .iter()
        .any(|block| block["type"] == "tool_use"));
    assert!(output["content"]
        .as_array()
        .unwrap()
        .iter()
        .any(|block| block["type"] == "server_tool_use"));
}

#[test]
fn stream_maps_cyber_policy_error_and_web_search() {
    let context = TranslationContext::default();
    let mut state = CodexToClaudeStreamState::with_identity("msg_test");
    let error = convert_codex_response_to_claude_stream(
        &context,
        "gpt-5",
        b"{}",
        b"{}",
        br#"data: {"type":"error","error":{"type":"invalid_request","code":"cyber_policy","message":"flagged"}}"#,
        &mut state,
    );
    let error = String::from_utf8(error.concat()).unwrap();
    assert!(error.contains("invalid_request_error"));

    let web = convert_codex_response_to_claude_stream(
        &context,
        "gpt-5",
        b"{}",
        b"{}",
        br#"data: {"type":"response.output_item.done","item":{"type":"web_search_call","id":"ws1","action":{"query":"rust"},"results":[{"url":"https://example.com"}]}}"#,
        &mut state,
    );
    let web = String::from_utf8(web.concat()).unwrap();
    assert!(web.contains("server_tool_use"));
    assert!(web.contains("web_search_tool_result"));
}

#[test]
fn token_count_uses_claude_shape() {
    assert_eq!(claude_token_count(7), br#"{"input_tokens":7}"#);
}
