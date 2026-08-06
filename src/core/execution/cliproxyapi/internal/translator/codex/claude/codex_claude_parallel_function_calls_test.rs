// ref: internal/translator/codex/claude/codex_claude_parallel_function_calls_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::translator::TranslationContext;

use super::{convert_codex_response_to_claude_stream, CodexToClaudeStreamState};

#[test]
fn interleaved_calls_keep_distinct_ids_and_arguments() {
    let mut state = CodexToClaudeStreamState::with_identity("msg_parallel");
    let context = TranslationContext::default();
    let chunks: &[&[u8]] = &[
        br#"data: {"type":"response.created","response":{"id":"resp","model":"gpt-5"}}"#,
        br#"data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"call_a","name":"Read"}}"#,
        br#"data: {"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","call_id":"call_b","name":"Read"}}"#,
        br#"data: {"type":"response.function_call_arguments.delta","output_index":2,"delta":"{\"file_path\":\"b\"}"}"#,
        br#"data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"file_path\":\"a\"}"}"#,
        br#"data: {"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","call_id":"call_b","name":"Read","arguments":"{\"file_path\":\"b\"}"}}"#,
        br#"data: {"type":"response.output_text.delta","delta":"deferred"}"#,
        br#"data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","call_id":"call_a","name":"Read","arguments":"{\"file_path\":\"a\"}"}}"#,
    ];
    let mut bytes = Vec::new();
    for chunk in chunks {
        for event in convert_codex_response_to_claude_stream(
            &context, "gpt-5", b"{}", b"{}", chunk, &mut state,
        ) {
            bytes.extend(event);
        }
    }
    let output = String::from_utf8(bytes).unwrap();
    assert!(output.contains("call_a"));
    assert!(output.contains("call_b"));
    assert!(output.contains("file_path"));
    assert!(output.find("call_a").unwrap() < output.find("call_b").unwrap());
    assert!(output.find("call_b").unwrap() < output.find("deferred").unwrap());
}
