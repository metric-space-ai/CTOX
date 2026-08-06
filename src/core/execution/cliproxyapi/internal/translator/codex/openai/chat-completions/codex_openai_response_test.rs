// ref: internal/translator/codex/openai/chat-completions/codex_openai_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{
    convert_codex_response_to_openai_chat_non_stream, convert_codex_response_to_openai_chat_stream,
    CodexToChatStreamState,
};

#[test]
fn aggregate_keeps_assistant_role_text_tools_and_usage() {
    let output: Value = serde_json::from_slice(&convert_codex_response_to_openai_chat_non_stream(
        br#"{"tools":[{"type":"function","function":{"name":"lookup"}}]}"#,
        b"{}",
        br#"{"type":"response.completed","response":{"id":"resp","model":"gpt-5","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]},{"type":"function_call","call_id":"c1","name":"lookup","arguments":"{}"}],"usage":{"input_tokens":1,"output_tokens":2}}}"#,
    ))
    .unwrap();
    assert_eq!(output["choices"][0]["message"]["role"], "assistant");
    assert_eq!(output["choices"][0]["message"]["content"], "hello");
    assert_eq!(output["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(output["usage"]["prompt_tokens"], 1);
}

#[test]
fn stream_maps_text_and_terminal_finish_reason() {
    let mut state = CodexToChatStreamState::default();
    let created = convert_codex_response_to_openai_chat_stream(
        "gpt-5", b"{}", b"{}", br#"data: {"type":"response.created","response":{"id":"resp","model":"gpt-5","created_at":1}}"#, &mut state,
    );
    assert!(created.is_empty());
    let text = convert_codex_response_to_openai_chat_stream(
        "gpt-5",
        b"{}",
        b"{}",
        br#"data: {"type":"response.output_text.delta","delta":"hello"}"#,
        &mut state,
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&text[0]).unwrap()["choices"][0]["delta"]["content"],
        "hello"
    );
    let done = convert_codex_response_to_openai_chat_stream(
        "gpt-5", b"{}", b"{}", br#"data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":2}}}"#, &mut state,
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&done[0]).unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
}
