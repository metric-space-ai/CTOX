// ref: internal/translator/codex/openai/chat-completions/codex_openai_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_openai_chat_request_to_codex;

#[test]
fn tool_call_and_output_are_adjacent_without_empty_assistant_message() {
    let output: Value = serde_json::from_slice(&convert_openai_chat_request_to_codex(
        "gpt-5",
        br#"{"messages":[{"role":"system","content":"rules"},{"role":"user","content":"weather"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]},{"role":"tool","tool_call_id":"c1","content":"sunny"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}"#,
        true,
    ))
    .unwrap();
    let input = output["input"].as_array().unwrap();
    assert_eq!(input.len(), 4);
    assert_eq!(input[0]["role"], "developer");
    assert_eq!(input[2]["type"], "function_call");
    assert_eq!(input[2]["arguments"], "{\"q\":\"x\"}");
    assert_eq!(input[3]["type"], "function_call_output");
    assert_eq!(input[3]["call_id"], "c1");
}

#[test]
fn multimodal_user_content_and_text_format_are_preserved() {
    let output: Value = serde_json::from_slice(&convert_openai_chat_request_to_codex(
        "gpt-5",
        br#"{"messages":[{"role":"user","content":[{"type":"text","text":"read"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},{"type":"file","file":{"file_data":"SGVsbG8=","filename":"a.txt"}}]}],"response_format":{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object"}}}}"#,
        false,
    ))
    .unwrap();
    assert_eq!(output["input"][0]["content"][1]["type"], "input_image");
    assert_eq!(output["input"][0]["content"][2]["type"], "input_file");
    assert_eq!(output["text"]["format"]["type"], "json_schema");
}
