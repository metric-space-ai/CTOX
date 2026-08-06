// ref: internal/translator/antigravity/openai/chat-completions/antigravity_openai_file_data_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_openai_chat_request_to_antigravity;

#[test]
fn normalizes_file_data_url() {
    let output: Value = serde_json::from_slice(&convert_openai_chat_request_to_antigravity(
        "gemini-3.5-flash",
        br#"{"messages":[{"role":"user","content":[{"type":"file","file":{"filename":"test.pdf","file_data":"data:application/pdf;base64,JVBERi0xLjQK"}}]}]}"#,
        false,
    ))
    .unwrap();
    assert_eq!(
        output.pointer("/request/contents/0/parts/0/inlineData/mimeType"),
        Some(&Value::String("application/pdf".into()))
    );
    assert_eq!(
        output.pointer("/request/contents/0/parts/0/inlineData/data"),
        Some(&Value::String("JVBERi0xLjQK".into()))
    );
}
