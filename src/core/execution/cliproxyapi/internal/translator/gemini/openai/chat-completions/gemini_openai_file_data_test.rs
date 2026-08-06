// ref: internal/translator/gemini/openai/chat-completions/gemini_openai_file_data_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_openai_chat_request_to_gemini;

#[test]
fn normalizes_file_data_url() {
    let input = br#"{"messages":[{"role":"user","content":[{"type":"file","file":{"filename":"test.pdf","file_data":"data:application/pdf;base64,JVBERi0xLjQK"}}]}]}"#;
    let output: Value = serde_json::from_slice(&convert_openai_chat_request_to_gemini(
        "gemini-2.5-pro",
        input,
        false,
    ))
    .unwrap();
    assert_eq!(
        output["contents"][0]["parts"][0]["inlineData"]["mime_type"],
        "application/pdf"
    );
    assert_eq!(
        output["contents"][0]["parts"][0]["inlineData"]["data"],
        "JVBERi0xLjQK"
    );
}
