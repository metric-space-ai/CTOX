// ref: internal/translator/gemini/interactions/interactions_gemini_file_data_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_interactions_request_to_gemini;

#[test]
fn normalizes_openai_file_data_url() {
    let output: Value = serde_json::from_slice(&convert_interactions_request_to_gemini(
        "gemini-3.5-flash",
        br#"{"model":"gemini-3.5-flash","input":[{"type":"user_input","content":[{"type":"file","file":{"filename":"test.pdf","file_data":"data:application/pdf;base64,JVBERi0xLjQK"}}]}]}"#,
        false,
    ))
    .unwrap();
    assert_eq!(
        output["contents"][0]["parts"][0]["inlineData"]["mimeType"],
        "application/pdf"
    );
    assert_eq!(
        output["contents"][0]["parts"][0]["inlineData"]["data"],
        "JVBERi0xLjQK"
    );
}
