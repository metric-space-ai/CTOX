// ref: internal/translator/antigravity/interactions/interactions_antigravity_file_data_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_interactions_request_to_antigravity;

#[test]
fn normalizes_openai_file_data_url() {
    let input = br#"{"model":"gemini-3.5-flash","input":[{"type":"user_input","content":[{"type":"file","file":{"filename":"test.pdf","file_data":"data:application/pdf;base64,JVBERi0xLjQK"}}]}]}"#;
    let out: Value = serde_json::from_slice(&convert_interactions_request_to_antigravity(
        "gemini-3.5-flash",
        input,
        false,
    ))
    .unwrap();
    assert_eq!(
        out.pointer("/request/contents/0/parts/0/inlineData/mimeType"),
        Some(&Value::String("application/pdf".into()))
    );
    assert_eq!(
        out.pointer("/request/contents/0/parts/0/inlineData/data"),
        Some(&Value::String("JVBERi0xLjQK".into()))
    );
}
