// ref: internal/translator/openai/interactions/chat-completions/openai_interactions_file_data_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use gjson;
use serde_json::Value;

use super::convert_openai_request_to_interactions;

fn parse(raw: &[u8]) -> Value {
    serde_json::from_slice(raw).unwrap()
}

#[test]
fn normalizes_openai_file_data_url_to_interactions_document() {
    let input = br#"{"model":"gemini-3.5-flash","messages":[{"role":"user","content":[{"type":"file","file":{"filename":"test.pdf","file_data":"data:application/pdf;base64,JVBERi0xLjQK"}}]}]}"#;
    let out = parse(&convert_openai_request_to_interactions(
        "gemini-3.5-flash",
        input,
        false,
    ));
    let document = &out["input"][0]["content"][0];
    assert_eq!(document["mime_type"], "application/pdf");
    assert_eq!(document["data"], "JVBERi0xLjQK");
}

#[test]
fn preserves_raw_file_data_with_explicit_mime_type() {
    let input = br#"{"model":"gemini-3.5-flash","messages":[{"role":"user","content":[{"type":"document","mime_type":"application/pdf","data":"JVBERi0xLjQK"}]}]}"#;
    let out = parse(&convert_openai_request_to_interactions(
        "gemini-3.5-flash",
        input,
        false,
    ));
    let document = &out["input"][0]["content"][0];
    assert_eq!(document["mime_type"], "application/pdf");
    assert_eq!(document["data"], "JVBERi0xLjQK");
}

#[test]
fn detects_payload_via_gjson_path_lookup() {
    // Mirrors the upstream Go assertion that uses `gjson.GetBytes` rather
    // than `serde_json::from_slice`, exercising the public gjson path. This
    // catches regressions where the response is structurally valid but the
    // byte path diverges from the upstream wire format.
    let input = br#"{"model":"gemini-3.5-flash","messages":[{"role":"user","content":[{"type":"file","file":{"filename":"test.pdf","file_data":"data:application/pdf;base64,JVBERi0xLjQK"}}]}]}"#;
    let output = convert_openai_request_to_interactions("gemini-3.5-flash", input, false);
    let document = gjson::get(
        std::str::from_utf8(&output).unwrap_or(""),
        "input.0.content.0",
    );
    assert_eq!(document.get("mime_type").str(), "application/pdf");
    assert_eq!(document.get("data").str(), "JVBERi0xLjQK");
}
