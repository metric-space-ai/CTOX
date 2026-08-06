// ref: internal/translator/claude/gemini/claude_gemini_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::convert_gemini_request_to_claude;
use serde_json::Value;

fn convert(raw: &str) -> Value {
    serde_json::from_slice(&convert_gemini_request_to_claude(
        "claude-sonnet-4",
        raw.as_bytes(),
        false,
    ))
    .unwrap()
}

#[test]
fn invalid_json_is_a_byte_identical_noop() {
    let raw = br#" {not-json}\n"#;
    assert_eq!(
        convert_gemini_request_to_claude("claude-sonnet-4", raw, false),
        raw
    );
}

#[test]
fn preserves_custom_tool_ids_and_drops_temperature() {
    for key in ["id", "call_id"] {
        let raw = format!(
            r#"{{"generationConfig":{{"temperature":0.2,"topP":0.8}},"contents":[{{"role":"model","parts":[{{"functionCall":{{"name":"lookup","{key}":"call_gateway","args":{{"q":"x"}}}}}}]}},{{"role":"user","parts":[{{"functionResponse":{{"name":"lookup","{key}":"call_gateway","response":{{"ok":true}}}}}}]}}]}}"#
        );
        let out = convert(&raw);
        assert_eq!(
            out.pointer("/messages/0/content/0/id").unwrap(),
            "call_gateway"
        );
        assert_eq!(
            out.pointer("/messages/1/content/0/tool_use_id").unwrap(),
            "call_gateway"
        );
        assert!(out.get("temperature").is_none());
        assert_eq!(out["top_p"], 0.8);
    }
}

#[test]
fn camel_inline_data_and_non_image_mime_splitting_match_upstream() {
    let out = convert(
        r#"{"contents":[{"role":"user","parts":[{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}},{"inlineData":{"mimeType":"audio/wav","data":"UklGRg=="}},{"inlineData":{"mimeType":"video/mp4","data":"AAAA"}},{"inlineData":{"mimeType":"application/pdf","data":"JVBERi0="}}]}]}"#,
    );
    assert_eq!(out.pointer("/messages/0/content/0/type").unwrap(), "image");
    assert_eq!(
        out.pointer("/messages/0/content/0/source/media_type")
            .unwrap(),
        "image/png"
    );
    assert_eq!(out.pointer("/messages/0/content/1/type").unwrap(), "text");
    assert_eq!(out.pointer("/messages/0/content/2/type").unwrap(), "text");
    assert_eq!(
        out.pointer("/messages/0/content/3/type").unwrap(),
        "document"
    );
}
