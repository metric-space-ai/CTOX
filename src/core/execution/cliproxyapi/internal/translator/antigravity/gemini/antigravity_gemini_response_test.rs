// ref: internal/translator/antigravity/gemini/antigravity_gemini_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{
    convert_antigravity_response_to_gemini, convert_antigravity_response_to_gemini_non_stream,
};
use serde_json::Value;

#[test]
fn non_stream_restores_cpa_usage_metadata() {
    let output = convert_antigravity_response_to_gemini_non_stream(
        b"{}",
        br#"{"response":{"cpaUsageMetadata":{"totalTokenCount":4},"candidates":[]}}"#,
    );
    let output: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(output["usageMetadata"]["totalTokenCount"], 4);
    assert!(output.get("cpaUsageMetadata").is_none());
}

#[test]
fn stream_requires_alt_and_preserves_upstream_nonempty_alt_quirk() {
    assert!(convert_antigravity_response_to_gemini(b"{}", br#"{"response":{}}"#, None).is_empty());
    assert_eq!(
        convert_antigravity_response_to_gemini(b"{}", br#"[]"#, Some("sse")),
        vec![b"[]".to_vec()]
    );
}
