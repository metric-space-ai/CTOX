// ref: internal/runtime/executor/claude_executor_thinking_signature_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::normalize_claude_sampling_for_upstream;

#[test]
fn request_normalization_preserves_opaque_thinking_signatures() {
    let body = br#"{"temperature":1,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"x","signature":"opaque-signature"}]}]}"#;
    let output = normalize_claude_sampling_for_upstream(body);
    let root: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(
        root.pointer("/messages/0/content/0/signature")
            .and_then(serde_json::Value::as_str),
        Some("opaque-signature")
    );
}
