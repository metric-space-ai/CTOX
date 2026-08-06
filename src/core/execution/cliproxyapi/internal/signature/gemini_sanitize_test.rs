// ref: internal/signature/gemini_sanitize_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{sanitize_gemini_request_thought_signatures, GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR};

#[test]
fn parallel_synthetic_history_adds_bypass_only_to_first_call() {
    let output = sanitize_gemini_request_thought_signatures(
        br#"{"contents":[{"role":"model","parts":[{"functionCall":{"name":"first","args":{}}},{"functionCall":{"name":"second","args":{}}}]}]}"#,
    );
    let output: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(
        output["contents"][0]["parts"][0]["thoughtSignature"],
        GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR
    );
    assert!(output["contents"][0]["parts"][1]
        .get("thoughtSignature")
        .is_none());
}

#[test]
fn unsigned_thought_is_a_byte_identical_noop() {
    let input = br#" {"contents":[{"role":"model","parts":[{"text":"hidden","thought":true}]}]} "#;
    assert_eq!(sanitize_gemini_request_thought_signatures(input), input);
}

#[test]
fn function_response_cannot_replay_any_signature_location() {
    let output = sanitize_gemini_request_thought_signatures(
        br#"{"contents":[{"role":"user","parts":[{"functionResponse":{"name":"f","thought_signature":"nested"},"thoughtSignature":"top","extra_content":{"google":{"thought_signature":"carrier"}}}]}]}"#,
    );
    let output: Value = serde_json::from_slice(&output).unwrap();
    let part = &output["contents"][0]["parts"][0];
    assert!(part.get("thoughtSignature").is_none());
    assert!(part
        .pointer("/functionResponse/thought_signature")
        .is_none());
    assert!(part
        .pointer("/extra_content/google/thought_signature")
        .is_none());
}
