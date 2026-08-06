// ref: internal/translator/antigravity/gemini/noop_optimization_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::convert_gemini_request_to_antigravity;

#[test]
fn normalized_request_remains_semantically_stable() {
    let input =
        br#"{"contents":[{"role":"user","parts":[{"text":"hello"}]}],"safetySettings":null}"#;
    let first = convert_gemini_request_to_antigravity("gemini-3", input, false);
    let first_value: serde_json::Value = serde_json::from_slice(&first).unwrap();
    assert_eq!(
        first_value["request"]["contents"][0]["parts"][0]["text"],
        "hello"
    );
    assert!(first_value["request"]["safetySettings"].is_null());
}
