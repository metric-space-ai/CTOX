// ref: internal/translator/antigravity/gemini/antigravity_gemini_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::convert_gemini_request_to_antigravity;
use serde_json::Value;

fn convert(input: &[u8]) -> Value {
    serde_json::from_slice(&convert_gemini_request_to_antigravity(
        "gemini-3", input, false,
    ))
    .unwrap()
}

#[test]
fn wraps_model_system_roles_and_safety() {
    let value=convert(br#"{"model":"old","system_instruction":{"parts":[{"text":"rules"}]},"contents":[{"parts":[]},{"role":"bad","parts":[]}]}"#);
    assert_eq!(value["model"], "gemini-3");
    assert!(value["request"].get("model").is_none());
    assert!(value["request"].get("system_instruction").is_none());
    assert!(value["request"].get("systemInstruction").is_some());
    assert_eq!(value["request"]["contents"][0]["role"], "user");
    assert_eq!(value["request"]["contents"][1]["role"], "model");
    assert_eq!(
        value["request"]["safetySettings"].as_array().unwrap().len(),
        5
    );
}

#[test]
fn groups_calls_with_ordered_responses_and_backfills_names() {
    let value=convert(br#"{"contents":[{"role":"model","parts":[{"functionCall":{"name":"a"}},{"functionCall":{"name":"b"}}]},{"role":"user","parts":[{"functionResponse":{"name":""}},{"functionResponse":{"name":" "}}]}]}"#);
    assert_eq!(
        value["request"]["contents"][1]["parts"][0]["functionResponse"]["name"],
        "a"
    );
    assert_eq!(
        value["request"]["contents"][1]["parts"][1]["functionResponse"]["name"],
        "b"
    );
}

#[test]
fn deduplicates_colliding_declarations_without_aliasing() {
    let value=convert(br#"{"contents":[],"tools":[{"functionDeclarations":[{"name":"read file","parameters":{}},{"name":"read/file","parameters":{}},{"name":"read file"}]}]}"#);
    let declarations = value["request"]["tools"][0]["functionDeclarations"]
        .as_array()
        .unwrap();
    assert_eq!(declarations.len(), 2);
    assert_ne!(declarations[0]["name"], declarations[1]["name"]);
    assert!(declarations
        .iter()
        .all(|value| value.get("parameters").is_none()));
}
