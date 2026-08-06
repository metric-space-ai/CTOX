// ref: internal/translator/gemini/gemini/gemini_gemini_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_gemini_request_to_gemini;

fn convert(input: &[u8]) -> Value {
    serde_json::from_slice(&convert_gemini_request_to_gemini("gemini-3", input, false)).unwrap()
}

#[test]
fn roles_default_and_alternate_from_previous_valid_role() {
    let output = convert(br#"{"contents":[{"parts":[]},{"role":"invalid","parts":[]},{"role":"user","parts":[]},{"parts":[]}]}"#);
    assert_eq!(output["contents"][0]["role"], "user");
    assert_eq!(output["contents"][1]["role"], "model");
    assert_eq!(output["contents"][2]["role"], "user");
    assert_eq!(output["contents"][3]["role"], "model");
}

#[test]
fn legacy_tool_and_response_schema_keys_are_rewritten() {
    let output = convert(br#"{"contents":[],"tools":[{"functionDeclarations":[{"name":"f","parameters":{"type":"object"}}]}],"generationConfig":{"responseSchema":{"type":"string"}}}"#);
    assert!(output["tools"][0].get("functionDeclarations").is_none());
    assert_eq!(
        output["tools"][0]["function_declarations"][0]["parametersJsonSchema"]["type"],
        "object"
    );
    assert_eq!(
        output["generationConfig"]["responseJsonSchema"]["type"],
        "string"
    );
}

#[test]
fn backfill_is_ordered_and_limited_to_available_calls() {
    let output = convert(br#"{"contents":[{"role":"model","parts":[{"functionCall":{"name":"a"}},{"functionCall":{"name":"b"}}]},{"role":"user","parts":[{"functionResponse":{"name":""}},{"functionResponse":{"name":" "}},{"functionResponse":{"name":""}}]}]}"#);
    assert_eq!(
        output["contents"][1]["parts"][0]["functionResponse"]["name"],
        "a"
    );
    assert_eq!(
        output["contents"][1]["parts"][1]["functionResponse"]["name"],
        "b"
    );
    assert_eq!(
        output["contents"][1]["parts"][2]["functionResponse"]["name"],
        ""
    );
}
