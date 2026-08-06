// ref: internal/translator/gemini/openai/responses/signature_carrier_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};

use super::convert_openai_responses_request_to_gemini;

const NATIVE: &str = "EjQKMgEMOdbHO0Gd+c9Mxk4ELwPGbpCEcp2mFfYYLix2UVtBH3fL8GECc4+JITVnHF4qZDsA";
const WRAPPED_NATIVE: &str = "cpa-gemini-responses-carrier-v1:next:function:RWpRS01nRU1PZGJITzBHZCtjOU14azRFTHdQR2JwQ0VjcDJtRmZZWUxpeDJVVnRCSDNmTDhHRUNjNCtKSVRWbkhGNHFaRHNB";

fn convert(input: Value) -> Value {
    serde_json::from_slice(&convert_openai_responses_request_to_gemini(
        "gemini-alias",
        &serde_json::to_vec(&input).unwrap(),
        false,
    ))
    .unwrap()
}

#[test]
fn marked_carrier_unwraps_only_for_matching_adjacent_target() {
    let output = convert(json!({"input":[
        {"type":"reasoning","encrypted_content":WRAPPED_NATIVE,"summary":[]},
        {"type":"function_call","call_id":"a","name":"run","arguments":"{}"}
    ]}));
    assert_eq!(
        output["contents"][0]["parts"][0]["thoughtSignature"],
        NATIVE
    );
    assert_eq!(output["contents"][0]["parts"][0]["functionCall"]["id"], "a");
}

#[test]
fn malformed_nested_and_spoofed_carriers_never_reach_gemini() {
    for encrypted in [
        "cpa-gemini-responses-carrier-v1:next:text:not-base64!",
        "cpa-gemini-responses-carrier-v1:next:text:Y3BhLWdlbWluaS1yZXNwb25zZXMtY2Fycmllci12MTpuZXh0OnRleHQ6YQ",
    ] {
        let output = convert(json!({"input":[
            {"type":"reasoning","encrypted_content":encrypted,"summary":[],"_cpa_reasoning_direction":"next","_cpa_reasoning_target":"text"},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}
        ]}));
        let encoded = output.to_string();
        assert!(!encoded.contains("_cpa_reasoning_"));
        assert!(!encoded.contains("not-base64"));
    }
}
