// ref: internal/translator/gemini/openai/responses/noop_optimization_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_openai_responses_request_to_gemini;

#[test]
fn generation_config_is_built_directly_without_intermediate_nulls() {
    let output: Value = serde_json::from_slice(&convert_openai_responses_request_to_gemini(
        "gemini",
        br#"{"input":"hello","max_output_tokens":128,"temperature":0.25,"top_p":0.75,"stop_sequences":["END"]}"#,
        false,
    ))
    .unwrap();
    assert_eq!(output["generationConfig"]["maxOutputTokens"], 128);
    assert_eq!(output["generationConfig"]["temperature"], 0.25);
    assert_eq!(output["generationConfig"]["topP"], 0.75);
    assert_eq!(output["generationConfig"]["stopSequences"][0], "END");
    assert!(!output.to_string().contains(":null"));
}
