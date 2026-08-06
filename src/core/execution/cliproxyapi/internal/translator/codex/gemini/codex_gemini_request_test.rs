// ref: internal/translator/codex/gemini/codex_gemini_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::codex_gemini_request::convert_gemini_request_to_codex;

fn convert(input: &str) -> Value {
    serde_json::from_slice(&convert_gemini_request_to_codex(
        "gpt-5.1-codex",
        input.as_bytes(),
        false,
    ))
    .unwrap()
}

#[test]
fn preserves_gateway_call_ids_and_fifo_pairing() {
    for field in ["id", "call_id"] {
        let raw = format!(
            r#"{{"contents":[{{"role":"model","parts":[{{"functionCall":{{"name":"lookup","{field}":"call_gateway","args":{{"query":"status"}}}}}}]}},{{"role":"user","parts":[{{"functionResponse":{{"name":"lookup","{field}":"call_gateway","response":{{"result":"ok"}}}}}}]}}]}}"#
        );
        let output = convert(&raw);
        assert_eq!(output["input"][0]["call_id"], "call_gateway");
        assert_eq!(output["input"][1]["call_id"], "call_gateway");
    }
}

#[test]
fn accepts_image_inline_data() {
    let output = convert(
        r#"{"contents":[{"role":"user","parts":[{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}]}]}"#,
    );
    assert_eq!(output["input"][0]["content"][0]["type"], "input_image");
    assert_eq!(
        output["input"][0]["content"][0]["image_url"],
        "data:image/png;base64,aGVsbG8="
    );
}

#[test]
fn splits_non_image_inline_data_by_mime() {
    let output = convert(
        r#"{"contents":[{"role":"user","parts":[{"inlineData":{"mimeType":"audio/wav","data":"UklGRg=="}},{"inlineData":{"mimeType":"video/mp4","data":"AAAAIGZ0eXA="}},{"inlineData":{"mimeType":"application/pdf","data":"JVBERi0="}}]}]}"#,
    );
    assert_eq!(output["input"][0]["content"][0]["type"], "input_audio");
    assert_eq!(output["input"][1]["content"][0]["type"], "input_file");
    assert_eq!(output["input"][2]["content"][0]["type"], "input_file");
}

#[test]
fn system_tools_choice_and_thinking_are_normalized() {
    let output = convert(
        r#"{"systemInstruction":{"parts":[{"text":"policy"}]},"service_tier":"fast","generationConfig":{"thinkingConfig":{"thinkingBudget":1024}},"tools":[{"functionDeclarations":[{"name":"lookup","parameters":{"$schema":"x","type":"OBJECT"}}]}],"toolConfig":{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["lookup"]}}}"#,
    );
    assert_eq!(output["input"][0]["role"], "developer");
    assert_eq!(output["service_tier"], "priority");
    assert_eq!(output["reasoning"]["effort"], "low");
    assert_eq!(output["tool_choice"]["name"], "lookup");
    assert_eq!(output["tools"][0]["parameters"]["type"], "object");
    assert_eq!(
        output["tools"][0]["parameters"]["additionalProperties"],
        false
    );
    assert!(output["tools"][0]["parameters"].get("$schema").is_none());
}
