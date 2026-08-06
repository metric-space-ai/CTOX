// ref: internal/runtime/executor/executor_payload_optimization_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};

use super::codex_openai_images::{
    build_codex_images_api_response, prepare_codex_direct_image_request,
    prepare_codex_openai_image_request, CodexImageAction, CodexImageResponseFormat,
    CodexImageResult, CODEX_IMAGE_EDIT_PATH,
};
use super::kimi_executor::normalize_kimi_tool_message_links;

// Upstream's ensureColonSpacedJSON case belongs to the still-inactive
// aistudio_executor.go mirror. It is deliberately not manufactured as an
// orphan Rust helper here. The semantic payload cases for the active Kimi and
// Codex implementations are exercised below. Go allocation benchmarks are
// excluded from the unit-test graph; Rust performance gates belong in benches/.

#[test]
fn canonical_kimi_history_is_byte_identical() {
    let input = br#"{"messages":[{"role":"assistant","reasoning_content":"checking","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"ok"}]}"#;
    assert_eq!(normalize_kimi_tool_message_links(input).unwrap(), input);
}

#[test]
fn kimi_repairs_links_without_rounding_large_arguments() {
    let input = br#"{"messages":[{"role":"assistant","content":"lookup","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":{"id":9007199254740993}}}]},{"role":"tool","call_id":"call_1","content":"ok"}]}"#;
    let value: Value =
        serde_json::from_slice(&normalize_kimi_tool_message_links(input).unwrap()).unwrap();
    assert_eq!(
        value["messages"][0]["tool_calls"][0]["function"]["arguments"]["id"].as_u64(),
        Some(9_007_199_254_740_993)
    );
    assert_eq!(value["messages"][1]["tool_call_id"], "call_1");
    assert_eq!(value["messages"][0]["reasoning_content"], "lookup");
}

#[test]
fn multipart_edit_appends_uploaded_image_to_existing_images() {
    let boundary = "ctox-boundary";
    let body = concat!(
        "--ctox-boundary\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n",
        "--ctox-boundary\r\nContent-Disposition: form-data; name=\"images\"\r\n\r\nexisting-1\r\n",
        "--ctox-boundary\r\nContent-Disposition: form-data; name=\"images\"\r\n\r\nexisting-2\r\n",
        "--ctox-boundary\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"source.png\"\r\n",
        "Content-Type: application/octet-stream\r\n\r\npng-data\r\n",
        "--ctox-boundary--\r\n"
    );
    let prepared = prepare_codex_direct_image_request(
        body.as_bytes(),
        "gpt-image-1.5",
        CODEX_IMAGE_EDIT_PATH,
        Some(&format!("multipart/form-data; boundary={boundary}")),
        false,
    )
    .unwrap();
    let value: Value = serde_json::from_slice(&prepared.body).unwrap();
    assert_eq!(value["images"][0], "existing-1");
    assert_eq!(value["images"][1], "existing-2");
    assert!(value["images"][2]["image_url"]
        .as_str()
        .unwrap()
        .starts_with("data:application/octet-stream;base64,"));
}

#[test]
fn codex_image_builders_preserve_prompt_images_tool_and_response() {
    let request = prepare_codex_openai_image_request(
        br#"{"prompt":"draw \"this\"","images":["data:image/png;base64,AA==","","data:image/jpeg;base64,BB=="]}"#,
        "gpt-image-2",
        CodexImageAction::Edit,
    )
    .unwrap();
    let request: Value = serde_json::from_slice(&request.responses_body).unwrap();
    assert_eq!(request["input"][0]["content"][0]["text"], "draw \"this\"");
    assert_eq!(request["input"][0]["content"].as_array().unwrap().len(), 3);
    assert_eq!(request["tools"][0]["model"], "gpt-image-2");

    let result = CodexImageResult {
        base64_data: "AA==".into(),
        output_format: "png".into(),
        revised_prompt: Some("revised \"prompt\"".into()),
        quality: Some("high".into()),
        size: Some("1024x1024".into()),
    };
    let response = build_codex_images_api_response(
        &[result],
        123,
        CodexImageResponseFormat::Base64Json,
        Some(json!({"images": 1})),
    )
    .unwrap();
    let response: Value = serde_json::from_slice(&response).unwrap();
    assert_eq!(response["data"][0]["b64_json"], "AA==");
    assert_eq!(response["data"][0]["revised_prompt"], "revised \"prompt\"");
    assert_eq!(response["usage"]["images"], 1);
}
