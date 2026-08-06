// ref: internal/translator/openai/gemini/openai_gemini_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_gemini_request_to_openai;

fn translate(input: &[u8]) -> Value {
    serde_json::from_slice(&convert_gemini_request_to_openai(
        "test-model",
        input,
        false,
    ))
    .expect("translator emits JSON")
}

#[test]
fn function_responses_consume_tool_call_ids_fifo() {
    let output = translate(
        br#"{
        "contents":[
            {"role":"model","parts":[
                {"functionCall":{"name":"read_file","args":{"path":"a.txt"}}},
                {"functionCall":{"name":"grep","args":{"pattern":"needle"}}},
                {"functionCall":{"name":"list_dir","args":{"path":"."}}}
            ]},
            {"role":"function","parts":[
                {"functionResponse":{"name":"read_file","response":{"result":"a"}}},
                {"functionResponse":{"name":"grep","response":{"result":"b"}}},
                {"functionResponse":{"name":"list_dir","response":{"result":"c"}}}
            ]}
        ]
    }"#,
    );
    let ids = output["messages"][0]["tool_calls"]
        .as_array()
        .expect("tool calls")
        .iter()
        .map(|call| call["id"].as_str().expect("call id"))
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), 3);
    assert!(ids
        .iter()
        .all(|id| id.starts_with("call_") && id.len() == 29));
    assert_ne!(ids[0], ids[1]);
    assert_ne!(ids[1], ids[2]);
    assert_eq!(output["messages"][1]["tool_call_id"], ids[0]);
    assert_eq!(output["messages"][2]["tool_call_id"], ids[1]);
    assert_eq!(output["messages"][3]["tool_call_id"], ids[2]);
}

#[test]
fn response_without_prior_call_gets_request_local_fallback_id() {
    let input = br#"{"contents":[{"role":"function","parts":[{"functionResponse":{"name":"read_file","response":{"result":"ok"}}}]}]}"#;
    let first = translate(input);
    let second = translate(input);
    let id = first["messages"][0]["tool_call_id"]
        .as_str()
        .expect("fallback id");
    assert!(id.starts_with("call_"));
    assert_eq!(
        first, second,
        "same request has deterministic local authority"
    );
}

#[test]
fn extra_function_responses_get_distinct_fallback_ids() {
    let output = translate(
        br#"{"contents":[
        {"role":"model","parts":[{"functionCall":{"name":"read_file","args":{"path":"a.txt"}}}]},
        {"role":"function","parts":[
            {"functionResponse":{"name":"read_file","response":{"result":"a"}}},
            {"functionResponse":{"name":"read_file","response":{"result":"extra"}}}
        ]}
    ]}"#,
    );
    let call = output["messages"][0]["tool_calls"][0]["id"]
        .as_str()
        .expect("call id");
    assert_eq!(output["messages"][1]["tool_call_id"], call);
    assert_ne!(output["messages"][2]["tool_call_id"], call);
    assert!(output["messages"][2]["tool_call_id"]
        .as_str()
        .is_some_and(|id| id.starts_with("call_")));
}

#[test]
fn preserves_explicit_id_and_call_id() {
    for (field, expected) in [
        ("id", "call_gateway_id"),
        ("call_id", "call_gateway_call_id"),
    ] {
        let input = format!(
            r#"{{"contents":[
                {{"role":"model","parts":[{{"functionCall":{{"name":"lookup","{field}":"{expected}","args":{{"q":"x"}}}}}}]}},
                {{"role":"function","parts":[{{"functionResponse":{{"name":"lookup","{field}":"{expected}","response":{{"result":"ok"}}}}}}]}}
            ]}}"#
        );
        let output = translate(input.as_bytes());
        assert_eq!(output["messages"][0]["tool_calls"][0]["id"], expected);
        assert_eq!(output["messages"][1]["tool_call_id"], expected);
    }
}

#[test]
fn accepts_snake_case_inline_data() {
    let output = translate(br#"{"contents":[{"role":"user","parts":[{"inline_data":{"mime_type":"image/png","data":"aGVsbG8="}}]}]}"#);
    assert_eq!(
        output["messages"][0]["content"][0]["image_url"]["url"],
        "data:image/png;base64,aGVsbG8="
    );
}

#[test]
fn splits_non_image_inline_data_by_mime() {
    let output = translate(
        br#"{"contents":[{"role":"user","parts":[
        {"inlineData":{"mimeType":"audio/wav","data":"UklGRg=="}},
        {"inlineData":{"mimeType":"video/mp4","data":"AAAAIGZ0eXA="}},
        {"inlineData":{"mimeType":"application/pdf","data":"JVBERi0="}}
    ]}]}"#,
    );
    let parts = output["messages"][0]["content"].as_array().expect("parts");
    assert_eq!(parts[0]["type"], "input_audio");
    assert_eq!(parts[0]["input_audio"]["format"], "wav");
    assert_eq!(parts[1]["type"], "video_url");
    assert_eq!(parts[2]["type"], "file");
    assert_eq!(parts[2]["file"]["filename"], "document.pdf");
}

#[test]
fn maps_model_generation_system_files_tools_and_choice() {
    let raw = br#"{
        "service_tier":"priority",
        "system_instruction":{"parts":[{"text":"policy"},{"file_data":{"file_uri":"gs://doc","mime_type":"text/plain"}}]},
        "contents":[{"role":"user","parts":[{"text":"a"},{"text":"b"}]}],
        "generationConfig":{
            "temperature":0.4,"maxOutputTokens":42,"topP":0.8,"topK":9,
            "stopSequences":["END"],"candidateCount":2,
            "responseModalities":[" TEXT ","image","unknown"],
            "thinkingConfig":{"thinking_budget":9000}
        },
        "tools":[{"functionDeclarations":[{"name":"lookup","description":"d","parametersJsonSchema":{"type":"object"}}]}],
        "toolConfig":{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["lookup"]}}
    }"#;
    let output: Value =
        serde_json::from_slice(&convert_gemini_request_to_openai("actual-model", raw, true))
            .expect("JSON");
    assert_eq!(output["model"], "actual-model");
    assert_eq!(output["stream"], true);
    assert_eq!(output["service_tier"], "priority");
    assert_eq!(output["temperature"], 0.4);
    assert_eq!(output["max_tokens"], 42);
    assert_eq!(output["top_p"], 0.8);
    assert_eq!(output["top_k"], 9);
    assert_eq!(output["stop"], serde_json::json!(["END"]));
    assert_eq!(output["n"], 2);
    assert_eq!(output["modalities"], serde_json::json!(["text", "image"]));
    assert_eq!(output["reasoning_effort"], "high");
    assert_eq!(output["messages"][0]["role"], "system");
    assert_eq!(output["messages"][1]["content"], "ab");
    assert_eq!(output["tools"][0]["function"]["name"], "lookup");
    assert_eq!(
        output["tools"][0]["function"]["parameters"]["type"],
        "object"
    );
    assert_eq!(output["tool_choice"]["function"]["name"], "lookup");
}
