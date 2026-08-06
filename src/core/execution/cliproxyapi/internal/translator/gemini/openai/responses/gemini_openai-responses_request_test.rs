// ref: internal/translator/gemini/openai/responses/gemini_openai-responses_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};

use super::convert_openai_responses_request_to_gemini;

fn convert(input: &Value) -> Value {
    serde_json::from_slice(&convert_openai_responses_request_to_gemini(
        "gemini-3.6-flash-high",
        &serde_json::to_vec(input).unwrap(),
        false,
    ))
    .unwrap()
}

#[test]
fn reasoning_function_and_reversed_outputs_keep_native_order_and_signature() {
    let output = convert(&json!({"input":[
        {"type":"reasoning","encrypted_content":"native-signature","summary":[{"type":"summary_text","text":"think"}]},
        {"type":"function_call","call_id":"a","name":"first","arguments":"{\"x\":1}"},
        {"type":"function_call","call_id":"b","name":"second","arguments":"{}"},
        {"type":"function_call_output","call_id":"b","output":"two"},
        {"type":"function_call_output","call_id":"a","output":"one"}
    ]}));
    let model = output["contents"][0]["parts"].as_array().unwrap();
    assert_eq!(model[0]["thought"], true);
    assert_eq!(
        model[1]["thoughtSignature"],
        "skip_thought_signature_validator"
    );
    assert_eq!(model[1]["functionCall"]["id"], "a");
    assert_eq!(model[2]["functionCall"]["id"], "b");
    let user = output["contents"][1]["parts"].as_array().unwrap();
    assert_eq!(user[0]["functionResponse"]["id"], "a");
    assert_eq!(user[1]["functionResponse"]["id"], "b");
}

#[test]
fn root_placeholders_match_pinned_go_while_nested_metadata_is_cleaned() {
    let output = convert(&json!({
        "input":"clean",
        "tools":[
            {"type":"function","name":"placeholder","parameters":{"type":"object","properties":{"_":{"type":"boolean"}},"required":["_"]}},
            {"type":"function","name":"nested","parameters":{"type":"object","properties":{"child":{"type":"object","title":"Child","properties":{"reason":{"type":"string","description":"Brief explanation of why you are calling this tool"}},"required":["reason"]}}}}
        ]
    }));
    let declarations = output["tools"][0]["functionDeclarations"]
        .as_array()
        .unwrap();
    assert_eq!(
        declarations[0]["parametersJsonSchema"]["required"],
        json!(["_"])
    );
    assert_eq!(
        declarations[0]["parametersJsonSchema"]["properties"]["_"]["type"],
        "boolean"
    );
    let child = &declarations[1]["parametersJsonSchema"]["properties"]["child"];
    assert!(child.get("title").is_none());
    assert_eq!(child["properties"], json!({}));
    assert!(child.get("required").is_none());
}

#[test]
fn system_roles_structured_output_and_trailing_prefill_follow_upstream() {
    let output = convert(&json!({
        "instructions":"system",
        "input":[
            {"role":"developer","content":"policy"},
            {"role":"user","content":[{"type":"input_text","text":"question"}]},
            {"role":"assistant","content":[{"type":"output_text","text":"prefill"}]}
        ],
        "text":{"format":{"type":"json_schema","schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer","stale"]}}}
    }));
    assert_eq!(output["systemInstruction"]["parts"][0]["text"], "system");
    assert_eq!(output["systemInstruction"]["parts"][1]["text"], "policy");
    assert_eq!(output["contents"].as_array().unwrap().len(), 1);
    assert_eq!(
        output["generationConfig"]["responseMimeType"],
        "application/json"
    );
    assert_eq!(
        output["generationConfig"]["responseJsonSchema"]["required"],
        json!(["answer", "stale"])
    );
}
