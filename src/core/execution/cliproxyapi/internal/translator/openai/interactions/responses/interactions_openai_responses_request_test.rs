// ref: internal/translator/openai/interactions/responses/interactions_openai_responses_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{
    convert_interactions_request_to_openai_responses,
    convert_openai_responses_request_to_interactions,
};
use serde_json::{json, Value};

fn value(raw: Vec<u8>) -> Value {
    serde_json::from_slice(&raw).unwrap()
}

#[test]
fn converts_responses_request_to_interactions() {
    let out = value(convert_openai_responses_request_to_interactions(
        "gpt-test",
        br#"{
            "model":"gpt-test", "instructions":"be brief", "stream":true,
            "input":[
              {"type":"message","role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":"data:image/png;base64,aGVsbG8="}]},
              {"type":"function_call","name":"lookup","call_id":"call_1","arguments":"{\"q\":\"x\"}"},
              {"type":"function_call_output","call_id":"call_1","output":{"ok":true}}
            ],
            "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],
            "tool_choice":"auto", "reasoning":{"effort":"high","summary":"auto"},
            "response_format":{"type":"json_object"}
        }"#,
        true,
    ));
    assert_eq!(out["input"][0]["type"], "user_input");
    assert_eq!(out["input"][0]["content"][0]["text"], "hi");
    assert_eq!(out["input"][0]["content"][1]["mime_type"], "image/png");
    assert_eq!(out["input"][1]["call_id"], "call_1");
    assert_eq!(out["input"][1]["arguments"], json!({"q":"x"}));
    assert_eq!(out["input"][2]["type"], "function_result");
    assert_eq!(out["input"][2]["name"], "lookup");
    assert_eq!(out["system_instruction"], "be brief");
    assert_eq!(out["generation_config"]["thinking_level"], "high");
    assert_eq!(out["tools"][0]["name"], "lookup");
    assert_eq!(out["generation_config"]["tool_choice"], "auto");
    assert_eq!(out["response_format"]["type"], "json_object");
}

#[test]
fn preserves_explicit_stream_and_previous_response_id() {
    let out = value(convert_openai_responses_request_to_interactions(
        "gpt-test",
        br#"{"input":"hi","stream":false,"previous_response_id":"resp_123"}"#,
        true,
    ));
    assert_eq!(out["stream"], false);
    assert_eq!(out["previous_interaction_id"], "resp_123");

    let out = value(convert_openai_responses_request_to_interactions(
        "gpt-test",
        br#"{"input":"hi"}"#,
        true,
    ));
    assert_eq!(out["stream"], true);
}

#[test]
fn converts_interactions_tool_messages_and_ids() {
    let out = value(convert_interactions_request_to_openai_responses(
        "gpt-test",
        br#"{"input":[
            {"type":"function_call","name":"lookup","call_id":"call_gateway","arguments":{"q":"x"}},
            {"type":"function_result","name":"lookup","call_id":"call_gateway","result":{"ok":true}}
        ]}"#,
        false,
    ));
    assert_eq!(out["input"][0]["type"], "function_call");
    assert_eq!(out["input"][0]["call_id"], "call_gateway");
    assert_eq!(out["input"][0]["arguments"], r#"{"q":"x"}"#);
    assert_eq!(out["input"][1]["type"], "function_call_output");
    assert_eq!(out["input"][1]["call_id"], "call_gateway");
}

#[test]
fn converts_interactions_system_tools_thinking_and_media() {
    let out = value(convert_interactions_request_to_openai_responses(
        "gpt-test",
        br#"{
          "system_instruction":"You are helpful.",
          "previous_interaction_id":"interaction_123",
          "input":[{"type":"model_output","content":[
            {"type":"text","text":"hello"},
            {"type":"image","mime_type":"image/png","data":"aGVsbG8="},
            {"type":"audio","mime_type":"audio/wav","data":"UklGRg=="},
            {"type":"document","mime_type":"application/pdf","data":"JVBERi0="}
          ]}],
          "tools":[{"function_declarations":[{"name":"lookup","description":"Find data","parameters":{"type":"object"}}]}],
          "generation_config":{"tool_choice":"auto","thinking_level":"HIGH","thinking_summaries":"auto"},
          "response_format":{"type":"json_object"}, "response_modalities":["text"],
          "service_tier":"priority", "stream":true
        }"#,
        false,
    ));
    assert_eq!(out["instructions"], "You are helpful.");
    assert_eq!(out["previous_response_id"], "interaction_123");
    assert_eq!(out["input"][0]["content"][0]["type"], "output_text");
    assert_eq!(out["input"][0]["content"][1]["type"], "output_image");
    assert_eq!(
        out["input"][0]["content"][1]["image_url"],
        "data:image/png;base64,aGVsbG8="
    );
    assert_eq!(out["input"][0]["content"][2]["type"], "output_text");
    assert_eq!(out["input"][0]["content"][3]["type"], "output_file");
    assert_eq!(out["tools"][0]["type"], "function");
    assert_eq!(out["tools"][0]["name"], "lookup");
    assert_eq!(out["tool_choice"], "auto");
    assert_eq!(out["reasoning"]["effort"], "high");
    assert_eq!(out["reasoning"]["summary"], "auto");
    assert_eq!(out["text"]["format"]["type"], "json_object");
    assert_eq!(out["modalities"][0], "text");
    assert_eq!(out["service_tier"], "priority");
    assert_eq!(out["stream"], true);
}

#[test]
fn drops_interactions_fields_without_responses_equivalents() {
    let out = value(convert_interactions_request_to_openai_responses(
        "gpt-test",
        br#"{"input":"hi","store":true,"background":true,"webhook_config":{"url":"https://example.com"}}"#,
        false,
    ));
    assert!(out.get("store").is_none());
    assert!(out.get("background").is_none());
    assert!(out.get("webhook_config").is_none());
}
