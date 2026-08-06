// ref: internal/translator/openai/interactions/chat-completions/interactions_openai_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{convert_interactions_request_to_openai, convert_openai_request_to_interactions};

fn parse(raw: &[u8]) -> Value {
    serde_json::from_slice(raw).unwrap()
}

#[test]
fn preserves_expressible_fields_for_interactions_to_openai() {
    let out = parse(&convert_interactions_request_to_openai(
        "gpt-test",
        br#"{"model":"gpt-test","tool_choice":{"type":"function","function":{"name":"lookup"}},"response_modalities":["text","image"],"service_tier":"priority","input":"hi"}"#,
        false,
    ));
    assert_eq!(out["tool_choice"]["type"], "function");
    assert_eq!(out["tool_choice"]["function"]["name"], "lookup");
    assert_eq!(out["modalities"][0], "text");
    assert_eq!(out["modalities"][1], "image");
    assert_eq!(out["service_tier"], "priority");
}

#[test]
fn maps_openai_messages_tools_and_stream_to_interactions() {
    // The user message contains non-ASCII characters; build the request body
    // through a UTF-8 `String` so the bytes survive the byte-string parser
    // without tripping rustfmt's raw-byte-string ASCII check.
    let payload = String::from(
        r#"{"model":"gemini-3.1-flash-lite","stream":true,"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"今天北京的天气怎么样？"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"tool_choice":"auto","max_completion_tokens":128}"#,
    );
    let out = parse(&convert_openai_request_to_interactions(
        "gemini-3.1-flash-lite",
        payload.as_bytes(),
        false,
    ));
    assert_eq!(out["model"], "gemini-3.1-flash-lite");
    assert_eq!(out["stream"], true);
    assert_eq!(out["system_instruction"], "be brief");
    assert_eq!(out["input"][0]["type"], "user_input");
    assert_eq!(
        out["input"][0]["content"][0]["text"],
        "今天北京的天气怎么样？"
    );
    assert_eq!(out["tools"][0]["type"], "function");
    assert_eq!(out["tools"][0]["name"], "get_weather");
    assert_eq!(
        out["tools"][0]["parameters"]["properties"]["location"]["type"],
        "string"
    );
    assert_eq!(out["generation_config"]["tool_choice"], "auto");
    assert_eq!(out["generation_config"]["max_output_tokens"], 128);
}

#[test]
fn maps_openai_tool_calls_and_results_to_interactions() {
    let out = parse(&convert_openai_request_to_interactions(
        "gemini-3.1-flash-lite",
        br#"{"model":"gemini-3.1-flash-lite","messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"ok"}]}"#,
        false,
    ));
    assert_eq!(out["input"][0]["type"], "function_call");
    assert_eq!(out["input"][0]["call_id"], "call_1");
    assert_eq!(out["input"][0]["arguments"]["q"], "x");
    assert_eq!(out["input"][1]["type"], "function_result");
    assert_eq!(out["input"][1]["result"], "ok");
}

#[test]
fn accepts_image_content_in_interactions_to_openai() {
    let out = parse(&convert_interactions_request_to_openai(
        "gpt-test",
        br#"{"model":"gpt-test","input":[{"type":"user_input","content":[{"type":"image","mime_type":"image/png","data":"aGVsbG8="}]}]}"#,
        false,
    ));
    assert_eq!(out["messages"][0]["content"][0]["type"], "image_url");
    assert_eq!(
        out["messages"][0]["content"][0]["image_url"]["url"],
        "data:image/png;base64,aGVsbG8="
    );
}

#[test]
fn preserves_non_image_media_content() {
    let out = parse(&convert_interactions_request_to_openai(
        "gpt-test",
        br#"{"model":"gpt-test","input":[{"type":"user_input","content":[{"type":"audio","mime_type":"audio/wav","data":"UklGRg=="},{"type":"video","mime_type":"video/mp4","data":"AAAAIGZ0eXA="},{"type":"document","mime_type":"application/pdf","data":"JVBERi0="}]}]}"#,
        false,
    ));
    assert_eq!(out["messages"][0]["content"][0]["type"], "input_audio");
    assert_eq!(
        out["messages"][0]["content"][0]["input_audio"]["format"],
        "wav"
    );
    assert_eq!(out["messages"][0]["content"][1]["type"], "video_url");
    assert_eq!(out["messages"][0]["content"][2]["type"], "file");
}

#[test]
fn maps_interactions_tool_messages_directly() {
    let out = parse(&convert_interactions_request_to_openai(
        "gpt-test",
        br#"{"model":"gpt-test","input":[{"type":"user_input","content":[{"type":"text","text":"hi"}]},{"type":"function_call","name":"lookup","call_id":"call_1","arguments":{"q":"x"}},{"type":"function_result","name":"lookup","call_id":"call_1","result":{"ok":true}}]}"#,
        false,
    ));
    assert_eq!(
        out["messages"][1]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert_eq!(
        out["messages"][1]["tool_calls"][0]["function"]["arguments"],
        "{\"q\":\"x\"}"
    );
    assert_eq!(out["messages"][2]["tool_call_id"], "call_1");
}
