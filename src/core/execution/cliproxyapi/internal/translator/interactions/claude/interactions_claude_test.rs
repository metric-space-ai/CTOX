// ref: internal/translator/interactions/claude/interactions_claude_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{
    convert_claude_request_to_interactions, convert_interactions_response_to_claude,
    convert_interactions_response_to_claude_non_stream, InteractionsToClaudeStreamState,
};

fn parse(raw: &[u8]) -> Value {
    serde_json::from_slice(raw).unwrap()
}

#[test]
fn request_maps_messages_tools_stream_and_generation_config() {
    let out = parse(&convert_claude_request_to_interactions(
        "gemini-3.1-flash-lite",
        r#"{"model":"gemini-3.1-flash-lite","stream":true,"max_tokens":1024,"tools":[{"name":"get_weather","description":"Weather","input_schema":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}],"messages":[{"role":"user","content":[{"type":"text","text":"今天北京的天气怎么样？"}]}]}"#.as_bytes(),
        true,
    ));
    assert_eq!(out["model"], "gemini-3.1-flash-lite");
    assert_eq!(out["stream"], true);
    assert_eq!(
        out.pointer("/generation_config/max_output_tokens"),
        Some(&Value::from(1024))
    );
    assert_eq!(
        out.pointer("/input/0/type"),
        Some(&Value::String("user_input".into()))
    );
    assert_eq!(
        out.pointer("/input/0/content/0/text"),
        Some(&Value::String("今天北京的天气怎么样？".into()))
    );
    assert_eq!(
        out.pointer("/tools/0/parameters/properties/location/type"),
        Some(&Value::String("string".into()))
    );
    assert_eq!(
        out.pointer("/tools/0/type"),
        Some(&Value::String("function".into()))
    );
}

#[test]
fn request_maps_tool_use_result_system_thinking_and_choice() {
    let out = parse(&convert_claude_request_to_interactions(
        "",
        r#"{"model":"gemini-3.1-flash-lite","system":[{"type":"text","text":"one"},{"type":"text","text":"two"}],"thinking":{"type":"enabled","budget_tokens":2048},"output_config":{"effort":"low"},"tool_choice":{"type":"tool","name":"get_weather"},"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"plan"},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"location":"北京"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"晴"}]}]}"#.as_bytes(),
        false,
    ));
    assert_eq!(out["model"], "gemini-3.1-flash-lite");
    assert_eq!(out["system_instruction"], "one\ntwo");
    assert_eq!(
        out.pointer("/generation_config/thinking_level"),
        Some(&Value::String("low".into()))
    );
    assert_eq!(
        out.pointer("/generation_config/thinking_config/thinking_budget"),
        Some(&Value::from(2048))
    );
    assert_eq!(
        out.pointer("/generation_config/tool_choice/name"),
        Some(&Value::String("get_weather".into()))
    );
    assert_eq!(
        out.pointer("/input/0/type"),
        Some(&Value::String("thought".into()))
    );
    assert_eq!(
        out.pointer("/input/1/type"),
        Some(&Value::String("function_call".into()))
    );
    assert_eq!(
        out.pointer("/input/1/call_id"),
        Some(&Value::String("toolu_1".into()))
    );
    assert_eq!(
        out.pointer("/input/2/type"),
        Some(&Value::String("function_result".into()))
    );
    assert_eq!(
        out.pointer("/input/2/result"),
        Some(&Value::String("晴".into()))
    );
}

#[test]
fn stream_maps_text_and_finish_usage() {
    let mut state = InteractionsToClaudeStreamState::with_identity("msg_test");
    let mut events = Vec::new();
    for raw in [
        b"event: interaction.created\ndata: {\"interaction\":{\"id\":\"interaction_1\",\"model\":\"gemini-3.1-flash-lite\"},\"event_type\":\"interaction.created\"}".as_slice(),
        b"event: step.start\ndata: {\"index\":0,\"step\":{\"type\":\"model_output\"},\"event_type\":\"step.start\"}".as_slice(),
        "event: step.delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"北京今天晴\"},\"event_type\":\"step.delta\"}".as_bytes(),
        b"event: step.stop\ndata: {\"index\":0,\"event_type\":\"step.stop\"}".as_slice(),
        b"event: interaction.completed\ndata: {\"interaction\":{\"usage\":{\"total_input_tokens\":3,\"total_output_tokens\":4}},\"event_type\":\"interaction.completed\"}".as_slice(),
        b"data: [DONE]".as_slice(),
    ] {
        events.extend(convert_interactions_response_to_claude(
            "gemini-3.1-flash-lite", &[], &[], raw, &mut state,
        ));
    }
    assert_eq!(
        event_payload(&events, "message_start")
            .unwrap()
            .pointer("/message/model"),
        Some(&Value::String("gemini-3.1-flash-lite".into()))
    );
    assert_eq!(
        event_payload(&events, "content_block_delta")
            .unwrap()
            .pointer("/delta/text"),
        Some(&Value::String("北京今天晴".into()))
    );
    assert_eq!(
        event_payload(&events, "message_delta")
            .unwrap()
            .pointer("/usage/output_tokens"),
        Some(&Value::from(4))
    );
    assert_eq!(
        event_payload(&events, "message_stop").unwrap()["type"],
        "message_stop"
    );
}

#[test]
fn stream_maps_tool_call_signature_and_stop_reason() {
    let mut state = InteractionsToClaudeStreamState::with_identity("msg_test");
    let mut events = Vec::new();
    for raw in [
        b"data: {\"interaction\":{\"id\":\"interaction_1\",\"model\":\"gemini-3.1-flash-lite\"},\"event_type\":\"interaction.created\"}".as_slice(),
        b"data: {\"index\":0,\"step\":{\"type\":\"function_call\",\"id\":\"toolu_1\",\"signature\":\"sig_1\",\"name\":\"get_weather\",\"arguments\":{}},\"event_type\":\"step.start\"}".as_slice(),
        "data: {\"index\":0,\"delta\":{\"type\":\"arguments_delta\",\"arguments\":\"{\\\"location\\\":\\\"北京\\\"}\"},\"event_type\":\"step.delta\"}".as_bytes(),
        b"data: {\"index\":0,\"event_type\":\"step.stop\"}".as_slice(),
        b"data: {\"interaction\":{\"usage\":{\"total_input_tokens\":1,\"total_output_tokens\":2}},\"event_type\":\"interaction.completed\"}".as_slice(),
    ] {
        events.extend(convert_interactions_response_to_claude(
            "gemini-3.1-flash-lite", &[], &[], raw, &mut state,
        ));
    }
    let start = event_payload(&events, "content_block_start").unwrap();
    assert_eq!(
        start.pointer("/content_block/type"),
        Some(&Value::String("tool_use".into()))
    );
    assert_eq!(
        start.pointer("/content_block/signature"),
        Some(&Value::String("sig_1".into()))
    );
    assert_eq!(
        event_payload(&events, "content_block_delta")
            .unwrap()
            .pointer("/delta/partial_json"),
        Some(&Value::String(r#"{"location":"北京"}"#.into()))
    );
    assert_eq!(
        event_payload(&events, "message_delta")
            .unwrap()
            .pointer("/delta/stop_reason"),
        Some(&Value::String("tool_use".into()))
    );
}

#[test]
fn stream_finish_reads_metadata_total_usage_and_injected_id() {
    let mut state = InteractionsToClaudeStreamState::with_identity("msg_test");
    let events = convert_interactions_response_to_claude(
        "claude-test", &[], &[],
        br#"data: {"event_type":"finish","metadata":{"total_usage":{"total_input_tokens":2,"total_output_tokens":6,"total_tokens":8}}}"#,
        &mut state,
    );
    let delta = event_payload(&events, "message_delta").unwrap();
    assert_eq!(delta.pointer("/usage/input_tokens"), Some(&Value::from(2)));
    assert_eq!(delta.pointer("/usage/output_tokens"), Some(&Value::from(6)));
    assert_eq!(
        event_payload(&events, "message_start")
            .unwrap()
            .pointer("/message/id"),
        Some(&Value::String("msg_test".into()))
    );
}

#[test]
fn non_stream_maps_text_tool_signature_stop_and_usage() {
    let out = parse(&convert_interactions_response_to_claude_non_stream(
        "gemini-3.1-flash-lite", &[], &[],
        br#"{"id":"interaction_1","model":"gemini-3.1-flash-lite","steps":[{"type":"model_output","content":[{"type":"text","text":"ok"}]},{"type":"function_call","call_id":"toolu_1","signature":"sig_1","name":"lookup","arguments":{"q":"x"}}],"usage":{"total_input_tokens":3,"total_output_tokens":4}}"#,
    ));
    assert_eq!(
        out.pointer("/content/0/text"),
        Some(&Value::String("ok".into()))
    );
    assert_eq!(
        out.pointer("/content/1/type"),
        Some(&Value::String("tool_use".into()))
    );
    assert_eq!(
        out.pointer("/content/1/signature"),
        Some(&Value::String("sig_1".into()))
    );
    assert_eq!(out["stop_reason"], "tool_use");
    assert_eq!(out.pointer("/usage/input_tokens"), Some(&Value::from(3)));
}

fn event_payload(events: &[Vec<u8>], name: &str) -> Option<Value> {
    let marker = format!("event: {name}");
    events.iter().find_map(|event| {
        event
            .windows(marker.len())
            .any(|window| window == marker.as_bytes())
            .then(|| {
                event.split(|byte| *byte == b'\n').find_map(|line| {
                    serde_json::from_slice::<Value>(line.strip_prefix(b"data: ")?).ok()
                })
            })
            .flatten()
    })
}
