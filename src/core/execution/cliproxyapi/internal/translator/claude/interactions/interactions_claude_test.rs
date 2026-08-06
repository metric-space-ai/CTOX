// ref: internal/translator/claude/interactions/interactions_claude_test.go:1-181 @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{
    convert_claude_response_to_interactions, convert_claude_response_to_interactions_non_stream,
    convert_interactions_request_to_claude, ClaudeToInteractionsState,
};

fn request(raw: &str) -> Value {
    serde_json::from_slice(&convert_interactions_request_to_claude(
        "claude-test",
        raw.as_bytes(),
        false,
    ))
    .unwrap()
}

#[test]
fn invalid_json_is_a_byte_identical_noop() {
    let raw = br#" {not-json}\n"#;
    assert_eq!(
        convert_interactions_request_to_claude("claude-test", raw, false),
        raw
    );
}

#[test]
fn converts_tool_messages_and_system_directly() {
    let out = request(
        r#"{"model":"claude-test","system_instruction":"be brief","input":[{"type":"user_input","content":[{"type":"text","text":"hi"}]},{"type":"function_call","name":"lookup","call_id":"toolu_1","arguments":{"q":"x"}},{"type":"function_result","name":"lookup","call_id":"toolu_1","result":{"ok":true}}]}"#,
    );
    assert_eq!(out["system"], "be brief");
    assert_eq!(out.pointer("/messages/0/content/0/text").unwrap(), "hi");
    assert_eq!(
        out.pointer("/messages/1/content/0/type").unwrap(),
        "tool_use"
    );
    assert_eq!(
        out.pointer("/messages/2/content/0/type").unwrap(),
        "tool_result"
    );
    assert_eq!(
        out.pointer("/messages/2/content/0/tool_use_id").unwrap(),
        "toolu_1"
    );
}

#[test]
fn converts_string_input_directly() {
    let out = request(r#"{"model":"claude-test","input":"hello"}"#);
    assert_eq!(out.pointer("/messages/0/role").unwrap(), "user");
    assert_eq!(out.pointer("/messages/0/content/0/text").unwrap(), "hello");
}

#[test]
fn maps_generation_config_tools_and_body_stream() {
    let out = request(
        r#"{"model":"claude-test","stream":true,"input":[{"type":"user_input","content":[{"type":"text","text":"hi"}]}],"tools":[{"type":"function","name":"lookup","description":"Lookup data","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}],"generation_config":{"max_output_tokens":99,"top_p":0.7,"stop_sequences":["END"],"tool_choice":{"type":"function","name":"lookup"},"thinking_level":"high"}}"#,
    );
    assert_eq!(out["stream"], true);
    assert_eq!(out["max_tokens"], 99);
    assert_eq!(
        out.pointer("/tools/0/input_schema/properties/q/type")
            .unwrap(),
        "string"
    );
    assert_eq!(out.pointer("/tool_choice/name").unwrap(), "lookup");
    assert!(out.pointer("/thinking/type").is_some());
}

#[test]
fn accepts_image_content() {
    let out = request(
        r#"{"input":[{"type":"user_input","content":[{"type":"image","mime_type":"image/png","data":"aGVsbG8="}]}]}"#,
    );
    assert_eq!(out.pointer("/messages/0/content/0/type").unwrap(), "image");
    assert_eq!(
        out.pointer("/messages/0/content/0/source/media_type")
            .unwrap(),
        "image/png"
    );
    assert_eq!(
        out.pointer("/messages/0/content/0/source/data").unwrap(),
        "aGVsbG8="
    );
}

#[test]
fn preserves_non_image_media_as_fallback_or_document() {
    let out = request(
        r#"{"input":[{"type":"thought","content":[{"type":"audio","mime_type":"audio/wav","data":"UklGRg=="},{"type":"video","mime_type":"video/mp4","data":"AAAAIGZ0eXA="},{"type":"document","mime_type":"application/pdf","data":"JVBERi0="}]}]}"#,
    );
    assert_eq!(out.pointer("/messages/0/role").unwrap(), "assistant");
    assert_eq!(out.pointer("/messages/0/content/0/type").unwrap(), "text");
    assert_eq!(out.pointer("/messages/0/content/1/type").unwrap(), "text");
    assert_eq!(
        out.pointer("/messages/0/content/2/type").unwrap(),
        "document"
    );
}

#[test]
fn converts_non_stream_message_and_usage() {
    let raw = br#"{"id":"msg_1","model":"claude-test","content":[{"type":"thinking","thinking":"reasoning"},{"type":"text","text":"ok"},{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"x"}}],"usage":{"input_tokens":3,"output_tokens":2,"cache_read_input_tokens":1,"cache_creation_input_tokens":4,"thinking_tokens":5}}"#;
    let out: Value = serde_json::from_slice(&convert_claude_response_to_interactions_non_stream(
        "claude-test",
        &[],
        &[],
        raw,
    ))
    .unwrap();
    assert_eq!(out.pointer("/steps/0/type").unwrap(), "thought");
    assert_eq!(out.pointer("/steps/1/content/0/text").unwrap(), "ok");
    assert_eq!(out.pointer("/steps/2/call_id").unwrap(), "toolu_1");
    assert_eq!(out.pointer("/usage/total_tokens").unwrap(), 5);
    assert_eq!(out.pointer("/usage/total_cached_tokens").unwrap(), 5);
}

#[test]
fn converts_sse_to_non_stream() {
    let raw = br#"data: {"type":"message_start","message":{"id":"msg_1","model":"claude-test","usage":{"input_tokens":3,"output_tokens":0}}}
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}
data: {"type":"content_block_stop","index":0}
data: {"type":"message_delta","usage":{"output_tokens":2}}"#;
    let out: Value = serde_json::from_slice(&convert_claude_response_to_interactions_non_stream(
        "claude-test",
        &[],
        &[],
        raw,
    ))
    .unwrap();
    assert_eq!(out.pointer("/steps/0/content/0/text").unwrap(), "ok");
    assert_eq!(out.pointer("/usage/total_tokens").unwrap(), 5);
}

#[test]
fn stream_merges_usage_and_status() {
    let mut state = ClaudeToInteractionsState::default();
    let mut events = Vec::new();
    for raw in [
        br#"data: {"type":"message_start","message":{"id":"msg_1","model":"claude-test","usage":{"input_tokens":3,"output_tokens":0}}}"#.as_slice(),
        br#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#.as_slice(),
        br#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}"#.as_slice(),
        br#"data: {"type":"content_block_stop","index":0}"#.as_slice(),
        br#"data: {"type":"message_delta","usage":{"output_tokens":2}}"#.as_slice(),
    ] {
        events.extend(convert_claude_response_to_interactions(
            "claude-test", &[], &[], raw, &mut state,
        ));
    }
    assert!(payload(&events, "interaction.status_update").is_some());
    let completed = payload(&events, "interaction.completed").unwrap();
    assert_eq!(
        completed
            .pointer("/interaction/usage/total_input_tokens")
            .unwrap(),
        3
    );
    assert_eq!(
        completed
            .pointer("/interaction/usage/total_output_tokens")
            .unwrap(),
        2
    );
    assert_eq!(
        completed
            .pointer("/interaction/usage/total_tokens")
            .unwrap(),
        5
    );
}

#[test]
fn stream_emits_text_delta() {
    let mut state = ClaudeToInteractionsState::default();
    let events = convert_claude_response_to_interactions(
        "claude-test", &[], &[],
        br#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}"#,
        &mut state,
    );
    assert_eq!(
        payload(&events, "step.delta")
            .unwrap()
            .pointer("/delta/text")
            .unwrap(),
        "ok"
    );
}

fn payload(events: &[Vec<u8>], event_type: &str) -> Option<Value> {
    events.iter().find_map(|event| {
        event.split(|byte| *byte == b'\n').find_map(|line| {
            let payload = line.strip_prefix(b"data: ")?;
            let value = serde_json::from_slice::<Value>(payload).ok()?;
            (value.get("event_type").and_then(Value::as_str) == Some(event_type)).then_some(value)
        })
    })
}
