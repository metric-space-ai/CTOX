// ref: internal/translator/claude/openai/responses/claude_openai-responses_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::convert_openai_responses_request_to_claude;
use base64::{engine::general_purpose, Engine as _};
use serde_json::{json, Value};

fn convert(input: Value) -> Value {
    serde_json::from_slice(&convert_openai_responses_request_to_claude(
        "claude-test",
        &serde_json::to_vec(&input).unwrap(),
        false,
    ))
    .unwrap()
}

fn append_bytes(target: &mut Vec<u8>, field: u8, value: &[u8]) {
    target.push((field << 3) | 2);
    target.push(value.len() as u8);
    target.extend_from_slice(value);
}

fn claude_signature() -> String {
    let mut channel = vec![0x08, 0x0c, 0x10, 0x02];
    append_bytes(&mut channel, 6, b"claude-sonnet-4-6");
    let mut container = Vec::new();
    append_bytes(&mut container, 1, &channel);
    let mut payload = Vec::new();
    append_bytes(&mut payload, 2, &container);
    payload.extend_from_slice(&[0x18, 0x01]);
    general_purpose::STANDARD.encode(payload)
}

#[test]
fn sanitizes_call_ids_and_keeps_tool_use_adjacent_to_its_result() {
    let output = convert(json!({"input":[
        {"type":"function_call","call_id":"call.with space:1","name":"Read","arguments":"{}"},
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"working"}]},
        {"type":"function_call_output","call_id":"call.with space:1","output":"ok"}
    ]}));
    assert_eq!(output["messages"][0]["content"], "working");
    assert_eq!(
        output["messages"][1]["content"][0]["id"],
        "call_with_space_1"
    );
    assert_eq!(
        output["messages"][2]["content"][0]["tool_use_id"],
        "call_with_space_1"
    );
}

#[test]
fn function_output_preserves_image_as_a_claude_media_block() {
    let output = convert(json!({"input":[
        {"type":"function_call","call_id":"call_1","name":"view_image","arguments":"{}"},
        {"type":"function_call_output","call_id":"call_1","output":[
            {"type":"input_image","image_url":"data:image/png;base64,AA==","detail":"high"}
        ]}
    ]}));
    let part = &output["messages"][1]["content"][0]["content"][0];
    assert_eq!(part["type"], "image");
    assert_eq!(part["source"]["media_type"], "image/png");
    assert_eq!(part["source"]["data"], "AA==");
}

#[test]
fn filters_custom_apply_patch_normalizes_union_and_preserves_cache_control() {
    let output = convert(json!({
        "input":[{"type":"message","role":"user","content":[
            {"type":"input_text","text":"cached","cache_control":{"type":"ephemeral"}},
            {"type":"input_text","text":"fresh"}
        ]}],
        "tools":[
            {"type":"custom","name":"apply_patch"},
            {"type":"function","name":"lookup","parameters":{
                "oneOf":[
                    {"type":"object","properties":{"query":{"type":"string"}}},
                    {"type":"object","properties":{"id":{"type":"string"}}}
                ]
            }}
        ]
    }));
    assert_eq!(output["tools"].as_array().unwrap().len(), 1);
    let schema = &output["tools"][0]["input_schema"];
    assert_eq!(schema["type"], "object");
    assert!(schema.get("oneOf").is_none());
    assert!(schema["properties"].get("query").is_some());
    assert!(schema["properties"].get("id").is_some());
    assert_eq!(
        output["messages"][0]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert!(output["messages"][0]["content"][1]
        .get("cache_control")
        .is_none());
}

#[test]
fn redacted_reasoning_restores_redacted_thinking_and_drops_empty_data() {
    let output = convert(json!({"input":[
        {"type":"reasoning","encrypted_content":"claude-redacted-thinking:EroBCkYIBRgCKkA","summary":[]},
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"visible answer"}]},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
    ]}));
    assert_eq!(
        output["messages"][0]["content"][0]["type"],
        "redacted_thinking"
    );
    assert_eq!(
        output["messages"][0]["content"][0]["data"],
        "EroBCkYIBRgCKkA"
    );
    assert!(output["messages"][0]["content"][0]
        .get("signature")
        .is_none());

    let empty = convert(json!({"input":[
        {"type":"reasoning","encrypted_content":"claude-redacted-thinking:","summary":[]},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
    ]}));
    assert_eq!(empty["messages"].as_array().unwrap().len(), 1);
    assert_eq!(empty["messages"][0]["role"], "user");
}

#[test]
fn reasoning_content_rebuilds_thinking_but_summary_wins_when_both_exist() {
    let signature = claude_signature();
    let from_content = convert(json!({"input":[
        {"type":"reasoning","encrypted_content":signature,"summary":[],"content":[{"type":"reasoning_text","text":"restored from content"}]},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
    ]}));
    assert_eq!(
        from_content["messages"][0]["content"][0]["thinking"],
        "restored from content"
    );

    let deduplicated = convert(json!({"input":[
        {"type":"reasoning","encrypted_content":claude_signature(),"summary":[{"type":"summary_text","text":"chain of thought"}],"content":[{"type":"reasoning_text","text":"chain of thought"}]},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
    ]}));
    assert_eq!(
        deduplicated["messages"][0]["content"][0]["thinking"],
        "chain of thought"
    );
}

#[test]
fn system_level_inputs_become_separate_top_level_blocks() {
    let output = convert(json!({
        "instructions":"I1",
        "input":[
            {"type":"message","role":"system","content":[{"type":"input_text","text":"S1"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"U1"}]},
            {"type":"message","role":"developer","content":"D1"},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"A1"}]},
            {"type":"message","role":"system","content":[{"type":"input_text","text":"S2"}]}
        ]
    }));
    let system = output["system"].as_array().unwrap();
    assert_eq!(system.len(), 4);
    for (block, expected) in system.iter().zip(["I1", "S1", "D1", "S2"]) {
        assert_eq!(block["type"], "text");
        assert_eq!(block["text"], expected);
    }
    assert_eq!(output["messages"].as_array().unwrap().len(), 2);
    assert_eq!(output["messages"][0]["role"], "user");
    assert_eq!(output["messages"][1]["role"], "assistant");
}

#[test]
fn system_only_input_keeps_fallback_user_message() {
    let output = convert(json!({"instructions":"I1"}));
    assert_eq!(output["system"].as_array().unwrap().len(), 1);
    assert_eq!(output["messages"].as_array().unwrap().len(), 1);
    assert_eq!(output["messages"][0]["role"], "user");
}

#[test]
fn unsupported_system_part_is_a_payload_free_typed_marker() {
    let output = convert(json!({"input":[
        {"type":"message","role":"developer","content":[
            {"type":"input_text","text":"D1"},
            {"type":"input_image","image_url":"data:image/png;base64,AAAA"}
        ]},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"U1"}]}
    ]}));
    assert_eq!(output["system"][0]["text"], "D1");
    assert_eq!(output["system"][1]["type"], "input_image");
    assert!(output["system"][1].get("source").is_none());
    assert!(output["system"][1].get("image_url").is_none());
}

#[test]
fn system_item_cache_control_applies_only_to_last_uncached_block() {
    let output = convert(json!({"input":[
        {"type":"message","role":"system","cache_control":{"type":"ephemeral"},"content":[
            {"type":"input_text","text":"S1"},
            {"type":"input_text","text":"S2"}
        ]},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"U1"}]}
    ]}));
    assert!(output["system"][0].get("cache_control").is_none());
    assert_eq!(output["system"][1]["cache_control"]["type"], "ephemeral");
}
