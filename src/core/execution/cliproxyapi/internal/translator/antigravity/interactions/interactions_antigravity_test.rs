// ref: internal/translator/antigravity/interactions/interactions_antigravity_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{
    convert_antigravity_response_to_interactions,
    convert_antigravity_response_to_interactions_non_stream,
    convert_interactions_request_to_antigravity, AntigravityToInteractionsState,
};
use crate::internal::util::sanitized_function_name_map;

fn parse(raw: &[u8]) -> Value {
    serde_json::from_slice(raw).unwrap()
}

#[test]
fn request_converts_system_steps_tools_and_generation_config() {
    let raw = br#"{
        "system_instruction":"be brief","stream":true,
        "input":[
          {"type":"user_input","content":[{"type":"text","text":"hi"}]},
          {"type":"function_call","name":"lookup","call_id":"call_1","arguments":{"q":"x"}},
          {"type":"function_result","name":"lookup","call_id":"call_1","result":{"ok":true}}
        ],
        "tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}],
        "generation_config":{"max_output_tokens":16,"top_p":0.8,"tool_choice":"auto","thinking_level":"high","thinking_summaries":"auto"},
        "reasoning":{"summary":"auto"}
    }"#;
    let out = parse(&convert_interactions_request_to_antigravity(
        "antigravity-test",
        raw,
        true,
    ));
    assert_eq!(
        out.pointer("/request/systemInstruction/parts/0/text"),
        Some(&Value::String("be brief".into()))
    );
    assert_eq!(
        out.pointer("/request/contents/0/parts/0/text"),
        Some(&Value::String("hi".into()))
    );
    assert_eq!(
        out.pointer("/request/contents/1/parts/0/functionCall/name"),
        Some(&Value::String("lookup".into()))
    );
    assert_eq!(
        out.pointer("/request/contents/2/parts/0/functionResponse/name"),
        Some(&Value::String("lookup".into()))
    );
    assert_eq!(
        out.pointer(
            "/request/tools/0/functionDeclarations/0/parametersJsonSchema/properties/q/type"
        ),
        Some(&Value::String("string".into()))
    );
    assert_eq!(
        out.pointer("/request/generationConfig/maxOutputTokens")
            .and_then(Value::as_i64),
        Some(16)
    );
    assert_eq!(
        out.pointer("/request/generationConfig/thinkingConfig/thinkingLevel"),
        Some(&Value::String("high".into()))
    );
    assert_eq!(
        out.pointer("/request/generationConfig/thinkingConfig/includeThoughts"),
        Some(&Value::Bool(true))
    );
    assert_eq!(
        out.pointer("/request/toolConfig/functionCallingConfig/mode"),
        Some(&Value::String("AUTO".into()))
    );
    assert_eq!(out.pointer("/request/stream"), Some(&Value::Bool(true)));
    assert!(out.get("input").is_none());
}

#[test]
fn reasoning_summary_is_independent_from_effort() {
    for (reasoning, expected) in [
        (r#"{"effort":"high"}"#, None),
        (r#"{"effort":"high","summary":"auto"}"#, Some(true)),
        (r#"{"effort":"high","summary":"none"}"#, Some(false)),
    ] {
        let raw = format!(r#"{{"input":"hi","reasoning":{reasoning}}}"#);
        let out = parse(&convert_interactions_request_to_antigravity(
            "model",
            raw.as_bytes(),
            false,
        ));
        assert_eq!(
            out.pointer("/request/generationConfig/thinkingConfig/thinkingLevel"),
            Some(&Value::String("high".into()))
        );
        assert_eq!(
            out.pointer("/request/generationConfig/thinkingConfig/includeThoughts")
                .and_then(Value::as_bool),
            expected
        );
    }
}

#[test]
fn request_deduplicates_disambiguates_and_rewrites_tool_names() {
    let first = "mcp__plugin_cloudflare_cloudflare-builds__workers_builds_get_build";
    let second = "mcp__plugin_cloudflare_cloudflare-builds__workers_builds_get_build_logs";
    let raw = format!(
        r#"{{
      "input":[
        {{"type":"function_call","name":"{second}","call_id":"call_1","arguments":{{}}}},
        {{"type":"function_result","name":"{second}","call_id":"call_1","result":{{}}}}
      ],
      "tools":[
        {{"functionDeclarations":[{{"name":"lookup"}},{{"name":"{first}"}}]}},
        {{"function_declarations":[{{"name":"lookup"}},{{"name":"{second}"}}]}}
      ],
      "tool_choice":{{"type":"function","function":{{"name":"{second}"}}}}
    }}"#
    );
    let out = parse(&convert_interactions_request_to_antigravity(
        "model",
        raw.as_bytes(),
        false,
    ));
    let declarations = out
        .pointer("/request/tools/0/functionDeclarations")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(declarations.len(), 3);
    let second_mapped = declarations[2]["name"].as_str().unwrap();
    assert_ne!(declarations[1]["name"], declarations[2]["name"]);
    assert!(second_mapped.len() <= 64);
    assert_eq!(
        out.pointer("/request/contents/0/parts/0/functionCall/name")
            .and_then(Value::as_str),
        Some(second_mapped)
    );
    assert_eq!(
        out.pointer("/request/contents/1/parts/0/functionResponse/name")
            .and_then(Value::as_str),
        Some(second_mapped)
    );
    assert_eq!(
        out.pointer("/request/toolConfig/functionCallingConfig/allowedFunctionNames/0")
            .and_then(Value::as_str),
        Some(second_mapped)
    );
}

#[test]
fn request_preserves_name_mapping_whitespace() {
    let raw = br#"{
      "input":[{"type":"function_call","name":" read/file ","arguments":{}}],
      "tools":[{"type":"function","name":" read/file ","parameters":{"type":"object"}}],
      "tool_choice":{"type":"function","function":{"name":" read/file "}}
    }"#;
    let out = parse(&convert_interactions_request_to_antigravity(
        "model", raw, false,
    ));
    let declaration = out
        .pointer("/request/tools/0/functionDeclarations/0/name")
        .and_then(Value::as_str)
        .unwrap();
    assert!(!declaration.is_empty());
    assert_eq!(
        out.pointer("/request/contents/0/parts/0/functionCall/name")
            .and_then(Value::as_str),
        Some(declaration)
    );
    assert_eq!(
        out.pointer("/request/toolConfig/functionCallingConfig/allowedFunctionNames/0")
            .and_then(Value::as_str),
        Some(declaration)
    );
}

#[test]
fn non_stream_response_maps_steps_usage_and_restores_name() {
    let first = "mcp__plugin_cloudflare_cloudflare-builds__workers_builds_get_build";
    let second = "mcp__plugin_cloudflare_cloudflare-builds__workers_builds_get_build_logs";
    let original = format!(r#"{{"tools":[{{"name":"{first}"}},{{"name":"{second}"}}]}}"#);
    let mapped = sanitized_function_name_map(original.as_bytes())[second].clone();
    let raw = format!(
        r#"{{"response":{{"responseId":"resp_1","candidates":[{{"content":{{"parts":[{{"text":"ok"}},{{"functionCall":{{"name":"{mapped}","id":"call_1","args":{{"q":"x"}}}}}}]}}}}],"usageMetadata":{{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}}}}}"#
    );
    let out = parse(&convert_antigravity_response_to_interactions_non_stream(
        "model",
        original.as_bytes(),
        &[],
        raw.as_bytes(),
    ));
    assert_eq!(
        out.pointer("/steps/0/content/0/text"),
        Some(&Value::String("ok".into()))
    );
    assert_eq!(
        out.pointer("/steps/1/type"),
        Some(&Value::String("function_call".into()))
    );
    assert_eq!(
        out.pointer("/steps/1/name"),
        Some(&Value::String(second.into()))
    );
    assert_eq!(
        out.pointer("/usage/total_tokens").and_then(Value::as_i64),
        Some(5)
    );
}

#[test]
fn stream_response_emits_text_and_function_call_id() {
    let mut state =
        AntigravityToInteractionsState::with_identity("interaction_1", "2026-08-03T00:00:00Z");
    let events = convert_antigravity_response_to_interactions(
        "model", &[], &[],
        br#"data: {"response":{"candidates":[{"content":{"parts":[{"text":"ok"},{"functionCall":{"name":"lookup","id":"call_1","args":{"q":"x"}}}]}}]}}"#,
        &mut state,
    );
    let text = event_payload(&events, "step.delta", |value| {
        value.pointer("/delta/text").is_some()
    })
    .unwrap();
    assert_eq!(
        text.pointer("/delta/text"),
        Some(&Value::String("ok".into()))
    );
    let start = event_payload(&events, "step.start", |value| {
        value.pointer("/step/call_id").is_some()
    })
    .unwrap();
    assert_eq!(
        start.pointer("/step/call_id"),
        Some(&Value::String("call_1".into()))
    );
}

fn event_payload(
    events: &[Vec<u8>],
    event_type: &str,
    predicate: impl Fn(&Value) -> bool,
) -> Option<Value> {
    events.iter().find_map(|event| {
        let text = String::from_utf8_lossy(event);
        let data = text.lines().find_map(|line| line.strip_prefix("data: "))?;
        let value = serde_json::from_str::<Value>(data).ok()?;
        (value.get("event_type").and_then(Value::as_str) == Some(event_type) && predicate(&value))
            .then_some(value)
    })
}
