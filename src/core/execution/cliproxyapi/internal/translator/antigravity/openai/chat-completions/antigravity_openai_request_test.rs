// ref: internal/translator/antigravity/openai/chat-completions/antigravity_openai_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};

use super::convert_openai_chat_request_to_antigravity;

fn convert(model: &str, input: Value, stream: bool) -> Value {
    serde_json::from_slice(&convert_openai_chat_request_to_antigravity(
        model,
        &serde_json::to_vec(&input).unwrap(),
        stream,
    ))
    .unwrap()
}

#[test]
fn empty_text_parts_and_empty_assistant_turns_are_removed() {
    let output = convert(
        "gemini-3-flash",
        json!({"messages":[
            {"role":"user","content":[
                {"type":"text","text":""},
                {"type":"input_audio","input_audio":{"data":"SUQzBA==","format":"mp3"}}
            ]},
            {"role":"assistant","content":[{"type":"text","text":""}],"tool_calls":[
                {"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}}
            ]},
            {"role":"tool","tool_call_id":"call_1","content":"{\"output\":\"ok\"}"},
            {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"","arguments":"{}"}}]},
            {"role":"user","content":"done"}
        ]}),
        false,
    );
    let contents = output
        .pointer("/request/contents")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(contents[0]["parts"].as_array().unwrap().len(), 1);
    assert_eq!(
        contents[0]["parts"][0]["inlineData"]["mime_type"],
        "audio/mpeg"
    );
    assert_eq!(contents[1]["parts"].as_array().unwrap().len(), 1);
    assert!(contents[1]["parts"][0].get("functionCall").is_some());
    assert!(!contents.iter().any(|content| content["parts"]
        .as_array()
        .is_some_and(|parts| parts.iter().any(Value::is_null))));
}

#[test]
fn claude_drops_unsigned_reasoning_and_empty_model_turn() {
    let output = convert(
        "claude-sonnet-4-6",
        json!({"messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":"visible text","reasoning_content":"unsigned reasoning"},
            {"role":"assistant","content":"","reasoning_content":"unsigned reasoning"},
            {"role":"user","content":"say ok"}
        ]}),
        false,
    );
    let contents = output
        .pointer("/request/contents")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(contents.len(), 3);
    assert_eq!(contents[1]["parts"].as_array().unwrap().len(), 1);
    assert_eq!(contents[1]["parts"][0]["text"], "visible text");
    assert!(contents[1]["parts"][0].get("thought").is_none());
}

#[test]
fn gemini_preserves_reasoning_before_visible_text_and_tool_call() {
    let output = convert(
        "gemini-3-flash",
        json!({"messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":"visible answer","reasoning_content":"thinking only","tool_calls":[
                {"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"call_1","content":"{\"output\":\"ok\"}"},
            {"role":"user","content":"say ok"}
        ]}),
        true,
    );
    let parts = output
        .pointer("/request/contents/1/parts")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0]["text"], "thinking only");
    assert_eq!(parts[0]["thought"], true);
    assert_eq!(parts[1]["text"], "visible answer");
    assert_eq!(parts[2]["functionCall"]["name"], "read_file");
    assert!(parts[2].get("thoughtSignature").is_some());
    assert_eq!(
        output.pointer("/request/contents/2/parts/0/functionResponse/name"),
        Some(&Value::String("read_file".into()))
    );
}

#[test]
fn thinking_aliases_respect_explicit_summary_visibility() {
    for (extension, expected) in [
        (json!({}), None),
        (json!({"reasoning_effort":"high"}), Some(true)),
        (
            json!({"generationConfig":{"thinkingConfig":{"include_thoughts":true}}}),
            Some(true),
        ),
        (
            json!({"generationConfig":{"thinkingConfig":{"includeThoughts":"true"}}}),
            None,
        ),
        (json!({"thinking":{"include_thoughts":true}}), Some(true)),
        (json!({"reasoning":{"exclude":false}}), Some(true)),
        (json!({"reasoning":{"exclude":true}}), Some(false)),
        (
            json!({"reasoning_effort":"high","extra_body":{"google":{"thinking_config":{"include_thoughts":false}}}}),
            Some(false),
        ),
    ] {
        let mut input = json!({"messages":[{"role":"user","content":"hi"}]});
        input
            .as_object_mut()
            .unwrap()
            .extend(extension.as_object().cloned().unwrap_or_default());
        let output = convert("gemini-3.1-pro-low", input, false);
        assert_eq!(
            output
                .pointer("/request/generationConfig/thinkingConfig/includeThoughts")
                .and_then(Value::as_bool),
            expected
        );
        assert!(output
            .pointer("/request/generationConfig/thinkingConfig/include_thoughts")
            .is_none());
    }
}

#[test]
fn tools_are_deduplicated_disambiguated_and_tool_choice_is_mapped() {
    let first = "mcp__plugin_cloudflare_cloudflare-builds__workers_builds_get_build";
    let second = "mcp__plugin_cloudflare_cloudflare-builds__workers_builds_get_build_logs";
    let output = convert(
        "gemini-3-flash",
        json!({
            "messages":[
                {"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":second,"arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"call_1","content":"{}"}
            ],
            "tools":[
                {"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}},
                {"type":"function","function":{"name":"lookup","description":"duplicate","parameters":{"type":"object"}}},
                {"type":"function","function":{"name":first,"parameters":{"type":"object"}}},
                {"type":"function","function":{"name":second,"parameters":{"type":"object"}}}
            ],
            "tool_choice":{"type":"function","function":{"name":second}}
        }),
        false,
    );
    let declarations = output
        .pointer("/request/tools/0/functionDeclarations")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(declarations.len(), 3);
    let mapped = declarations[2]["name"].as_str().unwrap();
    assert_ne!(declarations[1]["name"], declarations[2]["name"]);
    assert!(mapped.len() <= 64);
    assert_eq!(
        output
            .pointer("/request/contents/0/parts/0/functionCall/name")
            .and_then(Value::as_str),
        Some(mapped)
    );
    assert_eq!(
        output
            .pointer("/request/contents/1/parts/0/functionResponse/name")
            .and_then(Value::as_str),
        Some(mapped)
    );
    assert_eq!(
        output
            .pointer("/request/toolConfig/functionCallingConfig/allowedFunctionNames/0")
            .and_then(Value::as_str),
        Some(mapped)
    );

    for (choice, mode) in [("none", "NONE"), ("auto", "AUTO"), ("required", "ANY")] {
        let output = convert(
            "gemini",
            json!({"messages":[{"role":"user","content":"hi"}],"tool_choice":choice}),
            false,
        );
        assert_eq!(
            output.pointer("/request/toolConfig/functionCallingConfig/mode"),
            Some(&Value::String(mode.into()))
        );
    }
}

#[test]
fn response_format_replaces_all_schema_aliases() {
    let stale = json!({
        "responseSchema":{"type":"string","description":"stale"},
        "responseJsonSchema":{"type":"string"},
        "response_schema":{"type":"string"},
        "response_json_schema":{"type":"string"}
    });
    for (format, schema_expected) in [
        (json!({"type":"json_object"}), false),
        (
            json!({"type":"json_schema","json_schema":{"name":"verdict","schema":{"type":"object","properties":{"score":{"type":"integer"}},"required":["score"]}}}),
            true,
        ),
    ] {
        let output = convert(
            "gemini",
            json!({
                "messages":[{"role":"user","content":"hi"}],
                "generationConfig":stale,"response_format":format,
            }),
            false,
        );
        let generation = output.pointer("/request/generationConfig").unwrap();
        assert_eq!(generation["responseMimeType"], "application/json");
        assert_eq!(generation.get("responseSchema").is_some(), schema_expected);
        if schema_expected {
            assert_eq!(
                generation.pointer("/responseSchema/properties/score/type"),
                Some(&Value::String("integer".into()))
            );
            assert!(generation.pointer("/responseSchema/description").is_none());
        }
        for alias in [
            "responseJsonSchema",
            "response_schema",
            "response_json_schema",
        ] {
            assert!(generation.get(alias).is_none());
        }
    }
}
