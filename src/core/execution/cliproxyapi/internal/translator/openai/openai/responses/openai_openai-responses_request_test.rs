// ref: internal/translator/openai/openai/responses/openai_openai-responses_request_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::convert_openai_responses_request_to_openai_chat_completions;
use serde_json::{json, Value};

fn pretty(raw: &[u8]) -> String {
    let value: Value = serde_json::from_slice(raw).unwrap();
    serde_json::to_string_pretty(&value).unwrap()
}

fn value(raw: Vec<u8>) -> Value {
    serde_json::from_slice(&raw).unwrap()
}

#[test]
fn merges_consecutive_function_calls() {
    let raw = br#"{
        "input": [
            {"type":"function_call","call_id":"exec_command:0","name":"exec_command","arguments":"{\"cmd\":\"ls\"}"},
            {"type":"function_call","call_id":"exec_command:1","name":"exec_command","arguments":"{\"cmd\":\"pwd\"}"},
            {"type":"function_call_output","call_id":"exec_command:0","output":"ok0"},
            {"type":"function_call_output","call_id":"exec_command:1","output":"ok1"}
        ]
    }"#;
    eprintln!("input:\n{}", pretty(raw));
    let out = convert_openai_responses_request_to_openai_chat_completions("kimi-k2.6", raw, true);
    eprintln!("output:\n{}", pretty(&out));
    let value = value(out);
    assert_eq!(value["messages"].as_array().unwrap().len(), 3);
    assert_eq!(value["messages"][0]["role"], "assistant");
    assert_eq!(
        value["messages"][0]["tool_calls"].as_array().unwrap().len(),
        2
    );
    assert_eq!(
        value["messages"][0]["tool_calls"][0]["id"],
        "exec_command:0"
    );
    assert_eq!(
        value["messages"][0]["tool_calls"][1]["id"],
        "exec_command:1"
    );
    assert_eq!(value["messages"][1]["tool_call_id"], "exec_command:0");
    assert_eq!(value["messages"][2]["tool_call_id"], "exec_command:1");
}

#[test]
fn splits_function_calls_when_interrupted() {
    let raw = br#"{
        "input": [
            {"type":"function_call","call_id":"call_a","name":"tool_a","arguments":"{}"},
            {"type":"message","role":"user","content":"next"},
            {"type":"function_call","call_id":"call_b","name":"tool_b","arguments":"{}"}
        ]
    }"#;
    let out = convert_openai_responses_request_to_openai_chat_completions("kimi-k2.6", raw, false);
    let value = value(out);
    assert_eq!(value["messages"].as_array().unwrap().len(), 3);
    assert_eq!(value["messages"][0]["tool_calls"][0]["id"], "call_a");
    assert_eq!(value["messages"][2]["tool_calls"][0]["id"], "call_b");
}

#[test]
fn defers_message_until_tool_output() {
    let raw = br#"{
        "input": [
            {"type":"function_call","call_id":"call_x","name":"exec_command","arguments":"{\"cmd\":\"echo hi\"}"},
            {"type":"message","role":"user","content":"Approved command prefix saved"},
            {"type":"function_call_output","call_id":"call_x","output":"ok"},
            {"type":"message","role":"user","content":"next"}
        ]
    }"#;
    let out = convert_openai_responses_request_to_openai_chat_completions("kimi-k2.6", raw, true);
    let value = value(out);
    assert_eq!(value["messages"].as_array().unwrap().len(), 4);
    assert_eq!(value["messages"][0]["role"], "assistant");
    assert_eq!(value["messages"][1]["role"], "tool");
    assert_eq!(value["messages"][1]["tool_call_id"], "call_x");
    assert_eq!(value["messages"][2]["role"], "user");
    assert_eq!(
        value["messages"][2]["content"],
        "Approved command prefix saved"
    );
    assert_eq!(value["messages"][3]["content"], "next");
}

#[test]
fn unwraps_stringified_tool_output_images() {
    let tests = [
        (
            "Codex input image",
            r#"[{"type":"input_text","text":"Captured screenshot."},{"detail":"original","image_url":"data:image/png;base64,AA==","type":"input_image"}]"#,
            1,
            "data:image/png;base64,AA==",
            Some("Captured screenshot."),
            "high",
        ),
        (
            "OpenAI image URL",
            r#"[{"type":"image_url","image_url":{"url":"https://example.com/generated.png","detail":"high"}}]"#,
            0,
            "https://example.com/generated.png",
            None,
            "high",
        ),
    ];
    for (name, output, image_index, expected_url, expected_text, detail) in tests {
        let raw = format!(
            r#"{{
                "input": [
                    {{"type":"function_call","call_id":"call_image","name":"view_image","arguments":"{{}}"}},
                    {{"type":"function_call_output","call_id":"call_image","output":{output:?}}}
                ]
            }}"#
        );
        let out = convert_openai_responses_request_to_openai_chat_completions(
            "k3",
            raw.as_bytes(),
            false,
        );
        let value = value(out);
        let content = &value["messages"][1]["content"];
        assert!(content.is_array(), "{name}: expected array, got {content}");
        let parts = content.as_array().unwrap();
        assert!(parts.len() > image_index, "{name}: image part missing");
        let image_part = &parts[image_index];
        assert_eq!(image_part["type"], "image_url", "{name}");
        assert_eq!(image_part["image_url"]["url"], expected_url, "{name}");
        assert_eq!(image_part["image_url"]["detail"], detail, "{name}");
        if let Some(text) = expected_text {
            assert_eq!(parts[0]["type"], "text", "{name}");
            assert_eq!(parts[0]["text"], text, "{name}");
        }
    }
}

#[test]
fn converts_structured_tool_output_images() {
    let raw = br#"{
        "input": [
            {"type":"function_call","call_id":"call_image","name":"view_image","arguments":"{}"},
            {
                "type":"function_call_output",
                "call_id":"call_image",
                "output":[
                    {"type":"input_text","text":"Captured screenshot."},
                    {"type":"input_image","image_url":"data:image/png;base64,AA==","detail":"original"}
                ]
            }
        ]
    }"#;
    let out = convert_openai_responses_request_to_openai_chat_completions("k3", raw, false);
    let value = value(out);
    let content = &value["messages"][1]["content"];
    assert!(content.is_array());
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,AA==");
    assert_eq!(content[1]["image_url"]["detail"], "high");
}

#[test]
fn keeps_non_image_tool_output_strings() {
    let cases = [
        ("plain text", "plain output"),
        ("JSON object", r#"{"status":"ok"}"#),
        (
            "text-only array",
            r#"[{"type":"input_text","text":"still text"}]"#,
        ),
        (
            "invalid image array",
            r#"[{"type":"input_image","detail":"low"}]"#,
        ),
        (
            "image array with trailing text",
            r#"[{"type":"input_image","image_url":"data:image/png;base64,AA=="}] trailing"#,
        ),
        (
            "truncated image array",
            r#"[{"type":"input_image","image_url":"data:image/png;base64,AA=="}"#,
        ),
        (
            "non-string image URL",
            r#"[{"type":"input_image","image_url":123}]"#,
        ),
        (
            "non-string image detail",
            r#"[{"type":"input_image","image_url":"data:image/png;base64,AA==","detail":123}]"#,
        ),
        (
            "non-string text in image array",
            r#"[{"type":"input_text","text":123},{"type":"input_image","image_url":"data:image/png;base64,AA=="}]"#,
        ),
    ];
    for (name, output) in cases {
        let raw = format!(
            r#"{{
                "input": [
                    {{"type":"function_call","call_id":"call_output","name":"inspect","arguments":"{{}}"}},
                    {{"type":"function_call_output","call_id":"call_output","output":{output:?}}}
                ]
            }}"#
        );
        let out = convert_openai_responses_request_to_openai_chat_completions(
            "k3",
            raw.as_bytes(),
            false,
        );
        let value = value(out);
        let content = &value["messages"][1]["content"];
        assert!(
            content.is_string(),
            "{name}: expected string, got {content}"
        );
        assert_eq!(content.as_str().unwrap(), output, "{name}");
    }
}

#[test]
fn attaches_reasoning_to_assistant_message() {
    let raw = br#"{
        "input": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [
                    {"type": "summary_text", "text": "first line\n"},
                    {"type": "summary_text", "text": "second line"}
                ]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}]
            },
            {"type": "message", "role": "user", "content": "next"}
        ]
    }"#;
    let out = convert_openai_responses_request_to_openai_chat_completions(
        "deepseek-v4-flash",
        raw,
        false,
    );
    let value = value(out);
    assert_eq!(value["messages"].as_array().unwrap().len(), 2);
    assert_eq!(value["messages"][0]["role"], "assistant");
    assert_eq!(
        value["messages"][0]["reasoning_content"],
        "first line\nsecond line"
    );
    assert_eq!(value["messages"][0]["content"][0]["text"], "answer");
    assert_eq!(value["messages"][1]["role"], "user");
}

#[test]
fn attaches_reasoning_to_tool_call_message() {
    let raw = br#"{
        "input": [
            {"type": "reasoning", "id": "rs_tool", "summary": [{"type": "summary_text", "text": "tool reasoning"}]},
            {"type":"function_call","call_id":"call_1","name":"exec_command","arguments":"{\"cmd\":\"pwd\"}"},
            {"type":"function_call_output","call_id":"call_1","output":"ok"}
        ]
    }"#;
    let out =
        convert_openai_responses_request_to_openai_chat_completions("deepseek-v4-flash", raw, true);
    let value = value(out);
    assert_eq!(value["messages"].as_array().unwrap().len(), 2);
    assert_eq!(value["messages"][0]["role"], "assistant");
    assert_eq!(value["messages"][0]["reasoning_content"], "tool reasoning");
    assert_eq!(value["messages"][0]["tool_calls"][0]["id"], "call_1");
    assert_eq!(value["messages"][1]["role"], "tool");
}

#[test]
fn keeps_reasoning_before_user_message() {
    let raw = br#"{
        "input": [
            {"type": "reasoning", "id": "rs_empty", "summary": []},
            {"type": "message", "role": "user", "content": "continue"}
        ]
    }"#;
    let out = convert_openai_responses_request_to_openai_chat_completions(
        "deepseek-v4-flash",
        raw,
        false,
    );
    let value = value(out);
    assert_eq!(value["messages"].as_array().unwrap().len(), 2);
    assert_eq!(value["messages"][0]["role"], "assistant");
    assert_eq!(
        value["messages"][0]["reasoning_content"],
        "[reasoning unavailable]"
    );
    assert_eq!(value["messages"][1]["role"], "user");
}

#[test]
fn flattens_namespace_tools() {
    let raw = br#"{
        "input": [
            {"role":"user","content":"Use add_numbers."}
        ],
        "tools": [
            {
                "type": "namespace",
                "name": "mcp__test_mcp__",
                "description": "Tools in the mcp__test_mcp__ namespace.",
                "tools": [
                    {
                        "type": "function",
                        "name": "add_numbers",
                        "description": "Add two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": { "type": "number" },
                                "b": { "type": "number" }
                            },
                            "required": ["a", "b"]
                        }
                    }
                ]
            }
        ],
        "tool_choice": "auto"
    }"#;
    let out = convert_openai_responses_request_to_openai_chat_completions(
        "deepseek-v4-flash",
        raw,
        false,
    );
    let value = value(out);
    assert_eq!(value["tools"].as_array().unwrap().len(), 1);
    assert_eq!(value["tools"][0]["type"], "function");
    assert_eq!(
        value["tools"][0]["function"]["name"],
        "mcp__test_mcp__add_numbers"
    );
    assert_eq!(
        value["tools"][0]["function"]["description"],
        "Add two numbers"
    );
    assert_eq!(
        value["tools"][0]["function"]["parameters"]["required"][0],
        "a"
    );
}

#[test]
fn qualifies_namespace_function_call_history() {
    let raw = br#"{
        "input": [
            {"type":"function_call","call_id":"call_get_me","name":"get_me","namespace":"mcp__github","arguments":"{}"},
            {"type":"function_call_output","call_id":"call_get_me","output":"ok"}
        ],
        "tools": [
            {
                "type":"namespace",
                "name":"mcp__github",
                "tools":[{"type":"function","name":"get_me","parameters":{"type":"object"}}]
            }
        ]
    }"#;
    let out = convert_openai_responses_request_to_openai_chat_completions(
        "deepseek-v4-flash",
        raw,
        false,
    );
    let value = value(out);
    let history = value["messages"][0]["tool_calls"][0]["function"]["name"]
        .as_str()
        .unwrap();
    let declared = value["tools"][0]["function"]["name"].as_str().unwrap();
    assert_eq!(history, "mcp__github__get_me");
    assert_eq!(history, declared);
}

#[test]
fn flattens_namespace_custom_tools() {
    let cases: [(&str, &[u8]); 2] = [
        (
            "top-level",
            br#"{"tools":[{"type":"namespace","name":"terminal","tools":[{"type":"custom","name":"exec","description":"Run a command"}]}]}"#,
        ),
        (
            "additional",
            br#"{"input":[{"type":"additional_tools","tools":[{"type":"namespace","name":"terminal","tools":[{"type":"custom","name":"exec","description":"Run a command"}]}]}]}"#,
        ),
    ];
    for (name, raw) in cases {
        let out =
            convert_openai_responses_request_to_openai_chat_completions("gpt-5.4", raw, false);
        let value = value(out);
        assert_eq!(value["tools"].as_array().unwrap().len(), 1, "{name}");
        assert_eq!(
            value["tools"][0]["function"]["name"], "terminal__exec",
            "{name}"
        );
        assert_eq!(
            value["tools"][0]["function"]["description"], "Run a command",
            "{name}"
        );
        assert_eq!(
            value["tools"][0]["function"]["parameters"]["type"], "object",
            "{name}"
        );
        assert_eq!(
            value["tools"][0]["function"]["parameters"]["properties"]["input"]["type"], "string",
            "{name}"
        );
        assert_eq!(
            value["tools"][0]["function"]["parameters"]["required"][0], "input",
            "{name}"
        );
    }
}

#[test]
fn preserves_structured_tool_choice() {
    let raw = br#"{
        "input": [{"role":"user","content":"Run command."}],
        "tools": [{"type": "function", "name": "run_command", "parameters": {"type": "object"}}],
        "tool_choice": {"type": "function", "function": {"name": "run_command"}}
    }"#;
    let out = convert_openai_responses_request_to_openai_chat_completions("gpt-5.4", raw, false);
    let value = value(out);
    assert_eq!(value["tool_choice"]["type"], "function");
    assert_eq!(value["tool_choice"]["function"]["name"], "run_command");
}

#[test]
fn omits_tool_settings_without_tools() {
    let cases: [(&str, &[u8]); 2] = [
        (
            "empty tools",
            br#"{"input": [{"role":"user","content":"say ok"}], "tools": [], "tool_choice": "auto", "parallel_tool_calls": false}"#,
        ),
        (
            "unconvertible tools",
            br#"{"tools": [{"type":"unsupported"}], "tool_choice": "auto", "parallel_tool_calls": false}"#,
        ),
    ];
    for (name, raw) in cases {
        let out =
            convert_openai_responses_request_to_openai_chat_completions("grok-4.5", raw, false);
        let value = value(out);
        for field in ["tools", "tool_choice", "parallel_tool_calls"] {
            assert!(
                value.get(field).is_none_or(Value::is_null),
                "{name}: {field} should be omitted"
            );
        }
    }
}

#[test]
fn preserves_parallel_tool_calls_with_tools() {
    let raw = br#"{
        "tools": [{"type": "function", "name": "run_command", "parameters": {"type": "object"}}],
        "parallel_tool_calls": false
    }"#;
    let out = convert_openai_responses_request_to_openai_chat_completions("grok-4.5", raw, false);
    let value = value(out);
    let parallel = &value["parallel_tool_calls"];
    assert!(parallel.is_boolean());
    assert_eq!(parallel.as_bool(), Some(false));
}

#[test]
fn normalizes_input_image_detail() {
    let cases: [(&str, &str, Option<&str>); 4] = [
        ("standard high", r#""high""#, Some("high")),
        ("Codex original", r#""original""#, Some("high")),
        ("unsupported value", r#""medium""#, None),
        ("non-string value", "123", None),
    ];
    for (name, detail_json, expected) in cases {
        let raw = format!(
            r#"{{
                "input": [
                    {{
                        "role": "user",
                        "content": [
                            {{
                                "type": "input_image",
                                "image_url": "https://example.com/image.png",
                                "detail": {detail_json}
                            }}
                        ]
                    }}
                ]
            }}"#
        );
        let out = convert_openai_responses_request_to_openai_chat_completions(
            "gpt-5.4",
            raw.as_bytes(),
            false,
        );
        let value = value(out);
        let url = value["messages"][0]["content"][0]["image_url"]["url"]
            .as_str()
            .unwrap();
        assert_eq!(url, "https://example.com/image.png", "{name}");
        let detail = &value["messages"][0]["content"][0]["image_url"]["detail"];
        if expected.is_none() {
            assert!(
                detail.is_null(),
                "{name}: detail should be omitted, got {detail}"
            );
        } else {
            assert_eq!(detail.as_str(), expected, "{name}");
        }
    }
}

#[test]
fn merges_includes_output_text_with_reasoning_unavailable_placeholder() {
    // sanity check: a reasoning item with no summary is replaced by a placeholder
    let raw = json!({
        "input":[
            {"type":"reasoning","id":"r1","summary":[]},
            {"type":"message","role":"user","content":"ping"}
        ]
    })
    .to_string();
    let out =
        convert_openai_responses_request_to_openai_chat_completions("k", raw.as_bytes(), false);
    let value = value(out);
    assert_eq!(
        value["messages"][0]["reasoning_content"],
        "[reasoning unavailable]"
    );
}
