// ref: internal/translator/openai/openai/responses/openai_openai-responses_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{
    convert_openai_chat_completions_response_to_openai_responses,
    convert_openai_chat_completions_response_to_openai_responses_non_stream,
};
use crate::sdk::translator::{TranslationContext, TranslationState};
use serde_json::Value;

fn parse_sse_event(raw: &[u8]) -> (String, Value) {
    let text = std::str::from_utf8(raw).expect("SSE bytes are utf-8");
    let mut event = String::new();
    let mut data = String::new();
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("event: ") {
            event = value.to_string();
        } else if let Some(value) = line.strip_prefix("data: ") {
            data.push_str(value);
        }
    }
    if data == "[DONE]" {
        (event, Value::String(data))
    } else {
        let value: Value = serde_json::from_str(&data).expect("data is valid JSON");
        (event, value)
    }
}

fn run_stream(request: &[u8], chunks: &[&[u8]]) -> Vec<(String, Value)> {
    let context = TranslationContext::default();
    let mut state: TranslationState = None;
    let mut events: Vec<(String, Value)> = Vec::new();
    for chunk in chunks {
        let out = convert_openai_chat_completions_response_to_openai_responses(
            &context, "model", request, request, chunk, &mut state,
        );
        for raw in out {
            events.push(parse_sse_event(&raw));
        }
    }
    events
}

#[test]
fn response_completed_waits_for_done_with_late_usage() {
    let request = br#"{"model":"gpt-5.4","tool_choice":"auto","parallel_tool_calls":true}"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"resp_late_usage","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"index":0,"id":"call_late_usage","type":"function","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_late_usage","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":[{"index":0,"function":{"arguments":"{\"filePath\":\"C:\\\\repo\\\\README.md\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        br#"data: {"id":"resp_late_usage","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#,
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    let completed: Vec<_> = events
        .iter()
        .filter(|(name, _)| name == "response.completed")
        .collect();
    assert_eq!(completed.len(), 1, "expected exactly 1 response.completed");
    let usage = &completed[0].1["response"]["usage"];
    assert_eq!(usage["input_tokens"], 11);
    assert_eq!(usage["output_tokens"], 7);
    assert_eq!(usage["total_tokens"], 18);
}

#[test]
fn response_completed_omits_usage_when_not_provided() {
    let request = br#"{"model":"gpt-5.4","tool_choice":"auto","parallel_tool_calls":true}"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"resp_no_usage","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"index":0,"id":"call_no_usage","type":"function","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_no_usage","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":[{"index":0,"function":{"arguments":"{\"filePath\":\"C:\\\\repo\\\\README.md\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    let completed: Vec<_> = events
        .iter()
        .filter(|(name, _)| name == "response.completed")
        .collect();
    assert_eq!(completed.len(), 1);
    assert!(completed[0].1["response"].get("usage").is_none());
}

#[test]
fn multiple_tool_calls_remain_separate() {
    let request = br#"{"model":"gpt-5.4","tool_choice":"auto","parallel_tool_calls":true}"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"resp_test","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"index":0,"id":"call_read","type":"function","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_test","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":[{"index":0,"function":{"arguments":"{\"filePath\":\"C:\\\\repo\",\"limit\":400,\"offset\":1}"}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_test","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"index":1,"id":"call_glob","type":"function","function":{"name":"glob","arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_test","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":[{"index":1,"function":{"arguments":"{\"path\":\"C:\\\\repo\",\"pattern\":\"*.{yml,yaml}\"}"}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_test","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":null},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":10,"total_tokens":20,"prompt_tokens":10}}"#,
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    let mut added_names: std::collections::HashMap<String, String> = Default::default();
    let mut done_args: std::collections::HashMap<String, String> = Default::default();
    let mut done_names: std::collections::HashMap<String, String> = Default::default();
    let mut output_items: std::collections::HashMap<String, Value> = Default::default();
    for (name, data) in &events {
        match name.as_str() {
            "response.output_item.added" => {
                if data["item"]["type"] == "function_call" {
                    added_names.insert(
                        data["item"]["call_id"].as_str().unwrap().to_string(),
                        data["item"]["name"].as_str().unwrap().to_string(),
                    );
                }
            }
            "response.output_item.done" => {
                if data["item"]["type"] == "function_call" {
                    let call_id = data["item"]["call_id"].as_str().unwrap().to_string();
                    done_args.insert(
                        call_id.clone(),
                        data["item"]["arguments"].as_str().unwrap().to_string(),
                    );
                    done_names.insert(call_id, data["item"]["name"].as_str().unwrap().to_string());
                }
            }
            "response.completed" => {
                for item in data["response"]["output"].as_array().unwrap_or(&Vec::new()) {
                    if item["type"] == "function_call" {
                        let call_id = item["call_id"].as_str().unwrap().to_string();
                        output_items.insert(call_id, item.clone());
                    }
                }
            }
            _ => {}
        }
    }
    assert_eq!(added_names.len(), 2);
    assert_eq!(done_args.len(), 2);
    assert_eq!(added_names["call_read"], "read");
    assert_eq!(added_names["call_glob"], "glob");
    assert!(!done_args["call_read"].contains("}{"));
    assert!(!done_args["call_glob"].contains("}{"));
    assert_eq!(done_names["call_read"], "read");
    assert_eq!(done_names["call_glob"], "glob");
    assert!(serde_json::from_str::<Value>(done_args["call_read"].as_str()).is_ok());
    assert!(serde_json::from_str::<Value>(done_args["call_glob"].as_str()).is_ok());
    let file_path = serde_json::from_str::<Value>(done_args["call_read"].as_str()).unwrap();
    assert_eq!(file_path["filePath"], "C:\\repo");
    let glob = serde_json::from_str::<Value>(done_args["call_glob"].as_str()).unwrap();
    assert_eq!(glob["path"], "C:\\repo");
    assert_eq!(glob["pattern"], "*.{yml,yaml}");
    assert_eq!(output_items.len(), 2);
    assert_eq!(output_items["call_read"]["name"], "read");
    assert_eq!(output_items["call_glob"]["name"], "glob");
}

#[test]
fn multi_choice_tool_calls_use_distinct_output_indexes() {
    let request = br#"{"model":"gpt-5.4","tool_choice":"auto","parallel_tool_calls":true}"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"resp_multi_choice","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"index":0,"id":"call_choice0","type":"function","function":{"name":"glob","arguments":""}}]},"finish_reason":null},{"index":1,"delta":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"index":0,"id":"call_choice1","type":"function","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_multi_choice","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":\"C:\\\\repo\",\"pattern\":\"*.go\"}"}}]},"finish_reason":null},{"index":1,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":[{"index":0,"function":{"arguments":"{\"filePath\":\"C:\\\\repo\\\\README.md\",\"limit\":20,\"offset\":1}"}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_multi_choice","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":null},"finish_reason":"tool_calls"},{"index":1,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":null},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":10,"total_tokens":20,"prompt_tokens":10}}"#,
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    #[derive(Default)]
    struct Call {
        output_index: i64,
        name: String,
        arguments: String,
    }
    let mut added: std::collections::HashMap<String, Call> = Default::default();
    let mut done: std::collections::HashMap<String, Call> = Default::default();
    for (name, data) in &events {
        if name == "response.output_item.added" && data["item"]["type"] == "function_call" {
            let call_id = data["item"]["call_id"].as_str().unwrap().to_string();
            added.insert(
                call_id,
                Call {
                    output_index: data["output_index"].as_i64().unwrap_or(0),
                    name: data["item"]["name"].as_str().unwrap_or("").to_string(),
                    arguments: String::new(),
                },
            );
        }
        if name == "response.output_item.done" && data["item"]["type"] == "function_call" {
            let call_id = data["item"]["call_id"].as_str().unwrap().to_string();
            done.insert(
                call_id,
                Call {
                    output_index: data["output_index"].as_i64().unwrap_or(0),
                    name: data["item"]["name"].as_str().unwrap_or("").to_string(),
                    arguments: data["item"]["arguments"].as_str().unwrap_or("").to_string(),
                },
            );
        }
    }
    assert_eq!(added.len(), 2);
    assert_eq!(done.len(), 2);
    assert_eq!(added["call_choice0"].name, "glob");
    assert_eq!(added["call_choice1"].name, "read");
    assert_ne!(
        added["call_choice0"].output_index,
        added["call_choice1"].output_index
    );
    assert!(serde_json::from_str::<Value>(done["call_choice0"].arguments.as_str()).is_ok());
    assert!(serde_json::from_str::<Value>(done["call_choice1"].arguments.as_str()).is_ok());
    assert_ne!(
        done["call_choice0"].output_index,
        done["call_choice1"].output_index
    );
    assert_eq!(done["call_choice0"].name, "glob");
    assert_eq!(done["call_choice1"].name, "read");
}

#[test]
fn mixed_message_and_tool_use_distinct_output_indexes() {
    let request = br#"{"model":"gpt-5.4","tool_choice":"auto","parallel_tool_calls":true}"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"resp_mixed","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","content":"hello","reasoning_content":null,"tool_calls":null},"finish_reason":null},{"index":1,"delta":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"index":0,"id":"call_choice1","type":"function","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_mixed","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":null},"finish_reason":"stop"},{"index":1,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":[{"index":0,"function":{"arguments":"{\"filePath\":\"C:\\\\repo\\\\README.md\",\"limit\":20,\"offset\":1}"}}]},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":10,"total_tokens":20,"prompt_tokens":10}}"#,
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    let mut message_index: i64 = -1;
    let mut tool_index: i64 = -1;
    for (name, data) in &events {
        if name != "response.output_item.added" {
            continue;
        }
        match data["item"]["type"].as_str() {
            Some("message") => {
                if data["item"]["id"] == "msg_resp_mixed_0" {
                    message_index = data["output_index"].as_i64().unwrap_or(-1);
                }
            }
            Some("function_call") => {
                if data["item"]["call_id"] == "call_choice1" {
                    tool_index = data["output_index"].as_i64().unwrap_or(-1);
                }
            }
            _ => {}
        }
    }
    assert!(message_index >= 0);
    assert!(tool_index >= 0);
    assert_ne!(message_index, tool_index);
}

#[test]
fn completed_omits_top_level_output_text() {
    let request = br#"{"model":"gpt-5.4"}"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"resp_output_text","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","content":"hello ","reasoning_content":null,"tool_calls":null},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_output_text","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":"world","reasoning_content":null,"tool_calls":null},"finish_reason":"stop"}],"usage":{"completion_tokens":2,"total_tokens":4,"prompt_tokens":2}}"#,
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    let completed: Option<&Value> = events
        .iter()
        .find(|(name, _)| name == "response.completed")
        .map(|(_, v)| v);
    assert!(completed.is_some());
    assert!(completed.unwrap()["response"].get("output_text").is_none());
    assert_eq!(
        completed.unwrap()["response"]["output"][0]["content"][0]["text"],
        "hello world"
    );
}

#[test]
fn tool_call_completed_omits_top_level_output_text() {
    let request = br#"{"model":"gpt-5.4","tool_choice":"auto","parallel_tool_calls":true}"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"resp_tool_output_text","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","content":"I will call the weather tool.","reasoning_content":null,"tool_calls":null},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_tool_output_text","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#,
        r#"data: {"id":"resp_tool_output_text","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":\"北京\",\"unit\":\"celsius\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":10,"total_tokens":20,"prompt_tokens":10}}"#.as_bytes(),
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    let completed: Option<&Value> = events
        .iter()
        .find(|(name, _)| name == "response.completed")
        .map(|(_, v)| v);
    assert!(completed.is_some());
    assert!(completed.unwrap()["response"].get("output_text").is_none());
    assert_eq!(
        completed.unwrap()["response"]["output"][0]["content"][0]["text"],
        "I will call the weather tool."
    );
    let args = completed.unwrap()["response"]["output"][1]["arguments"]
        .as_str()
        .unwrap();
    assert!(args.contains("北京"));
}

#[test]
fn function_call_done_and_completed_output_stay_ascending() {
    let request = br#"{"model":"gpt-5.4","tool_choice":"auto","parallel_tool_calls":true}"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"resp_order","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"index":0,"id":"call_glob","type":"function","function":{"name":"glob","arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_order","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":\"C:\\\\repo\",\"pattern\":\"*.go\"}"}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_order","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"index":1,"id":"call_read","type":"function","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_order","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":[{"index":1,"function":{"arguments":"{\"filePath\":\"C:\\\\repo\\\\README.md\",\"limit\":20,\"offset\":1}"}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"resp_order","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":null},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":10,"total_tokens":20,"prompt_tokens":10}}"#,
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    let mut done_indexes: Vec<i64> = Vec::new();
    let mut completed_order: Vec<String> = Vec::new();
    for (name, data) in &events {
        if name == "response.output_item.done" && data["item"]["type"] == "function_call" {
            done_indexes.push(data["output_index"].as_i64().unwrap_or(0));
        }
        if name == "response.completed" {
            for item in data["response"]["output"].as_array().unwrap_or(&Vec::new()) {
                if item["type"] == "function_call" {
                    completed_order.push(item["call_id"].as_str().unwrap().to_string());
                }
            }
        }
    }
    assert_eq!(done_indexes.len(), 2);
    assert!(done_indexes[0] < done_indexes[1]);
    assert_eq!(
        completed_order,
        vec!["call_glob".to_string(), "call_read".to_string()]
    );
}

#[test]
fn non_stream_omits_top_level_output_text() {
    let request = br#"{"model":"gpt-5.4"}"#;
    let raw = br#"{"id":"chatcmpl_output_text","object":"chat.completion","created":1773896263,"model":"model","choices":[{"index":0,"message":{"role":"assistant","content":"ping"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}"#;
    let mut state: TranslationState = None;
    let resp = convert_openai_chat_completions_response_to_openai_responses_non_stream(
        &TranslationContext::default(),
        "model",
        request,
        request,
        raw,
        &mut state,
    );
    let data: Value = serde_json::from_slice(&resp).unwrap();
    assert!(data.get("output_text").is_none());
    assert_eq!(data["output"][0]["content"][0]["text"], "ping");
}

#[test]
fn restores_namespace_function_call() {
    let request = br#"{
        "model":"deepseek-v4-flash",
        "tools":[
            {
                "type":"namespace",
                "name":"mcp__test_mcp__",
                "tools":[{"type":"function","name":"add_numbers","parameters":{"type":"object","properties":{}}}]
            }
        ]
    }"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"chatcmpl_namespace_stream","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_ns","type":"function","function":{"name":"mcp__test_mcp__add_numbers","arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"chatcmpl_namespace_stream","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":3,\"b\":5}"}}]},"finish_reason":"tool_calls"}]}"#,
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    let added: Option<&Value> = events
        .iter()
        .find(|(name, data)| {
            name == "response.output_item.added" && data["item"]["type"] == "function_call"
        })
        .map(|(_, v)| v);
    let done: Option<&Value> = events
        .iter()
        .find(|(name, data)| {
            name == "response.output_item.done" && data["item"]["type"] == "function_call"
        })
        .map(|(_, v)| v);
    let completed: Option<&Value> = events
        .iter()
        .find(|(name, _)| name == "response.completed")
        .map(|(_, v)| v);
    assert!(added.is_some());
    assert_eq!(added.unwrap()["item"]["name"], "add_numbers");
    assert_eq!(added.unwrap()["item"]["namespace"], "mcp__test_mcp__");
    assert!(done.is_some());
    assert_eq!(done.unwrap()["item"]["name"], "add_numbers");
    assert_eq!(done.unwrap()["item"]["namespace"], "mcp__test_mcp__");
    assert!(completed.is_some());
    assert_eq!(
        completed.unwrap()["response"]["output"][0]["name"],
        "add_numbers"
    );
    assert_eq!(
        completed.unwrap()["response"]["output"][0]["namespace"],
        "mcp__test_mcp__"
    );
}

#[test]
fn non_stream_restores_namespace_function_call() {
    let request = br#"{
        "model":"deepseek-v4-flash",
        "tools":[{"type":"namespace","name":"mcp__test_mcp__","tools":[{"type":"function","name":"add_numbers","parameters":{"type":"object","properties":{}}}]}]
    }"#;
    let raw = br#"{"id":"chatcmpl_namespace_nonstream","object":"chat.completion","created":1773896263,"model":"model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_ns","type":"function","function":{"name":"mcp__test_mcp__add_numbers","arguments":"{\"a\":3,\"b\":5}"}}]},"finish_reason":"tool_calls"}]}"#;
    let mut state: TranslationState = None;
    let resp = convert_openai_chat_completions_response_to_openai_responses_non_stream(
        &TranslationContext::default(),
        "model",
        request,
        b"",
        raw,
        &mut state,
    );
    let data: Value = serde_json::from_slice(&resp).unwrap();
    assert_eq!(data["output"][0]["name"], "add_numbers");
    assert_eq!(data["output"][0]["namespace"], "mcp__test_mcp__");
}

#[test]
fn custom_tool_name_arrives_late() {
    let request = br#"{
        "model":"gpt-5.4",
        "tools":[{"type":"custom","name":"exec"}]
    }"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"chatcmpl_custom_late_name","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_exec","type":"function","function":{"arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"chatcmpl_custom_late_name","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"exec","arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"chatcmpl_custom_late_name","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"input\":\"pwd\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    let added: Option<&Value> = events
        .iter()
        .find(|(name, data)| {
            name == "response.output_item.added" && data["item"]["call_id"] == "call_exec"
        })
        .map(|(_, v)| v);
    let input_done: Option<&Value> = events
        .iter()
        .find(|(name, _)| name == "response.custom_tool_call_input.done")
        .map(|(_, v)| v);
    let item_done: Option<&Value> = events
        .iter()
        .find(|(name, data)| {
            name == "response.output_item.done" && data["item"]["call_id"] == "call_exec"
        })
        .map(|(_, v)| v);
    let completed: Option<&Value> = events
        .iter()
        .find(|(name, _)| name == "response.completed")
        .map(|(_, v)| v);
    assert!(added.is_some());
    assert_eq!(added.unwrap()["item"]["type"], "custom_tool_call");
    assert_eq!(added.unwrap()["item"]["id"], "ctc_call_exec");
    assert_eq!(added.unwrap()["item"]["name"], "exec");
    assert!(item_done.is_some());
    assert_eq!(item_done.unwrap()["item"]["type"], "custom_tool_call");
    assert_eq!(item_done.unwrap()["item"]["id"], "ctc_call_exec");
    assert_eq!(item_done.unwrap()["item"]["name"], "exec");
    assert!(completed.is_some());
    assert_eq!(
        completed.unwrap()["response"]["output"][0]["type"],
        "custom_tool_call"
    );
    assert_eq!(
        completed.unwrap()["response"]["output"][0]["id"],
        "ctc_call_exec"
    );
    assert_eq!(completed.unwrap()["response"]["output"][0]["name"], "exec");
    assert!(input_done.is_some());
    assert_eq!(input_done.unwrap()["item_id"], "ctc_call_exec");
    assert_eq!(input_done.unwrap()["input"], "pwd");
}

#[test]
fn custom_tool_name_and_id_are_missing() {
    let request = br#"{"model":"gpt-5.4","tools":[{"type":"custom","name":"exec"}]}"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"chatcmpl_custom_missing_fields","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"arguments":"{\"input\":\"pwd\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    let want_call_id = "call_chatcmpl_custom_missing_fields_0_0";
    let added = events
        .iter()
        .find(|(name, _)| name == "response.output_item.added")
        .map(|(_, v)| v);
    let done = events
        .iter()
        .find(|(name, _)| name == "response.output_item.done")
        .map(|(_, v)| v);
    let completed = events
        .iter()
        .find(|(name, _)| name == "response.completed")
        .map(|(_, v)| v);
    assert_eq!(added.unwrap()["item"]["type"], "custom_tool_call");
    assert_eq!(added.unwrap()["item"]["id"], format!("ctc_{want_call_id}"));
    assert_eq!(added.unwrap()["item"]["call_id"], want_call_id);
    assert_eq!(added.unwrap()["item"]["name"], "exec");
    assert_eq!(done.unwrap()["item"]["type"], "custom_tool_call");
    assert_eq!(done.unwrap()["item"]["id"], format!("ctc_{want_call_id}"));
    assert_eq!(done.unwrap()["item"]["call_id"], want_call_id);
    assert_eq!(done.unwrap()["item"]["name"], "exec");
    assert_eq!(
        completed.unwrap()["response"]["output"][0]["type"],
        "custom_tool_call"
    );
    assert_eq!(
        completed.unwrap()["response"]["output"][0]["id"],
        format!("ctc_{want_call_id}")
    );
    assert_eq!(
        completed.unwrap()["response"]["output"][0]["call_id"],
        want_call_id
    );
    assert_eq!(completed.unwrap()["response"]["output"][0]["name"], "exec");
}

#[test]
fn tool_call_id_may_arrive_late_or_be_missing() {
    let cases = [
        (
            "late id",
            vec![
                br#"data: {"id":"chatcmpl_late_id","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"read","arguments":"{\"file"}}]},"finish_reason":null}]}"#.to_vec(),
                br#"data: {"id":"chatcmpl_late_id","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_late","function":{"arguments":"Path\":\"README.md\"}"}}]},"finish_reason":"tool_calls"}]}"#.to_vec(),
            ],
            "call_late",
        ),
        (
            "missing id",
            vec![br#"data: {"id":"chatcmpl_missing_id","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"read","arguments":"{\"filePath\":\"README.md\"}"}}]},"finish_reason":"tool_calls"}]}"#.to_vec()],
            "call_chatcmpl_missing_id_0_0",
        ),
    ];
    for (name, mut chunks, want_call_id) in cases {
        chunks.push(b"data: [DONE]".to_vec());
        let chunk_refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let events = run_stream(b"", &chunk_refs);
        let want_item_id = format!("fc_{want_call_id}");
        let added = events
            .iter()
            .find(|(name, _)| name == "response.output_item.added")
            .map(|(_, v)| v);
        let args_delta = events
            .iter()
            .find(|(name, _)| name == "response.function_call_arguments.delta")
            .map(|(_, v)| v);
        let args_done = events
            .iter()
            .find(|(name, _)| name == "response.function_call_arguments.done")
            .map(|(_, v)| v);
        let item_done = events
            .iter()
            .find(|(name, _)| name == "response.output_item.done")
            .map(|(_, v)| v);
        assert_eq!(added.unwrap()["item"]["id"], want_item_id, "{name}");
        assert_eq!(added.unwrap()["item"]["call_id"], want_call_id, "{name}");
        assert_eq!(args_delta.unwrap()["item_id"], want_item_id, "{name}");
        assert_eq!(
            args_delta.unwrap()["delta"],
            r#"{"filePath":"README.md"}"#,
            "{name}"
        );
        assert_eq!(args_done.unwrap()["item_id"], want_item_id, "{name}");
        assert_eq!(item_done.unwrap()["item"]["id"], want_item_id, "{name}");
    }
}

#[test]
fn restores_additional_namespace_function_call() {
    let request = br#"{
        "model":"gpt-5.4",
        "input":[{
            "type":"additional_tools",
            "tools":[{
                "type":"namespace",
                "name":"collaboration",
                "tools":[{"type":"function","name":"send_message","parameters":{"type":"object","properties":{}}}]
            }]
        }]
    }"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"chatcmpl_additional_namespace_stream","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_send","type":"function","function":{"name":"collaboration__send_message","arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"chatcmpl_additional_namespace_stream","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"target\":\"worker\",\"message\":\"ping\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    let added = events
        .iter()
        .find(|(n, _)| n == "response.output_item.added")
        .map(|(_, v)| v);
    let done = events
        .iter()
        .find(|(n, _)| n == "response.output_item.done")
        .map(|(_, v)| v);
    let completed = events
        .iter()
        .find(|(n, _)| n == "response.completed")
        .map(|(_, v)| v);
    assert_eq!(added.unwrap()["item"]["name"], "send_message");
    assert_eq!(added.unwrap()["item"]["namespace"], "collaboration");
    assert_eq!(done.unwrap()["item"]["name"], "send_message");
    assert_eq!(done.unwrap()["item"]["namespace"], "collaboration");
    assert_eq!(
        completed.unwrap()["response"]["output"][0]["name"],
        "send_message"
    );
    assert_eq!(
        completed.unwrap()["response"]["output"][0]["namespace"],
        "collaboration"
    );
}

#[test]
fn restores_additional_namespace_custom_tool_call() {
    let request = br#"{
        "model":"gpt-5.4",
        "input":[{
            "type":"additional_tools",
            "tools":[{
                "type":"namespace",
                "name":"terminal",
                "tools":[{"type":"custom","name":"exec"}]
            }]
        }]
    }"#;
    let chunks: &[&[u8]] = &[
        br#"data: {"id":"chatcmpl_additional_namespace_custom_stream","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_exec","type":"function","function":{"name":"terminal__exec","arguments":""}}]},"finish_reason":null}]}"#,
        br#"data: {"id":"chatcmpl_additional_namespace_custom_stream","object":"chat.completion.chunk","created":1773896263,"model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"input\":\"pwd\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        b"data: [DONE]",
    ];
    let events = run_stream(request, chunks);
    let added = events
        .iter()
        .find(|(n, _)| n == "response.output_item.added")
        .map(|(_, v)| v);
    let done = events
        .iter()
        .find(|(n, _)| n == "response.output_item.done")
        .map(|(_, v)| v);
    let completed = events
        .iter()
        .find(|(n, _)| n == "response.completed")
        .map(|(_, v)| v);
    let input_done = events
        .iter()
        .find(|(n, _)| n == "response.custom_tool_call_input.done")
        .map(|(_, v)| v);
    for label in ["added", "done", "completed"] {
        let path = match label {
            "added" | "done" => "/item",
            _ => "/response/output/0",
        };
        let value = match label {
            "added" => added.unwrap(),
            "done" => done.unwrap(),
            _ => completed.unwrap(),
        };
        let item = value.pointer(path).unwrap();
        assert_eq!(item["type"], "custom_tool_call", "{label}");
        assert_eq!(item["name"], "terminal__exec", "{label}");
    }
    assert_eq!(input_done.unwrap()["input"], "pwd");
    assert_eq!(done.unwrap()["item"]["input"], "pwd");
    assert_eq!(completed.unwrap()["response"]["output"][0]["input"], "pwd");
}

#[test]
fn non_stream_restores_additional_namespace_custom_tool_call() {
    let request = br#"{
        "model":"gpt-5.4",
        "input":[{
            "type":"additional_tools",
            "tools":[{"type":"namespace","name":"terminal","tools":[{"type":"custom","name":"exec"}]}]
        }]
    }"#;
    let raw = br#"{"id":"chatcmpl_additional_namespace_custom_nonstream","object":"chat.completion","created":1773896263,"model":"model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_exec","type":"function","function":{"name":"terminal__exec","arguments":"{\"input\":\"pwd\"}"}}]},"finish_reason":"tool_calls"}]}"#;
    let mut state: TranslationState = None;
    let resp = convert_openai_chat_completions_response_to_openai_responses_non_stream(
        &TranslationContext::default(),
        "model",
        request,
        b"",
        raw,
        &mut state,
    );
    let data: Value = serde_json::from_slice(&resp).unwrap();
    assert_eq!(data["output"][0]["type"], "custom_tool_call");
    assert_eq!(data["output"][0]["name"], "terminal__exec");
    assert_eq!(data["output"][0]["input"], "pwd");
}
