// ref: internal/translator/claude/gemini/claude_gemini_response.go:1-600 @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::internal::translator::common::{gemini_token_count_json, SseDecoder};

#[derive(Default)]
pub struct ClaudeToGeminiState {
    decoder: SseDecoder,
    tools: HashMap<u64, ToolBuffer>,
    response_id: String,
    model: String,
}

#[derive(Default)]
struct ToolBuffer {
    id: String,
    name: String,
    arguments: String,
}

pub fn convert_claude_response_to_gemini(
    model: &str,
    _original: &[u8],
    _request: &[u8],
    raw: &[u8],
    state: &mut ClaudeToGeminiState,
) -> Vec<Vec<u8>> {
    let mut output = Vec::new();
    for event in state.decoder.push(raw) {
        process_event(model, &event.data, state, &mut output);
    }
    output
}

fn process_event(
    model: &str,
    event: &[u8],
    state: &mut ClaudeToGeminiState,
    output: &mut Vec<Vec<u8>>,
) {
    let Ok(root) = serde_json::from_slice::<Value>(event) else {
        return;
    };
    let index = root.get("index").and_then(Value::as_u64).unwrap_or(0);
    match root.get("type").and_then(Value::as_str).unwrap_or_default() {
        "message_start" => {
            let message = &root["message"];
            state.response_id = message
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            state.model = message
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(model)
                .to_owned();
        }
        "content_block_start" => {
            let block = &root["content_block"];
            if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                state.tools.insert(
                    index,
                    ToolBuffer {
                        id: string(block, "id"),
                        name: string(block, "name"),
                        arguments: block
                            .get("input")
                            .filter(|input| input.as_object().is_some_and(|map| !map.is_empty()))
                            .map(Value::to_string)
                            .unwrap_or_default(),
                    },
                );
            }
        }
        "content_block_delta" => {
            let delta = &root["delta"];
            match delta.get("type").and_then(Value::as_str).unwrap_or_default() {
                "input_json_delta" => state
                    .tools
                    .entry(index)
                    .or_default()
                    .arguments
                    .push_str(
                        delta
                            .get("partial_json")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    ),
                "text_delta" => {
                    if let Some(text) = delta
                        .get("text")
                        .and_then(Value::as_str)
                        .filter(|text| !text.is_empty())
                    {
                        output.push(candidate(state, model, Some(json!({"text":text}))));
                    }
                }
                "thinking_delta" => {
                    if let Some(text) = delta
                        .get("thinking")
                        .and_then(Value::as_str)
                        .filter(|text| !text.is_empty())
                    {
                        output.push(candidate(
                            state,
                            model,
                            Some(json!({"thought":true,"text":text})),
                        ));
                    }
                }
                _ => {}
            }
        }
        "content_block_stop" => {
            if let Some(tool) = state.tools.remove(&index) {
                let args = serde_json::from_str::<Value>(tool.arguments.trim())
                    .unwrap_or_else(|_| json!({}));
                output.push(candidate(
                    state,
                    model,
                    Some(json!({"functionCall":{"id":tool.id,"name":tool.name,"args":args}})),
                ));
            }
        }
        "message_delta" => {
            let mut value = base_response(state, model, Vec::new());
            value["candidates"][0]["finishReason"] = Value::String(
                if root.pointer("/delta/stop_reason").and_then(Value::as_str)
                    == Some("max_tokens")
                {
                    "MAX_TOKENS"
                } else {
                    "STOP"
                }
                .to_owned(),
            );
            if let Some(usage) = root.get("usage") {
                value["usageMetadata"] = usage_metadata(usage);
            }
            output.push(serde_json::to_vec(&value).unwrap_or_default());
        }
        "error" => output.push(
            serde_json::to_vec(&json!({"error":{"code":400,
                "message":root.pointer("/error/message").and_then(Value::as_str).unwrap_or("Unknown error occurred"),
                "status":"INVALID_ARGUMENT"}}))
            .unwrap_or_default(),
        ),
        _ => {}
    }
}

fn candidate(state: &ClaudeToGeminiState, model: &str, part: Option<Value>) -> Vec<u8> {
    serde_json::to_vec(&base_response(state, model, part.into_iter().collect())).unwrap_or_default()
}

fn base_response(state: &ClaudeToGeminiState, model: &str, parts: Vec<Value>) -> Value {
    json!({
        "candidates":[{"content":{"role":"model","parts":parts}}],
        "usageMetadata":{"trafficType":"PROVISIONED_THROUGHPUT"},
        "modelVersion":if state.model.is_empty() {model} else {&state.model},
        "createTime":"1970-01-01T00:00:00Z",
        "responseId":state.response_id,
    })
}

fn usage_metadata(usage: &Value) -> Value {
    let input = integer(usage, "input_tokens");
    let output = integer(usage, "output_tokens");
    let mut value = json!({
        "promptTokenCount":input,"candidatesTokenCount":output,
        "totalTokenCount":input+output,"trafficType":"PROVISIONED_THROUGHPUT"
    });
    let cached =
        integer(usage, "cache_creation_input_tokens") + integer(usage, "cache_read_input_tokens");
    if cached != 0 {
        value["cachedContentTokenCount"] = Value::from(cached);
    }
    if let Some(thinking) = usage.get("thinking_tokens").and_then(Value::as_i64) {
        value["thoughtsTokenCount"] = Value::from(thinking);
    }
    value
}

pub fn convert_claude_response_to_gemini_non_stream(
    model: &str,
    _original: &[u8],
    _request: &[u8],
    raw: &[u8],
) -> Vec<u8> {
    if let Ok(root) = serde_json::from_slice::<Value>(raw) {
        let parts = root
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|block| match block.get("type").and_then(Value::as_str) {
                Some("text") => Some(json!({"text":string(block,"text")})),
                Some("thinking") => Some(json!({"thought":true,"text":string(block,"thinking")})),
                Some("tool_use") => Some(json!({"functionCall":{
                    "id":string(block,"id"),"name":string(block,"name"),
                    "args":block.get("input").cloned().unwrap_or_else(|| json!({}))
                }})),
                _ => None,
            })
            .collect::<Vec<_>>();
        let state = ClaudeToGeminiState {
            response_id: string(&root, "id"),
            model: root
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(model)
                .to_owned(),
            ..Default::default()
        };
        let mut output = base_response(&state, model, parts);
        output["candidates"][0]["finishReason"] = Value::String("STOP".to_owned());
        if let Some(usage) = root.get("usage") {
            output["usageMetadata"] = usage_metadata(usage);
        }
        return serde_json::to_vec(&output).unwrap_or_default();
    }

    let mut state = ClaudeToGeminiState::default();
    let mut chunks = Vec::new();
    for line in raw.split(|byte| *byte == b'\n') {
        let line = trim_ascii(line);
        let Some(payload) = line.strip_prefix(b"data:").map(trim_ascii) else {
            continue;
        };
        process_event(model, payload, &mut state, &mut chunks);
    }
    let mut parts = Vec::new();
    let mut usage = json!({"trafficType":"PROVISIONED_THROUGHPUT"});
    let mut finish = "STOP";
    for chunk in &chunks {
        let Ok(value) = serde_json::from_slice::<Value>(chunk) else {
            continue;
        };
        if let Some(chunk_parts) = value
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
        {
            parts.extend(chunk_parts.iter().cloned());
        }
        if let Some(metadata) = value.get("usageMetadata") {
            usage = metadata.clone();
        }
        if value
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)
            == Some("MAX_TOKENS")
        {
            finish = "MAX_TOKENS";
        }
    }
    let mut output = base_response(&state, model, consolidate_parts(parts));
    output["candidates"][0]["finishReason"] = Value::String(finish.to_owned());
    output["usageMetadata"] = usage;
    serde_json::to_vec(&output).unwrap_or_default()
}

fn consolidate_parts(parts: Vec<Value>) -> Vec<Value> {
    let mut output: Vec<Value> = Vec::new();
    for part in parts {
        let thought = part.get("thought").and_then(Value::as_bool) == Some(true);
        let Some(text) = part.get("text").and_then(Value::as_str) else {
            output.push(part);
            continue;
        };
        if let Some(previous) = output.last_mut().filter(|previous| {
            previous.get("text").is_some()
                && (previous.get("thought").and_then(Value::as_bool) == Some(true)) == thought
        }) {
            let joined = format!("{}{}", previous["text"].as_str().unwrap_or_default(), text);
            previous["text"] = Value::String(joined);
        } else {
            output.push(part);
        }
    }
    output
}

fn string(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn integer(value: &Value, key: &str) -> i64 {
    value.get(key).and_then(Value::as_i64).unwrap_or(0)
}

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

pub fn gemini_token_count(count: i64) -> Vec<u8> {
    gemini_token_count_json(count)
}
