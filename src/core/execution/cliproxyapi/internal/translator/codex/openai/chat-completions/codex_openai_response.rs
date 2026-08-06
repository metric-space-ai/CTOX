// ref: internal/translator/codex/openai/chat-completions/codex_openai_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::codex_openai_request::reverse_short_name_map;

#[derive(Default)]
struct ToolCallState {
    index: usize,
    arguments_emitted: bool,
    done: bool,
}

#[derive(Default)]
pub struct CodexToChatStreamState {
    response_id: String,
    created_at: i64,
    model: String,
    tool_calls: Vec<ToolCallState>,
    tool_call_keys: HashMap<String, usize>,
    current_tool_call: Option<usize>,
    last_image_hash_by_item_id: HashMap<String, [u8; 32]>,
}

pub fn convert_codex_response_to_openai_chat_stream(
    model_name: &str,
    original_request: &[u8],
    _request: &[u8],
    raw: &[u8],
    state: &mut CodexToChatStreamState,
) -> Vec<Vec<u8>> {
    if !raw.starts_with(b"data:") {
        return Vec::new();
    }
    let Ok(root) = serde_json::from_slice::<Value>(trim_ascii(&raw[5..])) else {
        return Vec::new();
    };
    let event_type = root.get("type").and_then(Value::as_str).unwrap_or("");
    if event_type == "response.created" {
        state.response_id = string(&root, "/response/id");
        state.created_at = integer(&root, "/response/created_at");
        state.model = string(&root, "/response/model");
        return Vec::new();
    }

    let model = root
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .or_else(|| (!state.model.is_empty()).then_some(state.model.as_str()))
        .or_else(|| (!model_name.is_empty()).then_some(model_name))
        .unwrap_or("model");
    let mut output = json!({
        "id":state.response_id,
        "object":"chat.completion.chunk",
        "created":state.created_at,
        "model":model,
        "choices":[{"index":0,"delta":{},"finish_reason":null,"native_finish_reason":null}]
    });
    if let Some(usage) = root.pointer("/response/usage") {
        output["usage"] = usage_value(usage);
    }

    match event_type {
        "response.reasoning_summary_text.delta" => {
            let Some(delta) = root.get("delta") else {
                return Vec::new();
            };
            output["choices"][0]["delta"] =
                json!({"role":"assistant","reasoning_content":value_string(delta)});
        }
        "response.reasoning_summary_text.done" => {
            output["choices"][0]["delta"] = json!({"role":"assistant","reasoning_content":"\n\n"});
        }
        "response.output_text.delta" => {
            let Some(delta) = root.get("delta") else {
                return Vec::new();
            };
            output["choices"][0]["delta"] =
                json!({"role":"assistant","content":value_string(delta)});
        }
        "response.image_generation_call.partial_image" => {
            let item_id = string(&root, "/item_id");
            let image = string(&root, "/partial_image_b64");
            if image.is_empty() || duplicate_image(state, &item_id, &image) {
                return Vec::new();
            }
            output["choices"][0]["delta"] = json!({"role":"assistant","images":[image_value(&image, &string(&root, "/output_format"), 0)]});
        }
        "response.completed" | "response.incomplete" => {
            let (finish, native) = if event_type == "response.incomplete" {
                let native = string(&root, "/response/incomplete_details/reason");
                let finish = match native.as_str() {
                    "max_tokens" | "max_output_tokens" => "length",
                    "content_filter" => "content_filter",
                    _ => "stop",
                };
                (finish, native)
            } else if !state.tool_calls.is_empty() {
                ("tool_calls", "tool_calls".into())
            } else {
                ("stop", "stop".into())
            };
            output["choices"][0]["finish_reason"] = Value::String(finish.into());
            output["choices"][0]["native_finish_reason"] = Value::String(native);
        }
        "response.output_item.added" => {
            let Some(item) = root.get("item").filter(|item| is_tool(item)) else {
                return Vec::new();
            };
            let index = register_tool(state, &root, item);
            let name = restore_name(
                original_request,
                item.get("name").and_then(Value::as_str).unwrap_or(""),
            );
            output["choices"][0]["delta"] = json!({"role":"assistant","tool_calls":[{
                "index":index,"id":item.get("call_id").and_then(Value::as_str).unwrap_or(""),"type":"function",
                "function":{"name":name,"arguments":""}
            }]});
        }
        "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
            let Some(index) = find_tool(state, &root, None) else {
                return Vec::new();
            };
            let delta = string(&root, "/delta");
            let tool = &mut state.tool_calls[index];
            if tool.done || delta.is_empty() {
                return Vec::new();
            }
            tool.arguments_emitted = true;
            output["choices"][0]["delta"] =
                json!({"tool_calls":[{"index":tool.index,"function":{"arguments":delta}}]});
        }
        "response.function_call_arguments.done" | "response.custom_tool_call_input.done" => {
            let Some(index) = find_tool(state, &root, None) else {
                return Vec::new();
            };
            let tool = &mut state.tool_calls[index];
            if tool.done || tool.arguments_emitted {
                return Vec::new();
            }
            tool.arguments_emitted = true;
            let field = if event_type == "response.custom_tool_call_input.done" {
                "/input"
            } else {
                "/arguments"
            };
            let arguments = string(&root, field);
            if arguments.is_empty() {
                return Vec::new();
            }
            output["choices"][0]["delta"] =
                json!({"tool_calls":[{"index":tool.index,"function":{"arguments":arguments}}]});
        }
        "response.output_item.done" => {
            let Some(item) = root.get("item") else {
                return Vec::new();
            };
            if item.get("type").and_then(Value::as_str) == Some("image_generation_call") {
                let image = item.get("result").and_then(Value::as_str).unwrap_or("");
                let item_id = item.get("id").and_then(Value::as_str).unwrap_or("");
                if image.is_empty() || duplicate_image(state, item_id, image) {
                    return Vec::new();
                }
                output["choices"][0]["delta"] = json!({"role":"assistant","images":[image_value(image, item.get("output_format").and_then(Value::as_str).unwrap_or(""), 0)]});
            } else if !is_tool(item) {
                return Vec::new();
            } else if let Some(index) = find_tool(state, &root, Some(item)) {
                let tool = &mut state.tool_calls[index];
                if tool.done {
                    return Vec::new();
                }
                tool.done = true;
                if tool.arguments_emitted {
                    return Vec::new();
                }
                tool.arguments_emitted = true;
                let arguments = tool_arguments(item);
                if arguments.is_empty() {
                    return Vec::new();
                }
                output["choices"][0]["delta"] =
                    json!({"tool_calls":[{"index":tool.index,"function":{"arguments":arguments}}]});
            } else {
                let index = register_tool(state, &root, item);
                let tool = &mut state.tool_calls[index];
                tool.arguments_emitted = true;
                tool.done = true;
                let name = restore_name(
                    original_request,
                    item.get("name").and_then(Value::as_str).unwrap_or(""),
                );
                output["choices"][0]["delta"] = json!({"role":"assistant","tool_calls":[{
                    "index":tool.index,"id":item.get("call_id").and_then(Value::as_str).unwrap_or(""),"type":"function",
                    "function":{"name":name,"arguments":tool_arguments(item)}
                }]});
            }
        }
        _ => return Vec::new(),
    }
    encode(output)
}

pub fn convert_codex_response_to_openai_chat_non_stream(
    original_request: &[u8],
    _request: &[u8],
    raw: &[u8],
) -> Vec<u8> {
    let Ok(root) = serde_json::from_slice::<Value>(raw) else {
        return Vec::new();
    };
    let event_type = root.get("type").and_then(Value::as_str).unwrap_or("");
    if event_type != "response.completed" && event_type != "response.incomplete" {
        return Vec::new();
    }
    let response = root.get("response").cloned().unwrap_or(Value::Null);
    let created = response
        .get("created_at")
        .and_then(Value::as_i64)
        .unwrap_or_else(now);
    let mut output = json!({
        "id":response.get("id").and_then(Value::as_str).unwrap_or(""),
        "object":"chat.completion","created":created,
        "model":response.get("model").and_then(Value::as_str).unwrap_or("model"),
        "choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":null},"finish_reason":null,"native_finish_reason":null}]
    });
    if let Some(usage) = response.get("usage") {
        output["usage"] = usage_value(usage);
    }
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tools = Vec::new();
    let mut images = Vec::new();
    for item in response
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match item.get("type").and_then(Value::as_str).unwrap_or("") {
            "reasoning" => if let Some(part) = item.get("summary").and_then(Value::as_array).into_iter().flatten().find(|part| part.get("type").and_then(Value::as_str) == Some("summary_text")) {
                reasoning.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
            },
            "message" => if let Some(part) = item.get("content").and_then(Value::as_array).into_iter().flatten().find(|part| part.get("type").and_then(Value::as_str) == Some("output_text")) {
                text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
            },
            "function_call" | "custom_tool_call" => tools.push(json!({
                "id":item.get("call_id").and_then(Value::as_str).unwrap_or(""),"type":"function",
                "function":{"name":restore_name(original_request, item.get("name").and_then(Value::as_str).unwrap_or("")),"arguments":tool_arguments(item)}
            })),
            "image_generation_call" => {
                let image = item.get("result").and_then(Value::as_str).unwrap_or("");
                if !image.is_empty() { images.push(image_value(image, item.get("output_format").and_then(Value::as_str).unwrap_or(""), images.len())); }
            }
            _ => {}
        }
    }
    if !text.is_empty() {
        output["choices"][0]["message"]["content"] = Value::String(text);
    }
    if !reasoning.is_empty() {
        output["choices"][0]["message"]["reasoning_content"] = Value::String(reasoning);
    }
    if !tools.is_empty() {
        output["choices"][0]["message"]["tool_calls"] = Value::Array(tools.clone());
    }
    if !images.is_empty() {
        output["choices"][0]["message"]["images"] = Value::Array(images);
    }
    if let Some(status) = response.get("status").and_then(Value::as_str) {
        let (finish, native) = if status == "completed" {
            if tools.is_empty() {
                ("stop", "stop".to_owned())
            } else {
                ("tool_calls", "tool_calls".to_owned())
            }
        } else if status == "incomplete" {
            let native = string(&response, "/incomplete_details/reason");
            let finish = match native.as_str() {
                "max_tokens" | "max_output_tokens" => "length",
                "content_filter" => "content_filter",
                _ => "stop",
            };
            (finish, native)
        } else {
            ("", String::new())
        };
        if !finish.is_empty() {
            output["choices"][0]["finish_reason"] = Value::String(finish.into());
            output["choices"][0]["native_finish_reason"] = Value::String(native);
        }
    }
    serde_json::to_vec(&output).unwrap_or_default()
}

fn register_tool(state: &mut CodexToChatStreamState, event: &Value, item: &Value) -> usize {
    let index = state.tool_calls.len();
    state.tool_calls.push(ToolCallState {
        index,
        ..Default::default()
    });
    for key in tool_keys(event, Some(item)) {
        state.tool_call_keys.insert(key, index);
    }
    state.current_tool_call = Some(index);
    index
}

fn find_tool(state: &CodexToChatStreamState, event: &Value, item: Option<&Value>) -> Option<usize> {
    if let Some(id) = event
        .get("item_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        return state.tool_call_keys.get(&format!("item:{id}")).copied();
    }
    if let Some(id) = item
        .and_then(|item| item.get("id"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        return state.tool_call_keys.get(&format!("item:{id}")).copied();
    }
    if let Some(index) = event.get("output_index") {
        return state
            .tool_call_keys
            .get(&format!("output:{}", value_string(index)))
            .copied();
    }
    state.current_tool_call
}

fn tool_keys(event: &Value, item: Option<&Value>) -> Vec<String> {
    let mut keys = Vec::new();
    if let Some(id) = event
        .get("item_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        keys.push(format!("item:{id}"));
    }
    if let Some(id) = item
        .and_then(|item| item.get("id"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        keys.push(format!("item:{id}"));
    }
    if let Some(index) = event.get("output_index") {
        keys.push(format!("output:{}", value_string(index)));
    }
    keys
}

fn is_tool(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some("function_call" | "custom_tool_call")
    )
}

fn tool_arguments(item: &Value) -> String {
    let key = if item.get("type").and_then(Value::as_str) == Some("custom_tool_call") {
        "input"
    } else {
        "arguments"
    };
    item.get(key).and_then(Value::as_str).unwrap_or("").into()
}

fn restore_name(original_request: &[u8], name: &str) -> String {
    reverse_short_name_map(original_request)
        .remove(name)
        .unwrap_or_else(|| name.into())
}

fn duplicate_image(state: &mut CodexToChatStreamState, item_id: &str, image: &str) -> bool {
    if item_id.is_empty() {
        return false;
    }
    let hash: [u8; 32] = Sha256::digest(image.as_bytes()).into();
    if state.last_image_hash_by_item_id.get(item_id) == Some(&hash) {
        return true;
    }
    state
        .last_image_hash_by_item_id
        .insert(item_id.into(), hash);
    false
}

fn image_value(image: &str, format: &str, index: usize) -> Value {
    let normalized = format.to_ascii_lowercase();
    let mime = match normalized.as_str() {
        "" | "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        value if value.contains('/') => value,
        _ => "image/png",
    };
    json!({"type":"image_url","index":index,"image_url":{"url":format!("data:{mime};base64,{image}")}})
}

fn usage_value(usage: &Value) -> Value {
    let mut output = json!({});
    for (source, target) in [
        ("output_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
        ("input_tokens", "prompt_tokens"),
    ] {
        if let Some(value) = usage.get(source).and_then(Value::as_i64) {
            output[target] = Value::from(value);
        }
    }
    if let Some(value) = usage
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(Value::as_i64)
    {
        output["prompt_tokens_details"]["cached_tokens"] = Value::from(value);
    }
    if let Some(value) = usage
        .pointer("/input_tokens_details/cache_write_tokens")
        .and_then(Value::as_i64)
    {
        output["prompt_tokens_details"]["cached_creation_tokens"] = Value::from(value);
    }
    if let Some(value) = usage
        .pointer("/output_tokens_details/reasoning_tokens")
        .and_then(Value::as_i64)
    {
        output["completion_tokens_details"]["reasoning_tokens"] = Value::from(value);
    }
    output
}

fn string(root: &Value, pointer: &str) -> String {
    root.pointer(pointer).map(value_string).unwrap_or_default()
}
fn integer(root: &Value, pointer: &str) -> i64 {
    root.pointer(pointer).and_then(Value::as_i64).unwrap_or(0)
}
fn value_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        value => serde_json::to_string(value).unwrap_or_default(),
    }
}
fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs() as i64)
        .unwrap_or(0)
}
fn encode(value: Value) -> Vec<Vec<u8>> {
    serde_json::to_vec(&value).map_or_else(|_| Vec::new(), |value| vec![value])
}
fn trim_ascii(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}

#[cfg(test)]
mod tests {
    use super::{
        convert_codex_response_to_openai_chat_non_stream,
        convert_codex_response_to_openai_chat_stream, CodexToChatStreamState,
    };
    use serde_json::Value;

    #[test]
    fn non_stream_combines_reasoning_text_tools_images_and_usage() {
        let raw = br#"{"type":"response.completed","response":{"id":"r","model":"gpt","created_at":1,"status":"completed","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5},"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"why"}]},{"type":"message","content":[{"type":"output_text","text":"answer"}]},{"type":"function_call","call_id":"c","name":"run","arguments":"{}"},{"type":"image_generation_call","result":"AAAA","output_format":"png"}]}}"#;
        let output: Value = serde_json::from_slice(
            &convert_codex_response_to_openai_chat_non_stream(b"{}", b"", raw),
        )
        .unwrap();
        assert_eq!(output["choices"][0]["message"]["reasoning_content"], "why");
        assert_eq!(output["choices"][0]["message"]["content"], "answer");
        assert_eq!(output["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(output["choices"][0]["message"]["images"][0]["index"], 0);
        assert_eq!(output["usage"]["total_tokens"], 5);
    }

    #[test]
    fn stream_deduplicates_arguments_and_images_and_retains_finish_state() {
        let mut state = CodexToChatStreamState::default();
        let added = convert_codex_response_to_openai_chat_stream(
            "gpt", b"{}", b"", br#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc","type":"function_call","call_id":"c","name":"run"}}"#, &mut state,
        );
        assert_eq!(added.len(), 1);
        let delta = convert_codex_response_to_openai_chat_stream(
            "gpt", b"{}", b"", br#"data: {"type":"response.function_call_arguments.delta","item_id":"fc","delta":"{}"}"#, &mut state,
        );
        assert_eq!(delta.len(), 1);
        let done = convert_codex_response_to_openai_chat_stream(
            "gpt", b"{}", b"", br#"data: {"type":"response.output_item.done","item":{"id":"fc","type":"function_call","arguments":"{}"}}"#, &mut state,
        );
        assert!(done.is_empty());
        let first_image = convert_codex_response_to_openai_chat_stream(
            "gpt", b"{}", b"", br#"data: {"type":"response.image_generation_call.partial_image","item_id":"img","partial_image_b64":"AAAA"}"#, &mut state,
        );
        let duplicate = convert_codex_response_to_openai_chat_stream(
            "gpt", b"{}", b"", br#"data: {"type":"response.image_generation_call.partial_image","item_id":"img","partial_image_b64":"AAAA"}"#, &mut state,
        );
        assert_eq!(first_image.len(), 1);
        assert!(duplicate.is_empty());
        let finish = convert_codex_response_to_openai_chat_stream(
            "gpt",
            b"{}",
            b"",
            br#"data: {"type":"response.completed","response":{}}"#,
            &mut state,
        );
        let finish: Value = serde_json::from_slice(&finish[0]).unwrap();
        assert_eq!(finish["choices"][0]["finish_reason"], "tool_calls");
    }
}
