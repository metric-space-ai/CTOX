// ref: internal/translator/antigravity/interactions/interactions_antigravity_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;

use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::internal::translator::common::sse_event_data;
use crate::internal::util::{disambiguated_tool_name_map, restore_sanitized_tool_name};

#[derive(Clone, Debug)]
pub struct AntigravityToInteractionsState {
    started: bool,
    finished: bool,
    completed: bool,
    done: bool,
    active_step_open: bool,
    id: String,
    active_step_type: String,
    active_step_index: usize,
    step_index: usize,
    tool_name_map: HashMap<String, String>,
    timestamp: String,
}

impl Default for AntigravityToInteractionsState {
    fn default() -> Self {
        Self::with_identity(
            format!("interaction_{}", Uuid::new_v4().simple()),
            "1970-01-01T00:00:00Z",
        )
    }
}

impl AntigravityToInteractionsState {
    pub fn with_identity(id: impl Into<String>, timestamp: impl Into<String>) -> Self {
        Self {
            started: false,
            finished: false,
            completed: false,
            done: false,
            active_step_open: false,
            id: id.into(),
            active_step_type: String::new(),
            active_step_index: 0,
            step_index: 0,
            tool_name_map: HashMap::new(),
            timestamp: timestamp.into(),
        }
    }
}

pub fn convert_antigravity_response_to_interactions(
    model_name: &str,
    original_request: &[u8],
    _request: &[u8],
    raw: &[u8],
    state: &mut AntigravityToInteractionsState,
) -> Vec<Vec<u8>> {
    if state.tool_name_map.is_empty() {
        state.tool_name_map = disambiguated_tool_name_map(original_request);
    }
    let mut output = Vec::new();
    for payload in stream_payloads(raw) {
        if trim_ascii(&payload) == b"[DONE]" {
            if !state.completed {
                append_step_stop(&mut output, state);
                append_completed(&mut output, state, model_name, None);
            }
            append_done(&mut output, state);
            continue;
        }
        let Ok(root) = serde_json::from_slice::<Value>(&payload) else {
            continue;
        };
        let mut root = unwrap_response(&root);
        restore_function_names(&mut root, &state.tool_name_map);
        if root.is_null() {
            continue;
        }
        if !state.started {
            append_created(&mut output, state, model_name);
            append_status(&mut output, state);
            state.started = true;
        }
        for part in root
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            append_part(&mut output, state, part);
        }
        let has_finish = root.pointer("/candidates/0/finishReason").is_some();
        if has_finish && !state.finished {
            append_step_stop(&mut output, state);
            state.finished = true;
        }
        if has_usage(&root) && state.finished && !state.completed {
            append_completed(&mut output, state, model_name, Some(&root));
        }
    }
    output
}

pub fn convert_antigravity_response_to_interactions_non_stream(
    model_name: &str,
    original_request: &[u8],
    _request: &[u8],
    raw: &[u8],
) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(raw).unwrap_or(Value::Null);
    let mut root = unwrap_response(&root);
    restore_function_names(&mut root, &disambiguated_tool_name_map(original_request));
    let id = root
        .get("responseId")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("interaction_{}", Uuid::new_v4().simple()));
    let steps = root
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(part_to_step)
        .collect::<Vec<_>>();
    let mut out = json!({
        "id":id, "object":"interaction", "status":"completed",
        "model":model_name, "steps":steps,
    });
    if let Some(usage) = usage_node(&root) {
        out["usage"] = non_stream_usage(usage);
    }
    serde_json::to_vec(&out).unwrap_or_default()
}

fn stream_payloads(raw: &[u8]) -> Vec<Vec<u8>> {
    let trimmed = trim_ascii(raw);
    if let Some(payload) = trimmed.strip_prefix(b"data:") {
        return vec![trim_ascii(payload).to_vec()];
    }
    if let Ok(Value::Array(items)) = serde_json::from_slice::<Value>(trimmed) {
        let payloads = items
            .iter()
            .filter_map(|item| item.get("response").or(Some(item)))
            .filter_map(|item| serde_json::to_vec(item).ok())
            .collect::<Vec<_>>();
        if !payloads.is_empty() {
            return payloads;
        }
    }
    vec![trimmed.to_vec()]
}

fn unwrap_response(root: &Value) -> Value {
    let mut response = root
        .get("response")
        .cloned()
        .unwrap_or_else(|| root.clone());
    if response.get("usageMetadata").is_none() {
        if let Some(cpa) = response
            .as_object_mut()
            .and_then(|map| map.remove("cpaUsageMetadata"))
        {
            response["usageMetadata"] = cpa;
        }
    }
    response
}

fn restore_function_names(root: &mut Value, names: &HashMap<String, String>) {
    if names.is_empty() {
        return;
    }
    for part in root
        .get_mut("candidates")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|candidate| candidate.pointer_mut("/content/parts"))
        .filter_map(Value::as_array_mut)
        .flatten()
    {
        for field in ["functionCall", "functionResponse"] {
            if let Some(call) = part.get_mut(field).and_then(Value::as_object_mut) {
                if let Some(name) = call.get("name").and_then(Value::as_str) {
                    call.insert(
                        "name".into(),
                        Value::String(restore_sanitized_tool_name(names, name)),
                    );
                }
            }
        }
    }
}

fn append_created(output: &mut Vec<Vec<u8>>, state: &AntigravityToInteractionsState, model: &str) {
    event(
        output,
        "interaction.created",
        json!({
            "interaction":{"id":state.id,"status":"in_progress","object":"interaction","model":model},
            "event_type":"interaction.created",
        }),
    );
}

fn append_status(output: &mut Vec<Vec<u8>>, state: &AntigravityToInteractionsState) {
    event(
        output,
        "interaction.status_update",
        json!({
            "interaction_id":state.id,"status":"in_progress","event_type":"interaction.status_update",
        }),
    );
}

fn append_completed(
    output: &mut Vec<Vec<u8>>,
    state: &mut AntigravityToInteractionsState,
    model: &str,
    root: Option<&Value>,
) {
    let usage = root
        .and_then(usage_node)
        .map(stream_usage)
        .unwrap_or_else(|| json!({}));
    event(
        output,
        "interaction.completed",
        json!({
            "interaction":{
                "id":state.id,"status":"completed","usage":usage,
                "created":state.timestamp,"updated":state.timestamp,
                "service_tier":"standard","object":"interaction","model":model,
            },
            "event_type":"interaction.completed",
        }),
    );
    state.completed = true;
}

fn append_done(output: &mut Vec<Vec<u8>>, state: &mut AntigravityToInteractionsState) {
    if !state.done {
        output.push(sse_event_data("done", b"[DONE]"));
        state.done = true;
    }
}

fn ensure_step(
    output: &mut Vec<Vec<u8>>,
    state: &mut AntigravityToInteractionsState,
    step_type: &str,
    part: Option<&Value>,
) {
    if state.active_step_open && state.active_step_type == step_type {
        return;
    }
    append_step_stop(output, state);
    state.active_step_index = state.step_index;
    state.step_index += 1;
    state.active_step_type = step_type.to_owned();
    state.active_step_open = true;
    let mut step = json!({"type":step_type});
    if step_type == "function_call" {
        let null = Value::Null;
        let part = part.unwrap_or(&null);
        let id = part
            .get("id")
            .or_else(|| part.get("call_id"))
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("step_{}", Uuid::new_v4().simple()));
        step["id"] = Value::String(id.clone());
        step["call_id"] = Value::String(id);
        step["name"] = Value::String(
            part.get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
        );
        step["arguments"] = json!({});
    }
    event(
        output,
        "step.start",
        json!({
            "index":state.active_step_index,"step":step,"event_type":"step.start",
        }),
    );
}

fn append_step_stop(output: &mut Vec<Vec<u8>>, state: &mut AntigravityToInteractionsState) {
    if !state.active_step_open {
        return;
    }
    event(
        output,
        "step.stop",
        json!({
            "index":state.active_step_index,"event_type":"step.stop",
        }),
    );
    state.active_step_open = false;
    state.active_step_type.clear();
}

fn append_part(
    output: &mut Vec<Vec<u8>>,
    state: &mut AntigravityToInteractionsState,
    part: &Value,
) {
    if let Some(text) = part
        .get("text")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        if part.get("thought").and_then(Value::as_bool) == Some(true) {
            ensure_step(output, state, "thought", None);
            event(
                output,
                "step.delta",
                json!({
                    "index":state.active_step_index,
                    "delta":{"content":{"text":text,"type":"text"},"type":"thought_summary"},
                    "event_type":"step.delta",
                }),
            );
            append_signature(output, state, part);
        } else {
            ensure_step(output, state, "model_output", None);
            event(
                output,
                "step.delta",
                json!({
                    "index":state.active_step_index,"delta":{"text":text,"type":"text"},
                    "event_type":"step.delta",
                }),
            );
        }
        return;
    }
    if let Some(call) = part.get("functionCall") {
        append_signature(output, state, part);
        ensure_step(output, state, "function_call", Some(call));
        let arguments = call.get("args").cloned().unwrap_or_else(|| json!({}));
        event(
            output,
            "step.delta",
            json!({
                "index":state.active_step_index,
                "delta":{"arguments":arguments.to_string(),"type":"arguments_delta"},
                "event_type":"step.delta",
            }),
        );
        append_step_stop(output, state);
        return;
    }
    if let Some(response) = part.get("functionResponse") {
        ensure_step(output, state, "function_result", Some(response));
        event(
            output,
            "step.delta",
            json!({
                "index":state.active_step_index,
                "delta":{
                    "type":"function_result",
                    "name":response.get("name").and_then(Value::as_str).unwrap_or(""),
                    "result":response.get("response").cloned().unwrap_or_else(||json!({})),
                },
                "event_type":"step.delta",
            }),
        );
        append_step_stop(output, state);
    }
}

fn append_signature(
    output: &mut Vec<Vec<u8>>,
    state: &mut AntigravityToInteractionsState,
    part: &Value,
) {
    let signature = [
        part.get("thoughtSignature"),
        part.get("thought_signature"),
        part.pointer("/extra_content/google/thought_signature"),
    ]
    .into_iter()
    .flatten()
    .filter_map(Value::as_str)
    .map(str::trim)
    .find(|signature| !signature.is_empty());
    if let Some(signature) = signature {
        ensure_step(output, state, "thought", None);
        event(
            output,
            "step.delta",
            json!({
                "index":state.active_step_index,
                "delta":{"signature":signature,"type":"thought_signature"},
                "event_type":"step.delta",
            }),
        );
    }
}

fn part_to_step(part: &Value) -> Option<Value> {
    if let Some(call) = part.get("functionCall") {
        let mut step = json!({
            "type":"function_call",
            "name":call.get("name").and_then(Value::as_str).unwrap_or(""),
            "arguments":call.get("args").cloned().unwrap_or_else(||json!({})),
        });
        if let Some(id) = call
            .get("id")
            .or_else(|| call.get("call_id"))
            .and_then(Value::as_str)
        {
            step["call_id"] = Value::String(id.to_owned());
        }
        return Some(step);
    }
    if let Some(response) = part.get("functionResponse") {
        let mut step = json!({
            "type":"function_result",
            "name":response.get("name").and_then(Value::as_str).unwrap_or(""),
            "result":response.get("response").cloned().unwrap_or_else(||json!({})),
        });
        if let Some(id) = response
            .get("id")
            .or_else(|| response.get("call_id"))
            .and_then(Value::as_str)
        {
            step["call_id"] = Value::String(id.to_owned());
        }
        return Some(step);
    }
    if let Some(text) = part.get("text") {
        return Some(json!({
            "type":if part.get("thought").and_then(Value::as_bool)==Some(true){"thought"}else{"model_output"},
            "content":[{"type":"text","text":text.as_str().unwrap_or("")}],
        }));
    }
    part.get("inlineData")
        .or_else(|| part.get("inline_data"))
        .and_then(inline_data_step)
}

fn inline_data_step(inline: &Value) -> Option<Value> {
    let mime = inline
        .get("mimeType")
        .or_else(|| inline.get("mime_type"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let data = inline.get("data").and_then(Value::as_str).unwrap_or("");
    if mime.is_empty() || data.is_empty() {
        return None;
    }
    let kind = if mime.to_ascii_lowercase().starts_with("image/") {
        "image"
    } else if mime.to_ascii_lowercase().starts_with("audio/") {
        "audio"
    } else if mime.to_ascii_lowercase().starts_with("video/") {
        "video"
    } else {
        "document"
    };
    Some(json!({
        "type":"model_output",
        "content":[{"type":kind,"mime_type":mime,"data":data}],
    }))
}

fn usage_node(root: &Value) -> Option<&Value> {
    root.get("usageMetadata")
        .or_else(|| root.get("usage_metadata"))
        .or_else(|| root.get("cpaUsageMetadata"))
}

fn has_usage(root: &Value) -> bool {
    usage_node(root).is_some_and(|usage| {
        [
            "promptTokenCount",
            "candidatesTokenCount",
            "totalTokenCount",
            "thoughtsTokenCount",
            "cachedContentTokenCount",
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "thoughts_token_count",
            "cached_content_token_count",
        ]
        .iter()
        .any(|key| usage.get(key).is_some())
    })
}

fn usage_int(usage: &Value, camel: &str, snake: &str) -> i64 {
    usage
        .get(camel)
        .or_else(|| usage.get(snake))
        .and_then(Value::as_i64)
        .unwrap_or(0)
}

fn non_stream_usage(usage: &Value) -> Value {
    let mut out = Map::new();
    out.insert(
        "input_tokens".into(),
        Value::from(usage_int(usage, "promptTokenCount", "prompt_token_count")),
    );
    out.insert(
        "output_tokens".into(),
        Value::from(usage_int(
            usage,
            "candidatesTokenCount",
            "candidates_token_count",
        )),
    );
    out.insert(
        "total_tokens".into(),
        Value::from(usage_int(usage, "totalTokenCount", "total_token_count")),
    );
    if usage
        .get("thoughtsTokenCount")
        .or_else(|| usage.get("thoughts_token_count"))
        .is_some()
    {
        out.insert(
            "reasoning_tokens".into(),
            Value::from(usage_int(
                usage,
                "thoughtsTokenCount",
                "thoughts_token_count",
            )),
        );
    }
    if usage
        .get("cachedContentTokenCount")
        .or_else(|| usage.get("cached_content_token_count"))
        .is_some()
    {
        out.insert(
            "cached_tokens".into(),
            Value::from(usage_int(
                usage,
                "cachedContentTokenCount",
                "cached_content_token_count",
            )),
        );
    }
    Value::Object(out)
}

fn stream_usage(usage: &Value) -> Value {
    let input = usage_int(usage, "promptTokenCount", "prompt_token_count");
    json!({
        "total_tokens":usage_int(usage,"totalTokenCount","total_token_count"),
        "total_input_tokens":input,
        "input_tokens_by_modality":[{"modality":"text","tokens":input}],
        "total_cached_tokens":usage_int(usage,"cachedContentTokenCount","cached_content_token_count"),
        "total_output_tokens":usage_int(usage,"candidatesTokenCount","candidates_token_count"),
        "total_tool_use_tokens":0,
        "total_thought_tokens":usage_int(usage,"thoughtsTokenCount","thoughts_token_count"),
    })
}

fn event(output: &mut Vec<Vec<u8>>, name: &str, payload: Value) {
    if let Ok(raw) = serde_json::to_vec(&payload) {
        output.push(sse_event_data(name, &raw));
    }
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
