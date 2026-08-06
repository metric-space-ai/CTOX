// ref: internal/translator/claude/interactions/interactions_claude_response.go:1-583 @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;

use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::internal::translator::common::sse_event_data;

#[derive(Debug)]
pub struct ClaudeToInteractionsState {
    id: String,
    model: String,
    created: bool,
    status_updated: bool,
    completed: bool,
    done: bool,
    usage: Map<String, Value>,
    step_index: usize,
    active_source_index: Option<u64>,
    current_step: HashMap<u64, String>,
    tool_names: HashMap<u64, String>,
    tool_ids: HashMap<u64, String>,
    tool_args: HashMap<u64, String>,
    timestamp: String,
}

impl Default for ClaudeToInteractionsState {
    fn default() -> Self {
        Self::with_timestamp("1970-01-01T00:00:00Z")
    }
}

impl ClaudeToInteractionsState {
    pub fn with_timestamp(timestamp: impl Into<String>) -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            created: false,
            status_updated: false,
            completed: false,
            done: false,
            usage: Map::new(),
            step_index: 0,
            active_source_index: None,
            current_step: HashMap::new(),
            tool_names: HashMap::new(),
            tool_ids: HashMap::new(),
            tool_args: HashMap::new(),
            timestamp: timestamp.into(),
        }
    }
}

pub fn convert_claude_response_to_interactions(
    model: &str,
    _original: &[u8],
    _request: &[u8],
    raw: &[u8],
    state: &mut ClaudeToInteractionsState,
) -> Vec<Vec<u8>> {
    state.model = first_nonempty(&[&state.model, model]).to_owned();
    let Some(payload) = sse_payload(raw) else {
        return Vec::new();
    };
    if payload == b"[DONE]" {
        return append_done(Vec::new(), state);
    }
    let Ok(root) = serde_json::from_slice::<Value>(payload) else {
        return Vec::new();
    };
    match root.get("type").and_then(Value::as_str).unwrap_or_default() {
        "message_start" => {
            let message = &root["message"];
            state.id = first_nonempty(&[
                message
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                &state.id,
                &generated_id(),
            ])
            .to_owned();
            state.model = first_nonempty(&[
                message
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                &state.model,
                model,
            ])
            .to_owned();
            merge_usage(state, message.get("usage"));
            append_created(Vec::new(), state, model)
        }
        "content_block_start" => block_start(model, &root, state),
        "content_block_delta" => block_delta(model, &root, state),
        "content_block_stop" => block_stop(&root, state),
        "message_delta" => {
            merge_usage(state, root.get("usage"));
            let out = append_step_stop(Vec::new(), state);
            append_completed(out, state, model, root.get("usage"))
        }
        "message_stop" if !state.completed => {
            append_completed(Vec::new(), state, model, root.get("usage"))
        }
        "error" => {
            let out = append_created(Vec::new(), state, model);
            append_completed(out, state, model, root.get("usage"))
        }
        _ => Vec::new(),
    }
}

pub fn convert_claude_response_to_interactions_non_stream(
    model: &str,
    _original: &[u8],
    _request: &[u8],
    raw: &[u8],
) -> Vec<u8> {
    if let Ok(root) = serde_json::from_slice::<Value>(raw) {
        if root.get("content").is_some() {
            return message_to_interaction(model, &root);
        }
    }
    sse_to_interaction(model, raw)
}

fn message_to_interaction(model: &str, root: &Value) -> Vec<u8> {
    let steps = root
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(content_block_step)
        .collect::<Vec<_>>();
    let mut out = json!({
        "id":root.get("id").and_then(Value::as_str).unwrap_or(""),
        "object":"interaction", "status":"completed",
        "model":root.get("model").and_then(Value::as_str).unwrap_or(model),
        "steps":steps,
    });
    if out["id"].as_str().is_some_and(str::is_empty) {
        out["id"] = Value::String(generated_id());
    }
    if let Some(usage) = root.get("usage") {
        out["usage"] = interactions_usage(usage);
    }
    serde_json::to_vec(&out).unwrap_or_default()
}

fn sse_to_interaction(model: &str, raw: &[u8]) -> Vec<u8> {
    let mut state = ClaudeToInteractionsState {
        id: generated_id(),
        model: model.to_owned(),
        ..Default::default()
    };
    let mut steps = Vec::new();
    let mut text: HashMap<u64, String> = HashMap::new();
    for line in raw.split(|byte| *byte == b'\n') {
        let Some(payload) = sse_payload(line) else {
            continue;
        };
        if payload == b"[DONE]" {
            continue;
        }
        let Ok(root) = serde_json::from_slice::<Value>(payload) else {
            continue;
        };
        let index = root.get("index").and_then(Value::as_u64).unwrap_or(0);
        match root.get("type").and_then(Value::as_str).unwrap_or_default() {
            "message_start" => {
                let message = &root["message"];
                if let Some(id) = message.get("id").and_then(Value::as_str) {
                    state.id = id.to_owned();
                }
                if let Some(event_model) = message.get("model").and_then(Value::as_str) {
                    state.model = event_model.to_owned();
                }
                merge_usage(&mut state, message.get("usage"));
            }
            "content_block_start" => {
                let block = &root["content_block"];
                let kind = block_step_type(
                    block
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                );
                state.current_step.insert(index, kind.to_owned());
                if kind == "function_call" {
                    if let Some(name) = block.get("name").and_then(Value::as_str) {
                        state.tool_names.insert(index, name.to_owned());
                    }
                    if let Some(id) = block.get("id").and_then(Value::as_str) {
                        state.tool_ids.insert(index, id.to_owned());
                    }
                }
            }
            "content_block_delta" => {
                let delta = &root["delta"];
                match delta
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                {
                    "text_delta" => text.entry(index).or_default().push_str(
                        delta
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    ),
                    "thinking_delta" => text.entry(index).or_default().push_str(
                        delta
                            .get("thinking")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    ),
                    "input_json_delta" => state.tool_args.entry(index).or_default().push_str(
                        delta
                            .get("partial_json")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    ),
                    _ => {}
                }
            }
            "content_block_stop" => {
                let kind = state.current_step.remove(&index).unwrap_or_default();
                let step = if kind == "function_call" {
                    tool_step(
                        state.tool_names.remove(&index).unwrap_or_default(),
                        state.tool_ids.remove(&index).unwrap_or_default(),
                        state.tool_args.remove(&index).unwrap_or_default(),
                    )
                } else {
                    text_step(&kind, text.remove(&index).unwrap_or_default())
                };
                steps.push(step);
            }
            "message_delta" => merge_usage(&mut state, root.get("usage")),
            _ => {}
        }
    }
    serde_json::to_vec(&json!({
        "id":state.id, "object":"interaction", "status":"completed",
        "model":state.model, "steps":steps,
        "usage":interactions_usage(&Value::Object(state.usage)),
    }))
    .unwrap_or_default()
}

fn block_start(model: &str, root: &Value, state: &mut ClaudeToInteractionsState) -> Vec<Vec<u8>> {
    let mut out = append_created(Vec::new(), state, model);
    out = append_step_stop(out, state);
    let index = root.get("index").and_then(Value::as_u64).unwrap_or(0);
    let block = &root["content_block"];
    let kind = block_step_type(
        block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    );
    state.current_step.insert(index, kind.to_owned());
    if kind == "function_call" {
        if let Some(name) = block.get("name").and_then(Value::as_str) {
            state.tool_names.insert(index, name.to_owned());
        }
        if let Some(id) = block.get("id").and_then(Value::as_str) {
            state.tool_ids.insert(index, id.to_owned());
        }
    }
    let step = if kind == "function_call" {
        tool_step(
            state.tool_names.get(&index).cloned().unwrap_or_default(),
            state.tool_ids.get(&index).cloned().unwrap_or_default(),
            "{}".to_owned(),
        )
    } else {
        json!({"type":kind})
    };
    append_step_start(out, state, index, kind, step)
}

fn block_delta(model: &str, root: &Value, state: &mut ClaudeToInteractionsState) -> Vec<Vec<u8>> {
    let index = root.get("index").and_then(Value::as_u64).unwrap_or(0);
    let delta = &root["delta"];
    let kind = state.current_step.get(&index).cloned().unwrap_or_else(|| {
        let kind = delta_step_type(
            delta
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        state.current_step.insert(index, kind.to_owned());
        kind.to_owned()
    });
    let mut out = Vec::new();
    if state.active_source_index != Some(index) {
        out = append_created(out, state, model);
        out = append_step_stop(out, state);
        let step = if kind == "function_call" {
            tool_step(
                state.tool_names.get(&index).cloned().unwrap_or_default(),
                state.tool_ids.get(&index).cloned().unwrap_or_default(),
                "{}".to_owned(),
            )
        } else {
            json!({"type":kind})
        };
        out = append_step_start(out, state, index, &kind, step);
    }
    match delta
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "text_delta" => append_text_delta(
            out,
            state,
            delta
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            false,
        ),
        "thinking_delta" => append_text_delta(
            out,
            state,
            delta
                .get("thinking")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            true,
        ),
        "input_json_delta" => {
            let partial = delta
                .get("partial_json")
                .and_then(Value::as_str)
                .unwrap_or_default();
            state.tool_args.entry(index).or_default().push_str(partial);
            append_arguments_delta(out, state, partial)
        }
        _ => out,
    }
}

fn block_stop(root: &Value, state: &mut ClaudeToInteractionsState) -> Vec<Vec<u8>> {
    let index = root.get("index").and_then(Value::as_u64).unwrap_or(0);
    let out = append_step_stop(Vec::new(), state);
    state.current_step.remove(&index);
    state.tool_names.remove(&index);
    state.tool_ids.remove(&index);
    state.tool_args.remove(&index);
    out
}

fn content_block_step(block: &Value) -> Option<Value> {
    match block.get("type")?.as_str()? {
        "text" => Some(text_step(
            "model_output",
            block.get("text")?.as_str()?.to_owned(),
        )),
        "thinking" => Some(text_step(
            "thought",
            block.get("thinking")?.as_str()?.to_owned(),
        )),
        "tool_use" => Some(tool_step(
            block
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            block
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            serde_json::to_string(block.get("input").unwrap_or(&json!({}))).ok()?,
        )),
        _ => None,
    }
}

fn text_step(kind: &str, text: String) -> Value {
    json!({"type":kind,"content":[{"type":"text","text":text}]})
}

fn tool_step(name: String, id: String, arguments: String) -> Value {
    let arguments = serde_json::from_str::<Value>(arguments.trim()).unwrap_or_else(|_| json!({}));
    let mut step = json!({"type":"function_call","name":name,"arguments":arguments});
    if !id.is_empty() {
        step["id"] = Value::String(id.clone());
        step["call_id"] = Value::String(id);
    }
    step
}

fn merge_usage(state: &mut ClaudeToInteractionsState, usage: Option<&Value>) {
    let Some(usage) = usage.and_then(Value::as_object) else {
        return;
    };
    for key in [
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "thinking_tokens",
    ] {
        if let Some(value) = usage.get(key) {
            state.usage.insert(key.to_owned(), value.clone());
        }
    }
}

fn interactions_usage(usage: &Value) -> Value {
    let number = |key: &str| usage.get(key).and_then(Value::as_i64).unwrap_or(0);
    let input = number("input_tokens");
    let output = number("output_tokens");
    let cached = number("cache_read_input_tokens") + number("cache_creation_input_tokens");
    let thinking = number("thinking_tokens");
    let mut out = Map::new();
    if usage.get("input_tokens").is_some() {
        out.insert("input_tokens".into(), Value::from(input));
        out.insert("total_input_tokens".into(), Value::from(input));
    }
    if usage.get("output_tokens").is_some() {
        out.insert("output_tokens".into(), Value::from(output));
        out.insert("total_output_tokens".into(), Value::from(output));
    }
    if usage.get("input_tokens").is_some() || usage.get("output_tokens").is_some() {
        out.insert("total_tokens".into(), Value::from(input + output));
    }
    if cached != 0 {
        out.insert("cached_tokens".into(), Value::from(cached));
        out.insert("total_cached_tokens".into(), Value::from(cached));
    }
    if thinking != 0 {
        out.insert("reasoning_tokens".into(), Value::from(thinking));
        out.insert("total_thought_tokens".into(), Value::from(thinking));
    }
    Value::Object(out)
}

fn append_created(
    mut out: Vec<Vec<u8>>,
    state: &mut ClaudeToInteractionsState,
    model: &str,
) -> Vec<Vec<u8>> {
    if state.created {
        return out;
    }
    if state.id.is_empty() {
        state.id = generated_id();
    }
    let payload = json!({
        "interaction":{"id":state.id,"status":"in_progress","object":"interaction",
            "model":first_nonempty(&[&state.model,model])},
        "event_type":"interaction.created"
    });
    out.push(event("interaction.created", payload));
    state.created = true;
    append_status(out, state)
}

fn append_status(mut out: Vec<Vec<u8>>, state: &mut ClaudeToInteractionsState) -> Vec<Vec<u8>> {
    if !state.status_updated {
        out.push(event(
            "interaction.status_update",
            json!({"interaction_id":state.id,"status":"in_progress","event_type":"interaction.status_update"}),
        ));
        state.status_updated = true;
    }
    out
}

fn append_step_start(
    mut out: Vec<Vec<u8>>,
    state: &mut ClaudeToInteractionsState,
    source_index: u64,
    _kind: &str,
    step: Value,
) -> Vec<Vec<u8>> {
    state.active_source_index = Some(source_index);
    out.push(event(
        "step.start",
        json!({"index":state.step_index,"step":step,"event_type":"step.start"}),
    ));
    out
}

fn append_text_delta(
    mut out: Vec<Vec<u8>>,
    state: &ClaudeToInteractionsState,
    text: &str,
    thought: bool,
) -> Vec<Vec<u8>> {
    let delta = if thought {
        json!({"type":"thought_summary","content":{"type":"text","text":text}})
    } else {
        json!({"type":"text","text":text})
    };
    out.push(event(
        "step.delta",
        json!({"index":state.step_index,"delta":delta,"event_type":"step.delta"}),
    ));
    out
}

fn append_arguments_delta(
    mut out: Vec<Vec<u8>>,
    state: &ClaudeToInteractionsState,
    arguments: &str,
) -> Vec<Vec<u8>> {
    out.push(event(
        "step.delta",
        json!({"index":state.step_index,"delta":{"arguments":arguments,"type":"arguments_delta"},"event_type":"step.delta"}),
    ));
    out
}

fn append_step_stop(mut out: Vec<Vec<u8>>, state: &mut ClaudeToInteractionsState) -> Vec<Vec<u8>> {
    if state.active_source_index.is_some() {
        out.push(event(
            "step.stop",
            json!({"index":state.step_index,"event_type":"step.stop"}),
        ));
        state.active_source_index = None;
        state.step_index += 1;
    }
    out
}

fn append_completed(
    mut out: Vec<Vec<u8>>,
    state: &mut ClaudeToInteractionsState,
    model: &str,
    fallback_usage: Option<&Value>,
) -> Vec<Vec<u8>> {
    if state.completed {
        return out;
    }
    out = append_created(out, state, model);
    if state.usage.is_empty() {
        merge_usage(state, fallback_usage);
    }
    let now = state.timestamp.clone();
    out.push(event(
        "interaction.completed",
        json!({"interaction":{"id":state.id,"status":"completed",
            "usage":interactions_usage(&Value::Object(state.usage.clone())),
            "created":now,"updated":now,"service_tier":"standard","object":"interaction",
            "model":first_nonempty(&[&state.model,model])},"event_type":"interaction.completed"}),
    ));
    state.completed = true;
    out
}

fn append_done(mut out: Vec<Vec<u8>>, state: &mut ClaudeToInteractionsState) -> Vec<Vec<u8>> {
    if !state.done {
        out.push(sse_event_data("done", b"[DONE]"));
        state.done = true;
    }
    out
}

fn event(name: &str, value: Value) -> Vec<u8> {
    sse_event_data(name, &serde_json::to_vec(&value).unwrap_or_default())
}

fn sse_payload(raw: &[u8]) -> Option<&[u8]> {
    let raw = trim_ascii(raw);
    if raw == b"[DONE]" {
        return Some(raw);
    }
    trim_ascii(raw.strip_prefix(b"data:")?).into()
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

fn block_step_type(kind: &str) -> &'static str {
    match kind {
        "thinking" => "thought",
        "tool_use" => "function_call",
        _ => "model_output",
    }
}

fn delta_step_type(kind: &str) -> &'static str {
    match kind {
        "thinking_delta" => "thought",
        "input_json_delta" => "function_call",
        _ => "model_output",
    }
}

fn generated_id() -> String {
    format!("interaction_{}", Uuid::new_v4().simple())
}

fn first_nonempty<'a>(values: &[&'a str]) -> &'a str {
    values
        .iter()
        .copied()
        .find(|value| !value.is_empty())
        .unwrap_or("")
}
