// ref: internal/translator/interactions/claude/interactions_claude_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;

use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::internal::translator::common::{append_sse_event, interactions_usage};

#[derive(Debug)]
pub struct InteractionsToClaudeStreamState {
    id: String,
    fallback_id: String,
    model: String,
    started: bool,
    active_block: bool,
    active_block_type: String,
    block_index: usize,
    saw_tool_call: bool,
    completed: bool,
    stopped: bool,
    done: bool,
    step_types: HashMap<usize, String>,
    tool_names: HashMap<usize, String>,
    tool_ids: HashMap<usize, String>,
    tool_signatures: HashMap<usize, String>,
}

impl Default for InteractionsToClaudeStreamState {
    fn default() -> Self {
        Self::with_identity(format!("msg_{}", Uuid::new_v4().simple()))
    }
}

impl InteractionsToClaudeStreamState {
    pub fn with_identity(fallback_id: impl Into<String>) -> Self {
        Self {
            id: String::new(),
            fallback_id: fallback_id.into(),
            model: String::new(),
            started: false,
            active_block: false,
            active_block_type: String::new(),
            block_index: 0,
            saw_tool_call: false,
            completed: false,
            stopped: false,
            done: false,
            step_types: HashMap::new(),
            tool_names: HashMap::new(),
            tool_ids: HashMap::new(),
            tool_signatures: HashMap::new(),
        }
    }
}

pub fn convert_interactions_response_to_claude(
    model_name: &str,
    _original_request: &[u8],
    _request: &[u8],
    raw_json: &[u8],
    state: &mut InteractionsToClaudeStreamState,
) -> Vec<Vec<u8>> {
    if state.model.trim().is_empty() {
        state.model = model_name.to_owned();
    }
    let payload = interactions_sse_payload(raw_json);
    if payload.is_empty() {
        return Vec::new();
    }
    if payload.trim_ascii() == b"[DONE]" {
        return append_message_stop(Vec::new(), state);
    }
    let Ok(root) = serde_json::from_slice::<Value>(&payload) else {
        return Vec::new();
    };
    match root
        .get("event_type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "interaction.created" => {
            let interaction = root.get("interaction").unwrap_or(&Value::Null);
            if let Some(id) = interaction
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.trim().is_empty())
            {
                state.id = id.to_owned();
            }
            if let Some(model) = interaction
                .get("model")
                .and_then(Value::as_str)
                .filter(|model| !model.trim().is_empty())
            {
                state.model = model.to_owned();
            }
            append_message_start(Vec::new(), state)
        }
        "step.start" => step_start(&root, state),
        "step.delta" => step_delta(&root, state),
        "step.stop" => append_content_block_stop(Vec::new(), state),
        "interaction.completed" | "finish" => append_message_delta(Vec::new(), &root, state),
        "done" => append_message_stop(Vec::new(), state),
        _ => Vec::new(),
    }
}

pub fn convert_interactions_response_to_claude_non_stream(
    model_name: &str,
    _original_request: &[u8],
    _request: &[u8],
    raw_json: &[u8],
) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(raw_json).unwrap_or(Value::Null);
    let interaction = root.get("interaction").unwrap_or(&root);
    let id = first_nonempty(&[
        interaction
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        root.get("id").and_then(Value::as_str).unwrap_or_default(),
    ])
    .map(str::to_owned)
    .unwrap_or_else(|| format!("msg_{}", Uuid::new_v4().simple()));
    let model = first_nonempty(&[
        interaction
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        model_name,
    ])
    .unwrap_or_default()
    .to_owned();
    let steps = interaction
        .get("steps")
        .or_else(|| root.get("steps"))
        .and_then(Value::as_array);
    let mut content = Vec::new();
    let mut saw_tool_call = false;
    for step in steps.into_iter().flatten() {
        match step.get("type").and_then(Value::as_str).unwrap_or_default() {
            "thought" => content.extend(
                content_texts(step.get("content"))
                    .into_iter()
                    .map(|text| json!({"type":"thinking","thinking":text})),
            ),
            "function_call" => {
                saw_tool_call = true;
                let mut block = json!({
                    "type":"tool_use",
                    "id":tool_id(step),
                    "name":step.get("name").and_then(Value::as_str).unwrap_or_default(),
                    "input":first_existing(step, &["arguments","args"])
                        .filter(|value| value.is_object())
                        .cloned()
                        .unwrap_or_else(|| json!({})),
                });
                if let Some(signature) = signature(step).filter(|value| !value.is_empty()) {
                    block["signature"] = Value::String(signature.to_owned());
                }
                content.push(block);
            }
            _ => content.extend(
                content_texts(step.get("content"))
                    .into_iter()
                    .map(|text| json!({"type":"text","text":text})),
            ),
        }
    }
    let mut out = json!({
        "id":id,
        "type":"message",
        "role":"assistant",
        "model":model,
        "content":content,
        "stop_reason":if saw_tool_call {"tool_use"} else {"end_turn"},
        "stop_sequence":Value::Null,
        "usage":{"input_tokens":0,"output_tokens":0},
    });
    set_usage(&mut out["usage"], interactions_usage(&root));
    serde_json::to_vec(&out).unwrap_or_default()
}

fn step_start(root: &Value, state: &mut InteractionsToClaudeStreamState) -> Vec<Vec<u8>> {
    let mut out = append_message_start(Vec::new(), state);
    out = append_content_block_stop(out, state);
    let index = root.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
    let step = root.get("step").unwrap_or(&Value::Null);
    let step_type = step.get("type").and_then(Value::as_str).unwrap_or_default();
    state.step_types.insert(index, step_type.to_owned());
    match step_type {
        "function_call" => {
            state.saw_tool_call = true;
            state.tool_names.insert(
                index,
                step.get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            );
            state.tool_ids.insert(index, tool_id(step));
            state
                .tool_signatures
                .insert(index, signature(step).unwrap_or_default().to_owned());
            append_tool_block_start(out, index, state)
        }
        "thought" => append_content_block_start(out, "thinking", state),
        _ => append_content_block_start(out, "text", state),
    }
}

fn step_delta(root: &Value, state: &mut InteractionsToClaudeStreamState) -> Vec<Vec<u8>> {
    let index = root.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
    let delta = root.get("delta").unwrap_or(&Value::Null);
    match delta
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "thought_summary" => {
            let mut out = append_message_start(Vec::new(), state);
            out = ensure_content_block(out, "thinking", state);
            let text = first_nonempty(&[
                delta
                    .pointer("/content/text")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                delta
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            ])
            .unwrap_or_default()
            .to_owned();
            append_content_delta(out, "thinking_delta", "thinking", &text, state)
        }
        "thought_signature" if state.active_block && state.active_block_type == "thinking" => {
            append_content_delta(
                Vec::new(),
                "signature_delta",
                "signature",
                delta
                    .get("signature")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                state,
            )
        }
        "arguments_delta" => {
            let mut out = append_message_start(Vec::new(), state);
            if !state.active_block || state.active_block_type != "tool_use" {
                out = append_content_block_stop(out, state);
                if state
                    .tool_names
                    .get(&index)
                    .is_none_or(|name| name.is_empty())
                {
                    state.tool_names.insert(
                        index,
                        root.pointer("/step/name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    );
                }
                state
                    .tool_ids
                    .entry(index)
                    .or_insert_with(|| format!("toolu_{index}"));
                out = append_tool_block_start(out, index, state);
            }
            append_content_delta(
                out,
                "input_json_delta",
                "partial_json",
                delta
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                state,
            )
        }
        _ => {
            let mut out = append_message_start(Vec::new(), state);
            out = ensure_content_block(out, "text", state);
            append_content_delta(
                out,
                "text_delta",
                "text",
                delta
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                state,
            )
        }
    }
}

fn append_message_start(
    mut out: Vec<Vec<u8>>,
    state: &mut InteractionsToClaudeStreamState,
) -> Vec<Vec<u8>> {
    if state.started {
        return out;
    }
    let id = first_nonempty(&[&state.id, &state.fallback_id])
        .unwrap_or_default()
        .to_owned();
    out.push(event(
        "message_start",
        &json!({
            "type":"message_start",
            "message":{"id":id,"type":"message","role":"assistant","content":[],"model":state.model,"stop_reason":Value::Null,"stop_sequence":Value::Null,"usage":{"input_tokens":0,"output_tokens":0}}
        }),
    ));
    state.started = true;
    out
}

fn append_content_block_start(
    mut out: Vec<Vec<u8>>,
    block_type: &str,
    state: &mut InteractionsToClaudeStreamState,
) -> Vec<Vec<u8>> {
    if state.active_block && state.active_block_type == block_type {
        return out;
    }
    out = append_content_block_stop(out, state);
    let content_block = if block_type == "thinking" {
        json!({"type":"thinking","thinking":""})
    } else {
        json!({"type":"text","text":""})
    };
    out.push(event(
        "content_block_start",
        &json!({"type":"content_block_start","index":state.block_index,"content_block":content_block}),
    ));
    state.active_block = true;
    state.active_block_type = block_type.to_owned();
    out
}

fn append_tool_block_start(
    mut out: Vec<Vec<u8>>,
    step_index: usize,
    state: &mut InteractionsToClaudeStreamState,
) -> Vec<Vec<u8>> {
    out = append_content_block_stop(out, state);
    let mut block = json!({
        "type":"tool_use",
        "id":state.tool_ids.get(&step_index).filter(|id| !id.is_empty()).cloned().unwrap_or_else(|| format!("toolu_{step_index}")),
        "name":state.tool_names.get(&step_index).cloned().unwrap_or_default(),
        "input":{},
    });
    if let Some(signature) = state
        .tool_signatures
        .get(&step_index)
        .filter(|signature| !signature.is_empty())
    {
        block["signature"] = Value::String(signature.clone());
    }
    out.push(event(
        "content_block_start",
        &json!({"type":"content_block_start","index":state.block_index,"content_block":block}),
    ));
    state.active_block = true;
    state.active_block_type = "tool_use".into();
    out
}

fn ensure_content_block(
    out: Vec<Vec<u8>>,
    block_type: &str,
    state: &mut InteractionsToClaudeStreamState,
) -> Vec<Vec<u8>> {
    if state.active_block && state.active_block_type == block_type {
        out
    } else {
        append_content_block_start(out, block_type, state)
    }
}

fn append_content_delta(
    mut out: Vec<Vec<u8>>,
    delta_type: &str,
    field: &str,
    value: &str,
    state: &InteractionsToClaudeStreamState,
) -> Vec<Vec<u8>> {
    if value.is_empty() && delta_type != "input_json_delta" {
        return out;
    }
    let mut delta = Map::new();
    delta.insert("type".into(), Value::String(delta_type.into()));
    delta.insert(field.into(), Value::String(value.into()));
    out.push(event(
        "content_block_delta",
        &json!({"type":"content_block_delta","index":state.block_index,"delta":delta}),
    ));
    out
}

fn append_content_block_stop(
    mut out: Vec<Vec<u8>>,
    state: &mut InteractionsToClaudeStreamState,
) -> Vec<Vec<u8>> {
    if !state.active_block {
        return out;
    }
    out.push(event(
        "content_block_stop",
        &json!({"type":"content_block_stop","index":state.block_index}),
    ));
    state.active_block = false;
    state.active_block_type.clear();
    state.block_index += 1;
    out
}

fn append_message_delta(
    mut out: Vec<Vec<u8>>,
    root: &Value,
    state: &mut InteractionsToClaudeStreamState,
) -> Vec<Vec<u8>> {
    if state.completed {
        return out;
    }
    out = append_message_start(out, state);
    out = append_content_block_stop(out, state);
    let mut usage = json!({"input_tokens":0,"output_tokens":0});
    set_usage(&mut usage, interactions_usage(root));
    out.push(event(
        "message_delta",
        &json!({
            "type":"message_delta",
            "delta":{"stop_reason":if state.saw_tool_call {"tool_use"} else {"end_turn"},"stop_sequence":Value::Null},
            "usage":usage,
        }),
    ));
    state.completed = true;
    out
}

fn append_message_stop(
    mut out: Vec<Vec<u8>>,
    state: &mut InteractionsToClaudeStreamState,
) -> Vec<Vec<u8>> {
    if state.done {
        return out;
    }
    out = append_content_block_stop(out, state);
    if !state.completed {
        out = append_message_delta(out, &Value::Null, state);
    }
    if !state.stopped {
        out.push(event("message_stop", &json!({"type":"message_stop"})));
        state.stopped = true;
    }
    state.done = true;
    out
}

fn set_usage(target: &mut Value, usage: Option<&Value>) {
    let Some(usage) = usage else { return };
    if let Some(value) = first_usage_int(usage, &["input_tokens", "total_input_tokens"]) {
        target["input_tokens"] = Value::from(value);
    }
    if let Some(value) = first_usage_int(usage, &["output_tokens", "total_output_tokens"]) {
        target["output_tokens"] = Value::from(value);
    }
}

fn first_usage_int(root: &Value, paths: &[&str]) -> Option<i64> {
    paths.iter().find_map(|path| {
        root.get(*path).and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_u64().map(|value| value as i64))
        })
    })
}

fn interactions_sse_payload(raw: &[u8]) -> Vec<u8> {
    let trimmed = raw.trim_ascii();
    if trimmed.is_empty() || trimmed == b"[DONE]" {
        return trimmed.to_vec();
    }
    if let Some(payload) = trimmed.strip_prefix(b"data:") {
        return payload.trim_ascii().to_vec();
    }
    let data = trimmed
        .split(|byte| *byte == b'\n')
        .filter_map(|line| line.trim_ascii().strip_prefix(b"data:"))
        .map(|line| line.trim_ascii())
        .collect::<Vec<_>>();
    if data.is_empty() {
        trimmed.to_vec()
    } else {
        data.join(&b'\n')
    }
}

fn content_texts(content: Option<&Value>) -> Vec<String> {
    match content {
        Some(Value::String(text)) => vec![text.clone()],
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                first_nonempty(&[
                    part.get("text").and_then(Value::as_str).unwrap_or_default(),
                    part.pointer("/content/text")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                ])
                .map(str::to_owned)
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn tool_id(root: &Value) -> String {
    first_nonempty(&[
        root.get("call_id")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        root.get("id").and_then(Value::as_str).unwrap_or_default(),
        root.get("tool_use_id")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        "toolu_interactions",
    ])
    .unwrap_or_default()
    .to_owned()
}

fn signature(root: &Value) -> Option<&str> {
    [
        "/signature",
        "/thought_signature",
        "/thoughtSignature",
        "/extra_content/google/thought_signature",
    ]
    .into_iter()
    .find_map(|path| root.pointer(path).and_then(Value::as_str))
    .filter(|value| !value.trim().is_empty())
}

fn first_existing<'a>(root: &'a Value, fields: &[&str]) -> Option<&'a Value> {
    fields.iter().find_map(|field| root.get(*field))
}

fn first_nonempty<'a>(values: &'a [&'a str]) -> Option<&'a str> {
    values
        .iter()
        .copied()
        .find(|value| !value.trim().is_empty())
}

fn event(name: &str, payload: &Value) -> Vec<u8> {
    let payload = serde_json::to_vec(payload).unwrap_or_default();
    let mut out = Vec::with_capacity(name.len() + payload.len() + 16);
    append_sse_event(&mut out, name, &payload, 3);
    out
}
