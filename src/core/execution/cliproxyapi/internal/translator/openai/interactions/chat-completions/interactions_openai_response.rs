// ref: internal/translator/openai/interactions/chat-completions/interactions_openai_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

//! OpenAI chat-completions response body -> Interactions response body.
//!
//! Mirrors upstream's gjson/sjson byte-splice semantics on a typed
//! `serde_json::Value` builder. The stateful stream transformation uses the
//! shared `internal::translator::common::append_sse_event` helper to keep
//! the SSE envelope stable across leaves.

use std::collections::HashMap;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::openai_interactions_request::{
    first_nonempty, openai_reasoning_texts, openai_tool_call_to_interactions_step,
};
use crate::internal::translator::common::{append_sse_event, interactions_usage};
use crate::sdk::translator::{TranslationContext, TranslationState};

/// Request-local state carried across stream chunks for one OpenAI chat
/// stream. All fields are populated from a `TranslationState` Box<dyn Any>
/// slot so the orchestrator owns lifetime/serialization rules.
#[derive(Debug, Default)]
pub struct OpenAIToInteractionsStreamState {
    pub created: bool,
    pub status_updated: bool,
    pub completed: bool,
    pub done: bool,
    pub current_step_type: String,
    pub current_step_id: String,
    pub tool_call_ids: HashMap<i64, String>,
    pub tool_call_names: HashMap<i64, String>,
    pub id: String,
    pub model: String,
    pub step_index: i64,
    pub active_step_index: i64,
    pub active_step_open: bool,
    pub usage: Option<Value>,
    pub completed_at: String,
}

pub fn convert_openai_response_to_interactions(
    context: &TranslationContext,
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw_json: &[u8],
    state: &mut TranslationState,
) -> Vec<Vec<u8>> {
    if context.is_cancelled() {
        return Vec::new();
    }
    let state = stream_state(state);
    if state.id.is_empty() {
        state.id = deterministic_id("interaction", model_name, original_request, request);
    }
    if state.completed_at.is_empty() {
        state.completed_at = deterministic_timestamp(model_name, original_request, request);
    }
    state.model = model_name.to_owned();
    convert_openai_chat_stream_to_interactions(raw_json, state)
}

pub fn convert_openai_response_to_interactions_non_stream(
    _context: &TranslationContext,
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw_json: &[u8],
    _state: &mut TranslationState,
) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(raw_json).unwrap_or(Value::Null);
    let mut out = Map::new();
    out.insert("id".into(), Value::String(String::new()));
    out.insert("status".into(), Value::String("completed".into()));
    out.insert("object".into(), Value::String("interaction".into()));
    out.insert("model".into(), Value::String(String::new()));
    out.insert("steps".into(), Value::Array(Vec::new()));
    out["id"] = Value::String(first_nonempty(&[
        root.get("id").and_then(Value::as_str).unwrap_or(""),
        &deterministic_id("interaction", model_name, original_request, request),
    ]));
    out["model"] = Value::String(first_nonempty(&[
        model_name,
        root.get("model").and_then(Value::as_str).unwrap_or(""),
    ]));
    if let Some(choices) = root.get("choices").and_then(Value::as_array) {
        for choice in choices {
            let message = choice.get("message").unwrap_or(&Value::Null);
            if let Some(reasoning) = message.get("reasoning_content") {
                for text in openai_reasoning_texts(reasoning) {
                    append_step(&mut out, interactions_text_step("thought", &text));
                }
            }
            if let Some(content) = message.get("content") {
                let text = content.as_str().unwrap_or("");
                if !text.is_empty() {
                    append_step(&mut out, interactions_text_step("model_output", text));
                }
            }
            if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
                for tool_call in tool_calls {
                    if let Some(step) = openai_tool_call_to_interactions_step(tool_call) {
                        append_step(&mut out, step);
                    }
                }
            }
            if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
                out.insert(
                    "finish_reason".into(),
                    Value::String(finish_reason.to_owned()),
                );
            }
        }
    }
    set_interactions_usage_from_openai_chat(
        &mut out,
        "usage",
        root.get("usage").unwrap_or(&Value::Null),
    );
    serde_json::to_vec(&Value::Object(out)).unwrap_or_default()
}

fn stream_state(state: &mut TranslationState) -> &mut OpenAIToInteractionsStreamState {
    let replace = state
        .as_ref()
        .is_none_or(|value| !value.is::<OpenAIToInteractionsStreamState>());
    if replace {
        *state = Some(Box::new(OpenAIToInteractionsStreamState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<OpenAIToInteractionsStreamState>())
        .expect("OpenAI->Interactions state was initialized with the expected type")
}

fn convert_openai_chat_stream_to_interactions(
    raw_json: &[u8],
    st: &mut OpenAIToInteractionsStreamState,
) -> Vec<Vec<u8>> {
    let model_name = st.model.clone();
    let payload = openai_chat_sse_payload(raw_json);
    if payload.is_empty() {
        return Vec::new();
    }
    if payload.trim_ascii() == b"[DONE]" {
        let mut out = Vec::new();
        out.extend(append_interactions_step_stop(st));
        if !st.completed {
            out.extend(append_interactions_completed(st, &model_name, &Value::Null));
        }
        out.extend(append_interactions_done(st));
        return out;
    }
    let Ok(root) = serde_json::from_slice::<Value>(&payload) else {
        return Vec::new();
    };
    if let Some(usage) = root.get("usage") {
        st.usage = Some(usage.clone());
    }
    let mut out = Vec::new();
    if let Some(choices) = root.get("choices").and_then(Value::as_array) {
        if choices.is_empty() {
            if root.get("usage").is_some() {
                out.extend(append_interactions_step_stop(st));
                out.extend(append_interactions_completed(st, &model_name, &root));
            }
            return out;
        }
        for choice in choices {
            let delta = choice.get("delta").unwrap_or(&Value::Null);
            if let Some(reasoning) = delta.get("reasoning_content") {
                for text in openai_reasoning_texts(reasoning) {
                    out.extend(ensure_interactions_step(st, "thought", &root, &model_name));
                    out.extend(append_interactions_text_delta(st, &text, true));
                }
            }
            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                if !content.is_empty() {
                    out.extend(ensure_interactions_step(
                        st,
                        "model_output",
                        &root,
                        &model_name,
                    ));
                    out.extend(append_interactions_text_delta(st, content, false));
                }
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for tool_call in tool_calls {
                    out.extend(append_openai_tool_call_delta(
                        st,
                        &root,
                        tool_call,
                        &model_name,
                    ));
                }
            }
            if choice.get("finish_reason").is_some() {
                out.extend(append_interactions_step_stop(st));
            }
        }
    }
    out
}

fn append_openai_tool_call_delta(
    st: &mut OpenAIToInteractionsStreamState,
    root: &Value,
    tool_call: &Value,
    model_name: &str,
) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let index = tool_call.get("index").and_then(Value::as_i64).unwrap_or(0);
    if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
        if !id.is_empty() {
            st.tool_call_ids.insert(index, id.to_owned());
        }
    }
    if let Some(function) = tool_call.get("function") {
        if let Some(name) = function.get("name").and_then(Value::as_str) {
            if !name.is_empty() {
                st.tool_call_names.insert(index, name.to_owned());
            }
        }
    }
    let step_id = first_nonempty(&[
        st.tool_call_ids
            .get(&index)
            .map(String::as_str)
            .unwrap_or(""),
        &format!("call_{index}"),
    ]);
    let step_name = st.tool_call_names.get(&index).cloned().unwrap_or_default();
    if st.current_step_type != "function_call" || st.current_step_id != step_id {
        out.extend(append_interactions_step_stop(st));
        let mut step = Map::new();
        step.insert("type".into(), Value::String("function_call".into()));
        step.insert("id".into(), Value::String(step_id.clone()));
        step.insert("call_id".into(), Value::String(step_id.clone()));
        step.insert("name".into(), Value::String(step_name));
        step.insert("arguments".into(), json!({}));
        out.extend(append_interactions_created(st, model_name, root));
        out.extend(append_interactions_step_start(
            st,
            "function_call",
            &Value::Object(step),
        ));
    }
    if let Some(function) = tool_call.get("function") {
        if let Some(args) = function.get("arguments").and_then(Value::as_str) {
            if !args.is_empty() {
                out.extend(append_interactions_arguments_delta(st, args));
            }
        }
    }
    out
}

fn append_interactions_created(
    st: &mut OpenAIToInteractionsStreamState,
    model_name: &str,
    root: &Value,
) -> Vec<Vec<u8>> {
    if st.created {
        return Vec::new();
    }
    let interaction_id = first_nonempty(&[
        root.get("id").and_then(Value::as_str).unwrap_or(""),
        &st.id,
        "interaction_ctox",
    ]);
    st.id = interaction_id.clone();
    let mut interaction = Map::new();
    interaction.insert("id".into(), Value::String(interaction_id));
    interaction.insert("status".into(), Value::String("in_progress".into()));
    interaction.insert("object".into(), Value::String("interaction".into()));
    interaction.insert(
        "model".into(),
        Value::String(first_nonempty(&[
            model_name,
            root.get("model").and_then(Value::as_str).unwrap_or(""),
        ])),
    );
    let payload = json!({
        "interaction": Value::Object(interaction),
        "event_type": "interaction.created",
    });
    let mut out = Vec::new();
    out.push(emit_interactions_event("interaction.created", payload));
    st.created = true;
    out.extend(append_interactions_status_update(st));
    out
}

fn append_interactions_status_update(st: &mut OpenAIToInteractionsStreamState) -> Vec<Vec<u8>> {
    if st.status_updated {
        return Vec::new();
    }
    let payload = json!({
        "interaction_id": st.id,
        "status": "in_progress",
        "event_type": "interaction.status_update",
    });
    let out = vec![emit_interactions_event(
        "interaction.status_update",
        payload,
    )];
    st.status_updated = true;
    out
}

fn ensure_interactions_step(
    st: &mut OpenAIToInteractionsStreamState,
    step_type: &str,
    step: &Value,
    model_name: &str,
) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    out.extend(append_interactions_created(st, model_name, step));
    if st.active_step_open && st.current_step_type == step_type {
        return out;
    }
    out.extend(append_interactions_step_stop(st));
    out.extend(append_interactions_step_start(st, step_type, step));
    out
}

fn append_interactions_step_start(
    st: &mut OpenAIToInteractionsStreamState,
    step_type: &str,
    step: &Value,
) -> Vec<Vec<u8>> {
    let index = st.step_index;
    st.step_index += 1;
    st.active_step_index = index;
    st.current_step_type = step_type.to_owned();
    st.active_step_open = true;
    let mut step_obj = Map::new();
    step_obj.insert("type".into(), Value::String(step_type.to_owned()));
    if step_type == "function_call" {
        let id = first_nonempty(&[
            step.get("call_id").and_then(Value::as_str).unwrap_or(""),
            step.get("id").and_then(Value::as_str).unwrap_or(""),
            &st.current_step_id,
        ]);
        st.current_step_id = id.clone();
        if !id.is_empty() {
            step_obj.insert("id".into(), Value::String(id.clone()));
            step_obj.insert("call_id".into(), Value::String(id));
        }
        if let Some(name) = step.get("name").and_then(Value::as_str) {
            step_obj.insert("name".into(), Value::String(name.to_owned()));
        }
        step_obj.insert("arguments".into(), json!({}));
    } else {
        st.current_step_id.clear();
    }
    let payload = json!({
        "index": index,
        "step": Value::Object(step_obj),
        "event_type": "step.start",
    });
    vec![emit_interactions_event("step.start", payload)]
}

fn append_interactions_text_delta(
    st: &OpenAIToInteractionsStreamState,
    text: &str,
    thought: bool,
) -> Vec<Vec<u8>> {
    let payload = if thought {
        json!({
            "index": st.active_step_index,
            "delta": {
                "content": {"text": text, "type":"text"},
                "type":"thought_summary"
            },
            "event_type": "step.delta",
        })
    } else {
        json!({
            "index": st.active_step_index,
            "delta": {"text": text, "type":"text"},
            "event_type": "step.delta",
        })
    };
    vec![emit_interactions_event("step.delta", payload)]
}

fn append_interactions_arguments_delta(
    st: &OpenAIToInteractionsStreamState,
    arguments: &str,
) -> Vec<Vec<u8>> {
    let payload = json!({
        "index": st.active_step_index,
        "delta": {"arguments": arguments, "type":"arguments_delta"},
        "event_type": "step.delta",
    });
    vec![emit_interactions_event("step.delta", payload)]
}

fn append_interactions_step_stop(st: &mut OpenAIToInteractionsStreamState) -> Vec<Vec<u8>> {
    if !st.active_step_open {
        return Vec::new();
    }
    let payload = json!({
        "index": st.active_step_index,
        "event_type": "step.stop",
    });
    st.active_step_open = false;
    st.current_step_type.clear();
    st.current_step_id.clear();
    vec![emit_interactions_event("step.stop", payload)]
}

fn append_interactions_completed(
    st: &mut OpenAIToInteractionsStreamState,
    model_name: &str,
    root: &Value,
) -> Vec<Vec<u8>> {
    if st.completed {
        return Vec::new();
    }
    let mut out = Vec::new();
    if !st.created {
        out.extend(append_interactions_created(st, model_name, root));
    }
    let completed_at = st.completed_at.clone();
    let mut interaction = Map::new();
    interaction.insert("id".into(), Value::String(st.id.clone()));
    interaction.insert("status".into(), Value::String("completed".into()));
    interaction.insert("usage".into(), json!({}));
    interaction.insert("created".into(), Value::String(completed_at.clone()));
    interaction.insert("updated".into(), Value::String(completed_at));
    interaction.insert("service_tier".into(), Value::String("standard".into()));
    interaction.insert("object".into(), Value::String("interaction".into()));
    interaction.insert(
        "model".into(),
        Value::String(first_nonempty(&[
            model_name,
            root.get("model").and_then(Value::as_str).unwrap_or(""),
        ])),
    );
    let usage = root
        .get("usage")
        .cloned()
        .or_else(|| st.usage.clone())
        .unwrap_or(Value::Null);
    let mut interaction_value = Value::Object(interaction);
    set_interactions_usage_from_openai_chat(
        interaction_value
            .as_object_mut()
            .expect("interaction payload is an object"),
        "usage",
        &usage,
    );
    let payload = json!({
        "interaction": interaction_value,
        "event_type": "interaction.completed",
    });
    out.push(emit_interactions_event("interaction.completed", payload));
    st.completed = true;
    out
}

fn append_interactions_done(st: &mut OpenAIToInteractionsStreamState) -> Vec<Vec<u8>> {
    if st.done {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut frame = Vec::new();
    append_sse_event(&mut frame, "done", b"[DONE]", 1);
    out.push(frame);
    st.done = true;
    out
}

fn emit_interactions_event(event: &str, payload: Value) -> Vec<u8> {
    let bytes = serde_json::to_vec(&payload).unwrap_or_default();
    let mut out = Vec::with_capacity(event.len() + bytes.len() + 16);
    append_sse_event(&mut out, event, &bytes, 1);
    out
}

fn openai_chat_sse_payload(raw: &[u8]) -> Vec<u8> {
    let trimmed = raw.trim_ascii();
    if trimmed.is_empty() || trimmed == b"[DONE]" {
        return trimmed.to_vec();
    }
    if let Some(payload) = trimmed.strip_prefix(b"data:") {
        return payload.trim_ascii().to_vec();
    }
    let mut data_lines: Vec<Vec<u8>> = Vec::new();
    for line in trimmed.split(|byte| *byte == b'\n') {
        let trimmed_line = line.trim_ascii();
        if let Some(payload) = trimmed_line.strip_prefix(b"data:") {
            data_lines.push(payload.trim_ascii().to_vec());
        }
    }
    if data_lines.is_empty() {
        trimmed.to_vec()
    } else {
        let total = data_lines.iter().map(Vec::len).sum::<usize>() + data_lines.len() - 1;
        let mut joined = Vec::with_capacity(total);
        for (i, line) in data_lines.iter().enumerate() {
            if i > 0 {
                joined.push(b'\n');
            }
            joined.extend_from_slice(line);
        }
        joined
    }
}

fn interactions_text_step(step_type: &str, text: &str) -> Value {
    json!({
        "type": step_type,
        "content": [{"type":"text","text":text}],
    })
}

fn append_step(out: &mut Map<String, Value>, step: Value) {
    let steps = out
        .entry("steps")
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Some(array) = steps.as_array_mut() {
        array.push(step);
    }
}

pub(super) fn set_interactions_usage_from_openai_chat(
    out: &mut Map<String, Value>,
    path: &str,
    usage: &Value,
) {
    if usage == &Value::Null {
        return;
    }
    let mut path_root: Map<String, Value> = out
        .get(path)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(value) = usage.get("prompt_tokens").and_then(Value::as_i64) {
        path_root.insert("input_tokens".into(), value.into());
        path_root.insert("total_input_tokens".into(), value.into());
    }
    if let Some(value) = usage.get("completion_tokens").and_then(Value::as_i64) {
        path_root.insert("output_tokens".into(), value.into());
        path_root.insert("total_output_tokens".into(), value.into());
    }
    if let Some(value) = usage.get("total_tokens").and_then(Value::as_i64) {
        path_root.insert("total_tokens".into(), value.into());
    }
    if let Some(value) = usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_i64)
    {
        path_root.insert("cached_tokens".into(), value.into());
        path_root.insert("total_cached_tokens".into(), value.into());
    }
    if let Some(value) = usage
        .pointer("/completion_tokens_details/reasoning_tokens")
        .and_then(Value::as_i64)
    {
        path_root.insert("reasoning_tokens".into(), value.into());
        path_root.insert("total_thought_tokens".into(), value.into());
    }
    if !path_root.is_empty() {
        out.insert(path.into(), Value::Object(path_root));
    }
}

fn request_digest(model: &str, original_request: &[u8], request: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(model.as_bytes());
    hash.update([0]);
    hash.update(original_request);
    hash.update([0]);
    hash.update(request);
    hash.finalize().into()
}

fn deterministic_id(prefix: &str, model: &str, original_request: &[u8], request: &[u8]) -> String {
    let suffix = request_digest(model, original_request, request)
        .iter()
        .take(12)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{prefix}_{suffix}")
}

fn deterministic_timestamp(model: &str, original_request: &[u8], request: &[u8]) -> String {
    let digest = request_digest(model, original_request, request);
    let entropy = u64::from_be_bytes(digest[..8].try_into().expect("eight digest bytes"));
    // Keep synthesized request-local timestamps within 2020-2029. Native
    // upstream timestamps always win when a future wire shape exposes them.
    let seconds = 1_577_836_800 + entropy % 315_576_000;
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

// Re-export the upstream `interactionsUsage` lookup so callers can branch on
// `metadata.total_usage` for the `finish` event path.
pub fn openai_chat_interactions_usage(root: &Value) -> Option<&Value> {
    interactions_usage(root)
}
