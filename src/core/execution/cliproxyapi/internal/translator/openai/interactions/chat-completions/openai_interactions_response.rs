// ref: internal/translator/openai/interactions/chat-completions/openai_interactions_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

//! Interactions response body -> OpenAI chat-completions response body.
//!
//! Mirrors upstream's gjson/sjson byte-splice semantics on a typed
//! `serde_json::Value` builder. The stateful stream transformation uses the
//! shared `internal::translator::common::append_sse_event` helper to keep
//! the SSE envelope stable across leaves.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use super::interactions_openai_response::openai_chat_interactions_usage;
use crate::internal::translator::common::interactions_usage;
use crate::sdk::translator::{TranslationContext, TranslationState};

/// Request-local state carried across stream chunks for one Interactions
/// stream being projected back to OpenAI chat-completions.
#[derive(Debug, Default)]
pub struct InteractionsToOpenAIChatStreamState {
    pub id: String,
    pub model: String,
    pub created: bool,
    pub started: bool,
    pub completed: bool,
    pub saw_tool_call: bool,
    pub step_types: HashMap<i64, String>,
    pub tool_ids: HashMap<i64, String>,
    pub tool_names: HashMap<i64, String>,
    pub tool_arguments: HashMap<i64, String>,
    pub text_by_step_index: HashMap<i64, String>,
    pub created_emitted: bool,
    pub finished: bool,
    pub done_emitted: bool,
    pub created_at: i64,
}

pub fn convert_interactions_response_to_openai(
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
    if state.model.is_empty() && !model_name.is_empty() {
        state.model = model_name.to_owned();
    }
    if state.id.is_empty() {
        state.id = deterministic_id("chatcmpl", model_name, original_request, request);
    }
    if state.created_at == 0 {
        state.created_at = deterministic_unix_seconds(model_name, original_request, request);
    }
    convert_interactions_event_to_openai_chat(raw_json, state)
}

pub fn convert_interactions_response_to_openai_non_stream(
    _context: &TranslationContext,
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw_json: &[u8],
    _state: &mut TranslationState,
) -> Vec<u8> {
    let root: Value = serde_json::from_slice(raw_json).unwrap_or(Value::Null);
    let interaction = root.get("interaction").unwrap_or(&root);
    let mut out = Map::new();
    out.insert("id".into(), Value::String(String::new()));
    out.insert("object".into(), Value::String("chat.completion".into()));
    out.insert(
        "created".into(),
        Value::from(deterministic_unix_seconds(
            model_name,
            original_request,
            request,
        )),
    );
    out.insert("model".into(), Value::String(String::new()));
    out.insert(
        "choices".into(),
        Value::Array(vec![json!({
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop",
        })]),
    );
    out["id"] = Value::String(first_nonempty(&[
        interaction.get("id").and_then(Value::as_str).unwrap_or(""),
        root.get("id").and_then(Value::as_str).unwrap_or(""),
        &deterministic_id("chatcmpl", model_name, original_request, request),
    ]));
    out["model"] = Value::String(first_nonempty(&[
        interaction
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(""),
        model_name,
    ]));
    let steps = interaction
        .get("steps")
        .or_else(|| root.get("steps"))
        .and_then(Value::as_array);
    let mut text_builder = String::new();
    let mut reasoning_builder = String::new();
    let mut saw_tool_call = false;
    if let Some(steps) = steps {
        for step in steps {
            match step.get("type").and_then(Value::as_str).unwrap_or_default() {
                "model_output" => {
                    for text in interactions_content_texts_for_openai_chat(step.get("content")) {
                        text_builder.push_str(&text);
                    }
                }
                "thought" => {
                    for text in interactions_content_texts_for_openai_chat(step.get("content")) {
                        reasoning_builder.push_str(&text);
                    }
                }
                "function_call" => {
                    saw_tool_call = true;
                    let tool_call = openai_chat_tool_call_from_interactions(step, &Value::Null);
                    let choices = out
                        .entry("choices")
                        .or_insert_with(|| Value::Array(Vec::new()));
                    if let Some(items) = choices.as_array_mut() {
                        if let Some(choice) = items.get_mut(0) {
                            if let Some(object) = choice.as_object_mut() {
                                let calls = object
                                    .entry("message")
                                    .or_insert_with(|| json!({"role":"assistant"}));
                                let calls_object =
                                    calls.as_object_mut().expect("message payload is an object");
                                let calls = calls_object
                                    .entry("tool_calls")
                                    .or_insert_with(|| Value::Array(Vec::new()));
                                if let Some(arr) = calls.as_array_mut() {
                                    arr.push(tool_call);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    if !text_builder.is_empty() {
        if let Some(choice) = out
            .get_mut("choices")
            .and_then(|value| value.as_array_mut())
            .and_then(|items| items.get_mut(0))
            .and_then(|item| item.as_object_mut())
        {
            let message = choice
                .entry("message")
                .or_insert_with(|| json!({"role":"assistant"}));
            if let Some(object) = message.as_object_mut() {
                object.insert("content".into(), Value::String(text_builder));
            }
        }
    }
    if !reasoning_builder.is_empty() {
        if let Some(choice) = out
            .get_mut("choices")
            .and_then(|value| value.as_array_mut())
            .and_then(|items| items.get_mut(0))
            .and_then(|item| item.as_object_mut())
        {
            let message = choice
                .entry("message")
                .or_insert_with(|| json!({"role":"assistant"}));
            if let Some(object) = message.as_object_mut() {
                object.insert("reasoning_content".into(), Value::String(reasoning_builder));
            }
        }
    }
    if saw_tool_call {
        if let Some(choice) = out
            .get_mut("choices")
            .and_then(|value| value.as_array_mut())
            .and_then(|items| items.get_mut(0))
            .and_then(|item| item.as_object_mut())
        {
            let message = choice
                .entry("message")
                .or_insert_with(|| json!({"role":"assistant"}));
            if let Some(object) = message.as_object_mut() {
                object.insert("content".into(), Value::Null);
            }
            choice.insert("finish_reason".into(), Value::String("tool_calls".into()));
        }
    }
    set_openai_chat_usage_from_interactions(
        &mut out,
        "usage",
        openai_chat_interactions_usage(&root),
    );
    serde_json::to_vec(&Value::Object(out)).unwrap_or_default()
}

fn stream_state(state: &mut TranslationState) -> &mut InteractionsToOpenAIChatStreamState {
    let replace = state
        .as_ref()
        .is_none_or(|value| !value.is::<InteractionsToOpenAIChatStreamState>());
    if replace {
        *state = Some(Box::new(InteractionsToOpenAIChatStreamState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<InteractionsToOpenAIChatStreamState>())
        .expect("Interactions->OpenAI state was initialized with the expected type")
}

fn convert_interactions_event_to_openai_chat(
    raw_json: &[u8],
    st: &mut InteractionsToOpenAIChatStreamState,
) -> Vec<Vec<u8>> {
    let payload = openai_chat_interactions_payload(raw_json);
    if payload.is_empty() || payload.trim_ascii() == b"[DONE]" {
        return Vec::new();
    }
    let Ok(root) = serde_json::from_slice::<Value>(&payload) else {
        return Vec::new();
    };
    let chunks: Vec<Value> = match root
        .get("event_type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "interaction.created" => {
            let interaction = root.get("interaction").unwrap_or(&Value::Null);
            if let Some(id) = interaction.get("id").and_then(Value::as_str) {
                if !id.is_empty() {
                    st.id = id.to_owned();
                }
            }
            if let Some(model) = interaction.get("model").and_then(Value::as_str) {
                if !model.is_empty() {
                    st.model = model.to_owned();
                }
            }
            ensure_openai_chat_started(st)
        }
        "step.start" => interactions_step_start_to_openai_chat(&root, st),
        "step.delta" => interactions_step_delta_to_openai_chat(&root, st),
        "interaction.completed" | "finish" => append_openai_chat_completed(&root, st),
        "done" => Vec::new(),
        _ => Vec::new(),
    };
    chunks.into_iter().map(serialize_value).collect()
}

fn interactions_step_start_to_openai_chat(
    root: &Value,
    st: &mut InteractionsToOpenAIChatStreamState,
) -> Vec<Value> {
    let mut out = ensure_openai_chat_started(st);
    let index = root.get("index").and_then(Value::as_i64).unwrap_or(0);
    let step = root.get("step").unwrap_or(&Value::Null);
    let step_type = step.get("type").and_then(Value::as_str).unwrap_or_default();
    st.step_types.insert(index, step_type.to_owned());
    match step_type {
        "function_call" => {
            st.saw_tool_call = true;
            st.tool_ids.insert(
                index,
                first_nonempty(&[
                    step.get("call_id").and_then(Value::as_str).unwrap_or(""),
                    step.get("id").and_then(Value::as_str).unwrap_or(""),
                    &format!("call_{index}"),
                ]),
            );
            if let Some(name) = step.get("name").and_then(Value::as_str) {
                st.tool_names.insert(index, name.to_owned());
            }
            if let Some(args) = step.get("arguments") {
                let rendered = openai_chat_tool_arguments_text(args);
                if !rendered.is_empty() && rendered != "{}" {
                    st.tool_arguments.insert(index, rendered);
                }
            }
            out.push(openai_chat_tool_call_start_chunk(st, index));
            out
        }
        _ => out,
    }
}

fn interactions_step_delta_to_openai_chat(
    root: &Value,
    st: &mut InteractionsToOpenAIChatStreamState,
) -> Vec<Value> {
    let mut out = ensure_openai_chat_started(st);
    let index = root.get("index").and_then(Value::as_i64).unwrap_or(0);
    let delta = root.get("delta").unwrap_or(&Value::Null);
    match delta
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "thought_summary" => {
            let text = first_nonempty(&[
                delta
                    .pointer("/content/text")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                delta.get("text").and_then(Value::as_str).unwrap_or(""),
            ]);
            if text.is_empty() {
                return out;
            }
            out.push(openai_chat_delta_chunk(st, "reasoning_content", &text));
            out
        }
        "arguments_delta" => {
            let args = delta.get("arguments").and_then(Value::as_str).unwrap_or("");
            let entry = st.tool_arguments.entry(index).or_default();
            entry.push_str(args);
            out.push(openai_chat_tool_call_arguments_chunk(st, index, args));
            out
        }
        _ => {
            let text = delta.get("text").and_then(Value::as_str).unwrap_or("");
            if text.is_empty() {
                return out;
            }
            let entry = st.text_by_step_index.entry(index).or_default();
            entry.push_str(text);
            out.push(openai_chat_delta_chunk(st, "content", text));
            out
        }
    }
}

fn ensure_openai_chat_started(st: &mut InteractionsToOpenAIChatStreamState) -> Vec<Value> {
    if st.started {
        return Vec::new();
    }
    let mut chunk = openai_chat_base_chunk(st);
    if let Some(choice) = chunk
        .pointer_mut("/choices/0")
        .and_then(Value::as_object_mut)
    {
        let delta = choice
            .entry("delta")
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(delta_obj) = delta.as_object_mut() {
            delta_obj.insert("role".into(), Value::String("assistant".into()));
        }
    }
    st.started = true;
    vec![chunk]
}

fn append_openai_chat_completed(
    root: &Value,
    st: &mut InteractionsToOpenAIChatStreamState,
) -> Vec<Value> {
    if st.completed {
        return Vec::new();
    }
    let mut out = ensure_openai_chat_started(st);
    let mut chunk = openai_chat_base_chunk(st);
    let finish_reason = if st.saw_tool_call {
        "tool_calls"
    } else {
        "stop"
    };
    if let Some(choice) = chunk
        .pointer_mut("/choices/0")
        .and_then(Value::as_object_mut)
    {
        choice.insert("finish_reason".into(), Value::String(finish_reason.into()));
    }
    set_openai_chat_usage_from_interactions(
        chunk.as_object_mut().expect("chunk is an object"),
        "usage",
        interactions_usage(root),
    );
    st.completed = true;
    out.push(chunk);
    out
}

fn openai_chat_base_chunk(st: &InteractionsToOpenAIChatStreamState) -> Value {
    let mut chunk = Map::new();
    chunk.insert("id".into(), Value::String(String::new()));
    chunk.insert(
        "object".into(),
        Value::String("chat.completion.chunk".into()),
    );
    chunk.insert("created".into(), Value::from(st.created_at));
    chunk.insert("model".into(), Value::String(String::new()));
    chunk.insert(
        "choices".into(),
        Value::Array(vec![json!({
            "index": 0,
            "delta": {},
            "finish_reason": Value::Null,
        })]),
    );
    let mut value = Value::Object(chunk);
    if let Some(object) = value.as_object_mut() {
        object["id"] = Value::String(first_nonempty(&[&st.id, "chatcmpl_ctox"]));
        object["created"] = Value::from(st.created_at);
        object["model"] = Value::String(st.model.clone());
    }
    value
}

fn openai_chat_delta_chunk(
    st: &InteractionsToOpenAIChatStreamState,
    field: &str,
    value: &str,
) -> Value {
    let mut chunk = openai_chat_base_chunk(st);
    if let Some(choice) = chunk
        .pointer_mut("/choices/0")
        .and_then(Value::as_object_mut)
    {
        let delta = choice
            .entry("delta")
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(delta_obj) = delta.as_object_mut() {
            delta_obj.insert(field.into(), Value::String(value.to_owned()));
        }
    }
    chunk
}

fn openai_chat_tool_call_start_chunk(
    st: &InteractionsToOpenAIChatStreamState,
    index: i64,
) -> Value {
    let mut chunk = openai_chat_base_chunk(st);
    let mut tool_call = Map::new();
    tool_call.insert("index".into(), Value::from(index));
    tool_call.insert("id".into(), Value::String(String::new()));
    tool_call.insert("type".into(), Value::String("function".into()));
    let mut function = Map::new();
    function.insert("name".into(), Value::String(String::new()));
    function.insert("arguments".into(), Value::String(String::new()));
    tool_call.insert("function".into(), Value::Object(function));
    if let Some(choice) = chunk
        .pointer_mut("/choices/0")
        .and_then(Value::as_object_mut)
    {
        let delta = choice
            .entry("delta")
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(delta_obj) = delta.as_object_mut() {
            let calls = delta_obj
                .entry("tool_calls")
                .or_insert_with(|| Value::Array(Vec::new()));
            if let Some(arr) = calls.as_array_mut() {
                let id = first_nonempty(&[
                    st.tool_ids.get(&index).map(String::as_str).unwrap_or(""),
                    &format!("call_{index}"),
                ]);
                if let Some(object) = tool_call.get_mut("id") {
                    *object = Value::String(id);
                }
                if let Some(name) = st.tool_names.get(&index) {
                    if let Some(function) =
                        tool_call.get_mut("function").and_then(Value::as_object_mut)
                    {
                        function.insert("name".into(), Value::String(name.clone()));
                    }
                }
                arr.push(Value::Object(tool_call));
            }
        }
    }
    chunk
}

fn openai_chat_tool_call_arguments_chunk(
    st: &InteractionsToOpenAIChatStreamState,
    index: i64,
    arguments: &str,
) -> Value {
    let mut chunk = openai_chat_base_chunk(st);
    let mut tool_call = Map::new();
    tool_call.insert("index".into(), Value::from(index));
    let mut function = Map::new();
    function.insert("arguments".into(), Value::String(arguments.to_owned()));
    tool_call.insert("function".into(), Value::Object(function));
    if let Some(choice) = chunk
        .pointer_mut("/choices/0")
        .and_then(Value::as_object_mut)
    {
        let delta = choice
            .entry("delta")
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(delta_obj) = delta.as_object_mut() {
            let calls = delta_obj
                .entry("tool_calls")
                .or_insert_with(|| Value::Array(Vec::new()));
            if let Some(arr) = calls.as_array_mut() {
                arr.push(Value::Object(tool_call));
            }
        }
    }
    chunk
}

fn openai_chat_tool_call_from_interactions(step: &Value, fallback_args: &Value) -> Value {
    let mut tool_call = Map::new();
    tool_call.insert("id".into(), Value::String(String::new()));
    tool_call.insert("type".into(), Value::String("function".into()));
    let mut function = Map::new();
    function.insert("name".into(), Value::String(String::new()));
    function.insert("arguments".into(), Value::String("{}".into()));
    tool_call.insert("function".into(), Value::Object(function));
    let call_id = first_nonempty(&[
        step.get("call_id").and_then(Value::as_str).unwrap_or(""),
        step.get("id").and_then(Value::as_str).unwrap_or(""),
        "call_0",
    ]);
    if let Some(object) = tool_call.get_mut("id") {
        *object = Value::String(call_id);
    }
    if let Some(object) = tool_call.get_mut("function").and_then(Value::as_object_mut) {
        if let Some(name) = step.get("name").and_then(Value::as_str) {
            object.insert("name".into(), Value::String(name.to_owned()));
        }
        let arguments = step.get("arguments").unwrap_or(fallback_args);
        let rendered = openai_chat_tool_arguments_text(arguments);
        object.insert("arguments".into(), Value::String(rendered));
    }
    Value::Object(tool_call)
}

fn openai_chat_tool_arguments_text(arguments: &Value) -> String {
    match arguments {
        Value::String(text) => text.clone(),
        Value::Null => "{}".into(),
        _ => serde_json::to_string(arguments).unwrap_or_else(|_| "{}".into()),
    }
}

fn set_openai_chat_usage_from_interactions(
    out: &mut Map<String, Value>,
    path: &str,
    usage: Option<&Value>,
) {
    let Some(usage) = usage else { return };
    if usage == &Value::Null {
        return;
    }
    let mut path_root: Map<String, Value> = out
        .get(path)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(value) = interactions_usage_int(usage, &["input_tokens", "total_input_tokens"]) {
        path_root.insert("prompt_tokens".into(), value.into());
    }
    if let Some(value) = interactions_usage_int(usage, &["output_tokens", "total_output_tokens"]) {
        path_root.insert("completion_tokens".into(), value.into());
    }
    if let Some(value) = interactions_usage_int(usage, &["total_tokens"]) {
        path_root.insert("total_tokens".into(), value.into());
    }
    if let Some(value) = interactions_usage_int(usage, &["cached_tokens", "total_cached_tokens"]) {
        path_root.insert(
            "prompt_tokens_details".into(),
            json!({"cached_tokens":value}),
        );
    }
    if let Some(value) =
        interactions_usage_int(usage, &["reasoning_tokens", "total_thought_tokens"])
    {
        path_root.insert(
            "completion_tokens_details".into(),
            json!({"reasoning_tokens":value}),
        );
    }
    if !path_root.is_empty() {
        out.insert(path.into(), Value::Object(path_root));
    }
}

fn interactions_usage_int(root: &Value, paths: &[&str]) -> Option<i64> {
    paths
        .iter()
        .find_map(|path| root.get(path).and_then(Value::as_i64))
}

fn interactions_content_texts_for_openai_chat(content: Option<&Value>) -> Vec<String> {
    let Some(content) = content else {
        return Vec::new();
    };
    if let Some(text) = content.as_str() {
        return vec![text.to_owned()];
    }
    if let Some(items) = content.as_array() {
        return items
            .iter()
            .filter_map(|part| {
                let text = first_nonempty(&[
                    part.get("text").and_then(Value::as_str).unwrap_or(""),
                    part.pointer("/content/text")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                ]);
                (!text.is_empty()).then_some(text)
            })
            .collect();
    }
    Vec::new()
}

fn openai_chat_interactions_payload(raw: &[u8]) -> Vec<u8> {
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

fn deterministic_id(prefix: &str, model: &str, original_request: &[u8], request: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(model.as_bytes());
    hash.update([0]);
    hash.update(original_request);
    hash.update([0]);
    hash.update(request);
    let suffix = hash
        .finalize()
        .iter()
        .take(12)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{prefix}_{suffix}")
}

fn deterministic_unix_seconds(model: &str, original_request: &[u8], request: &[u8]) -> i64 {
    let mut hash = Sha256::new();
    hash.update(model.as_bytes());
    hash.update([0]);
    hash.update(original_request);
    hash.update([0]);
    hash.update(request);
    let digest = hash.finalize();
    let entropy = u64::from_be_bytes(digest[..8].try_into().expect("eight digest bytes"));
    (1_577_836_800 + entropy % 315_576_000) as i64
}

fn serialize_value(value: Value) -> Vec<u8> {
    serde_json::to_vec(&value).unwrap_or_default()
}

pub(super) fn first_nonempty(values: &[&str]) -> String {
    for value in values {
        if !value.trim().is_empty() {
            return (*value).to_owned();
        }
    }
    String::new()
}
