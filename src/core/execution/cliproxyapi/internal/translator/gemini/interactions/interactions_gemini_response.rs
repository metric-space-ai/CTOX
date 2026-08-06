// ref: internal/translator/gemini/interactions/interactions_gemini_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::internal::translator::antigravity::interactions::{
    convert_antigravity_response_to_interactions,
    convert_antigravity_response_to_interactions_non_stream, AntigravityToInteractionsState,
};

#[derive(Clone, Debug, Default)]
pub struct GeminiToInteractionsState {
    inner: Option<AntigravityToInteractionsState>,
}

impl GeminiToInteractionsState {
    #[must_use]
    pub fn with_identity(id: impl Into<String>, timestamp: impl Into<String>) -> Self {
        Self {
            inner: Some(AntigravityToInteractionsState::with_identity(id, timestamp)),
        }
    }

    fn inner(
        &mut self,
        original: &[u8],
        request: &[u8],
        raw: &[u8],
    ) -> &mut AntigravityToInteractionsState {
        self.inner.get_or_insert_with(|| {
            AntigravityToInteractionsState::with_identity(
                synthesized_id(b"stream", original, request, raw),
                "1970-01-01T00:00:00Z",
            )
        })
    }
}

pub fn convert_gemini_response_to_interactions_stream(
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
    state: &mut GeminiToInteractionsState,
) -> Vec<Vec<u8>> {
    convert_antigravity_response_to_interactions(
        model_name,
        original_request,
        request,
        raw,
        state.inner(original_request, request, raw),
    )
}

pub fn convert_gemini_response_to_interactions_non_stream(
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
) -> Vec<u8> {
    let mut root = serde_json::from_slice::<Value>(raw).unwrap_or(Value::Null);
    if root
        .get("responseId")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        root["responseId"] =
            Value::String(synthesized_id(b"nonstream", original_request, request, raw));
    }
    let normalized = serde_json::to_vec(&root).unwrap_or_default();
    convert_antigravity_response_to_interactions_non_stream(
        model_name,
        original_request,
        request,
        &normalized,
    )
}

#[derive(Clone, Debug, Default)]
pub struct InteractionsToGeminiState {
    response_id: String,
    model: String,
    step_names: HashMap<usize, String>,
}

pub fn convert_interactions_response_to_gemini_stream(
    model_name: &str,
    raw: &[u8],
    state: &mut InteractionsToGeminiState,
) -> Vec<Vec<u8>> {
    let payload = sse_payload(raw);
    let root = serde_json::from_slice::<Value>(payload).unwrap_or(Value::Null);
    let event = root.get("event_type").and_then(Value::as_str).unwrap_or("");
    match event {
        "interaction.created" => {
            state.response_id = root
                .pointer("/interaction/id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            state.model = root
                .pointer("/interaction/model")
                .and_then(Value::as_str)
                .unwrap_or(model_name)
                .to_owned();
            Vec::new()
        }
        "step.start" => {
            let index = root
                .get("index")
                .or_else(|| root.get("step_index"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            let name = root
                .pointer("/step/name")
                .or_else(|| root.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("");
            state.step_names.insert(index, name.to_owned());
            Vec::new()
        }
        "step.delta" => {
            let delta = root.get("delta").unwrap_or(&Value::Null);
            let kind = delta.get("type").and_then(Value::as_str).unwrap_or("");
            let part = match kind {
                "text_delta" => {
                    json!({"text":delta.get("text").cloned().unwrap_or(Value::String(String::new()))})
                }
                "thought_delta" | "thinking_delta" => {
                    json!({"text":delta.get("text").or_else(|| delta.get("thinking")).cloned().unwrap_or(Value::String(String::new())),"thought":true})
                }
                "function_call_delta" | "input_json_delta" => {
                    let index = root
                        .get("index")
                        .or_else(|| root.get("step_index"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    let name = state.step_names.get(&index).cloned().unwrap_or_default();
                    let args = delta
                        .get("arguments")
                        .or_else(|| delta.get("partial_json"))
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    json!({"functionCall":{"name":name,"args":args}})
                }
                _ => return Vec::new(),
            };
            vec![gemini_chunk(state, model_name, vec![part], None, None)]
        }
        "interaction.completed" | "finish" => {
            let usage = root
                .pointer("/interaction/usage")
                .or_else(|| root.get("usage"));
            vec![gemini_chunk(
                state,
                model_name,
                Vec::new(),
                Some("STOP"),
                usage,
            )]
        }
        _ => Vec::new(),
    }
}

pub fn convert_interactions_response_to_gemini_non_stream(model_name: &str, raw: &[u8]) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(raw).unwrap_or(Value::Null);
    let interaction = root.get("interaction").unwrap_or(&root);
    let mut parts = Vec::new();
    for step in interaction
        .get("steps")
        .or_else(|| root.get("steps"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        parts.extend(step_parts(step));
    }
    let mut state = InteractionsToGeminiState {
        response_id: interaction
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        model: interaction
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(model_name)
            .to_owned(),
        ..InteractionsToGeminiState::default()
    };
    if state.response_id.is_empty() {
        state.response_id = synthesized_id(b"reverse", b"", b"", raw);
    }
    gemini_chunk(
        &state,
        model_name,
        parts,
        Some("STOP"),
        interaction.get("usage"),
    )
}

fn step_parts(step: &Value) -> Vec<Value> {
    match step.get("type").and_then(Value::as_str).unwrap_or("") {
        "function_call" => vec![json!({"functionCall":{
            "name":step.get("name").cloned().unwrap_or(Value::String(String::new())),
            "id":step.get("call_id").cloned().unwrap_or(Value::String(String::new())),
            "args":step.get("arguments").cloned().unwrap_or_else(|| json!({}))
        }})],
        "function_result" => vec![json!({"functionResponse":{
            "name":step.get("name").cloned().unwrap_or(Value::String(String::new())),
            "id":step.get("call_id").cloned().unwrap_or(Value::String(String::new())),
            "response":step.get("result").cloned().unwrap_or_else(|| json!({}))
        }})],
        kind => step
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| {
                item.get("text")
                    .map(|text| json!({"text":text,"thought":kind == "thought"}))
            })
            .collect(),
    }
}

fn gemini_chunk(
    state: &InteractionsToGeminiState,
    fallback_model: &str,
    parts: Vec<Value>,
    finish: Option<&str>,
    usage: Option<&Value>,
) -> Vec<u8> {
    let model = if state.model.is_empty() {
        fallback_model
    } else {
        &state.model
    };
    let mut out = json!({
        "responseId":state.response_id,"modelVersion":model,
        "candidates":[{"index":0,"content":{"role":"model","parts":parts}}]
    });
    if let Some(finish) = finish {
        out["candidates"][0]["finishReason"] = Value::String(finish.to_owned());
    }
    if let Some(usage) = usage {
        out["usageMetadata"] = json!({
            "promptTokenCount":usage.get("input_tokens").cloned().unwrap_or(Value::from(0)),
            "candidatesTokenCount":usage.get("output_tokens").cloned().unwrap_or(Value::from(0)),
            "totalTokenCount":usage.get("total_tokens").cloned().unwrap_or(Value::from(0))
        });
    }
    serde_json::to_vec(&out).unwrap_or_default()
}

fn synthesized_id(tag: &[u8], original: &[u8], request: &[u8], raw: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"ctox-gemini-interactions-v1\0");
    for value in [tag, original, request, raw] {
        digest.update((value.len() as u64).to_be_bytes());
        digest.update(value);
    }
    let digest = digest.finalize();
    format!(
        "interaction_{}",
        digest[..16]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn sse_payload(raw: &[u8]) -> &[u8] {
    let trimmed = trim_ascii(raw);
    trimmed
        .strip_prefix(b"data:")
        .map(trim_ascii)
        .unwrap_or(trimmed)
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
