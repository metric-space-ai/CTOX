// ref: internal/translator/codex/gemini/codex_gemini_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;

use chrono::{DateTime, SecondsFormat, Utc};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::internal::translator::common::gemini_token_count_json;

use super::codex_gemini_request::build_short_name_map;

#[derive(Debug, Default)]
pub struct CodexToGeminiStreamState {
    model: String,
    created_at: i64,
    response_id: String,
    last_storage_output: Option<Vec<u8>>,
    has_output_text_delta: bool,
    last_image_hash_by_id: HashMap<String, [u8; 32]>,
}

pub fn convert_codex_response_to_gemini_stream(
    model_name: &str,
    original_request: &[u8],
    raw: &[u8],
    state: &mut CodexToGeminiStreamState,
) -> Vec<Vec<u8>> {
    if state.model.is_empty() {
        state.model = model_name.to_owned();
    }
    let Some(payload) = strip_data_prefix(raw) else {
        return Vec::new();
    };
    let Ok(event) = serde_json::from_slice::<Value>(payload) else {
        return Vec::new();
    };
    let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
    if let Some(created_at) = event
        .pointer("/response/created_at")
        .and_then(Value::as_i64)
    {
        state.created_at = created_at;
    }
    let mut response = stream_template(state);

    if kind == "response.image_generation_call.partial_image" {
        return image_event(&event, "item_id", "partial_image_b64", state, response);
    }
    if kind == "response.output_item.done" {
        let item = event.get("item").unwrap_or(&Value::Null);
        match item.get("type").and_then(Value::as_str).unwrap_or("") {
            "image_generation_call" => return image_event(item, "id", "result", state, response),
            "function_call" => {
                response["candidates"][0]["content"]["parts"] =
                    Value::Array(vec![function_call(item, original_request)]);
                response["candidates"][0]["finishReason"] = Value::String("STOP".to_owned());
                state.last_storage_output = serde_json::to_vec(&response).ok();
                return Vec::new();
            }
            "message" if !state.has_output_text_delta => {
                let parts = item
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .filter(|text| !text.is_empty())
                    .map(|text| json!({"text":text}))
                    .collect::<Vec<_>>();
                if parts.is_empty() {
                    return Vec::new();
                }
                state.has_output_text_delta = true;
                response["candidates"][0]["content"]["parts"] = Value::Array(parts);
                return encode_many([response]);
            }
            _ => return Vec::new(),
        }
    }

    match kind {
        "response.created" => {
            if let Some(model) = event.pointer("/response/model").and_then(Value::as_str) {
                response["modelVersion"] = Value::String(model.to_owned());
            }
            if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                state.response_id = id.to_owned();
                response["responseId"] = Value::String(id.to_owned());
            }
        }
        "response.reasoning_summary_text.delta" => {
            response["candidates"][0]["content"]["parts"] = json!([{
                "thought":true,"text":event.get("delta").and_then(Value::as_str).unwrap_or("")
            }]);
        }
        "response.output_text.delta" => {
            state.has_output_text_delta = true;
            response["candidates"][0]["content"]["parts"] = json!([{
                "text":event.get("delta").and_then(Value::as_str).unwrap_or("")
            }]);
        }
        "response.completed" | "response.incomplete" => {
            let input = event
                .pointer("/response/usage/input_tokens")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let output = event
                .pointer("/response/usage/output_tokens")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            response["usageMetadata"]["promptTokenCount"] = Value::from(input);
            response["usageMetadata"]["candidatesTokenCount"] = Value::from(output);
            response["usageMetadata"]["totalTokenCount"] = Value::from(input + output);
            if kind == "response.incomplete" {
                response["candidates"][0]["finishReason"] = Value::String(
                    finish_reason(
                        event
                            .pointer("/response/incomplete_details/reason")
                            .and_then(Value::as_str)
                            .unwrap_or(""),
                    )
                    .to_owned(),
                );
            }
        }
        _ => return Vec::new(),
    }
    let mut outputs = Vec::new();
    if let Some(stored) = state.last_storage_output.take() {
        outputs.push(stored);
    }
    outputs.extend(encode_many([response]));
    outputs
}

pub fn convert_codex_response_to_gemini_non_stream(
    model_name: &str,
    original_request: &[u8],
    raw: &[u8],
) -> Vec<u8> {
    let Ok(event) = serde_json::from_slice::<Value>(raw) else {
        return Vec::new();
    };
    let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
    if !matches!(kind, "response.completed" | "response.incomplete") {
        return Vec::new();
    }
    let response_data = event.get("response").unwrap_or(&Value::Null);
    let mut response = json!({
        "candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],
        "usageMetadata":{"trafficType":"PROVISIONED_THROUGHPUT"},
        "modelVersion":model_name,"createTime":"","responseId":""
    });
    if kind == "response.incomplete" {
        response["candidates"][0]["finishReason"] = Value::String(
            finish_reason(
                response_data
                    .pointer("/incomplete_details/reason")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            )
            .to_owned(),
        );
    }
    if let Some(id) = response_data.get("id").and_then(Value::as_str) {
        response["responseId"] = Value::String(id.to_owned());
    }
    if let Some(created) = response_data.get("created_at").and_then(Value::as_i64) {
        response["createTime"] = Value::String(format_time(created));
    }
    let input = response_data
        .pointer("/usage/input_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let output = response_data
        .pointer("/usage/output_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    response["usageMetadata"]["promptTokenCount"] = Value::from(input);
    response["usageMetadata"]["candidatesTokenCount"] = Value::from(output);
    response["usageMetadata"]["totalTokenCount"] = Value::from(input + output);

    let mut parts = Vec::new();
    for item in response_data
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match item.get("type").and_then(Value::as_str).unwrap_or("") {
            "reasoning" => {
                if let Some(content) = item.get("content").and_then(Value::as_str) {
                    parts.push(json!({"text":content,"thought":true}));
                }
            }
            "message" => parts.extend(
                item.get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .map(|text| json!({"text":text})),
            ),
            "image_generation_call" => {
                if let Some(data) = item.get("result").and_then(Value::as_str) {
                    if !data.is_empty() {
                        parts.push(json!({"inlineData":{"data":data,"mimeType":mime_type(item.get("output_format").and_then(Value::as_str).unwrap_or(""))}}));
                    }
                }
            }
            "function_call" => parts.push(function_call(item, original_request)),
            _ => {}
        }
    }
    response["candidates"][0]["content"]["parts"] = Value::Array(parts);
    serde_json::to_vec(&response).unwrap_or_default()
}

pub fn gemini_token_count(count: i64) -> Vec<u8> {
    gemini_token_count_json(count)
}

fn stream_template(state: &CodexToGeminiStreamState) -> Value {
    json!({
        "candidates":[{"content":{"role":"model","parts":[]}}],
        "usageMetadata":{"trafficType":"PROVISIONED_THROUGHPUT"},
        "modelVersion":state.model,
        "createTime":if state.created_at == 0 {String::new()} else {format_time(state.created_at)},
        "responseId":state.response_id
    })
}

fn image_event(
    value: &Value,
    id_key: &str,
    data_key: &str,
    state: &mut CodexToGeminiStreamState,
    mut response: Value,
) -> Vec<Vec<u8>> {
    let data = value.get(data_key).and_then(Value::as_str).unwrap_or("");
    if data.is_empty() {
        return Vec::new();
    }
    let id = value.get(id_key).and_then(Value::as_str).unwrap_or("");
    if !id.is_empty() {
        let hash: [u8; 32] = Sha256::digest(data.as_bytes()).into();
        if state.last_image_hash_by_id.get(id) == Some(&hash) {
            return Vec::new();
        }
        state.last_image_hash_by_id.insert(id.to_owned(), hash);
    }
    response["candidates"][0]["content"]["parts"] = json!([{
        "inlineData":{"data":data,"mimeType":mime_type(value.get("output_format").and_then(Value::as_str).unwrap_or(""))}
    }]);
    encode_many([response])
}

fn function_call(item: &Value, original_request: &[u8]) -> Value {
    let name = item.get("name").and_then(Value::as_str).unwrap_or("");
    let name = reverse_names(original_request)
        .get(name)
        .cloned()
        .unwrap_or_else(|| name.to_owned());
    let arguments = item
        .get("arguments")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}));
    let mut result = json!({"functionCall":{"name":name,"args":arguments}});
    if let Some(id) = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        result["functionCall"]["id"] = Value::String(id.to_owned());
    }
    result
}

fn reverse_names(original: &[u8]) -> HashMap<String, String> {
    let root = serde_json::from_slice::<Value>(original).unwrap_or(Value::Null);
    let names = root
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|tool| {
            tool.get("functionDeclarations")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|function| function.get("name").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    build_short_name_map(&names)
        .into_iter()
        .map(|(original, short)| (short, original))
        .collect()
}

fn strip_data_prefix(raw: &[u8]) -> Option<&[u8]> {
    raw.strip_prefix(b"data:")
        .map(|payload| payload.strip_prefix(b" ").unwrap_or(payload))
        .map(|payload| {
            let start = payload
                .iter()
                .position(|byte| !byte.is_ascii_whitespace())
                .unwrap_or(payload.len());
            let end = payload
                .iter()
                .rposition(|byte| !byte.is_ascii_whitespace())
                .map_or(start, |index| index + 1);
            &payload[start..end]
        })
}

fn finish_reason(reason: &str) -> &'static str {
    match reason {
        "max_tokens" | "max_output_tokens" => "MAX_TOKENS",
        "content_filter" => "SAFETY",
        _ => "OTHER",
    }
}

fn mime_type(format: &str) -> &str {
    if format.contains('/') {
        return format;
    }
    match format.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "image/png",
    }
}

fn format_time(timestamp: i64) -> String {
    DateTime::<Utc>::from_timestamp(timestamp, 0)
        .map(|value| value.to_rfc3339_opts(SecondsFormat::Nanos, true))
        .unwrap_or_default()
}

fn encode_many(values: impl IntoIterator<Item = Value>) -> Vec<Vec<u8>> {
    values
        .into_iter()
        .filter_map(|value| serde_json::to_vec(&value).ok())
        .collect()
}
