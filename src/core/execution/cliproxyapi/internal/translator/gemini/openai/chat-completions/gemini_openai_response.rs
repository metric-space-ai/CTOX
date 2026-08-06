// ref: internal/translator/gemini/openai/chat-completions/gemini_openai_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;

use chrono::DateTime;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::gemini_openai_request::sanitized_name_map;

#[derive(Clone, Debug, Default)]
pub struct GeminiToChatStreamState {
    created: i64,
    function_index: HashMap<i64, i64>,
    saw_tool: HashMap<i64, bool>,
    finish_reason: HashMap<i64, String>,
    names: HashMap<String, String>,
    call_id_nonce: Option<u128>,
    call_id_sequence: u64,
}

impl GeminiToChatStreamState {
    /// Supplies the request-local identity authority used for generated tool
    /// call IDs. `initial_sequence` matches the counter value before the next
    /// generated ID, so passing zero makes the first suffix `-1`.
    #[must_use]
    pub fn with_call_id_authority(call_id_nonce: u128, initial_sequence: u64) -> Self {
        Self {
            call_id_nonce: Some(call_id_nonce),
            call_id_sequence: initial_sequence,
            ..Self::default()
        }
    }

    fn initialize_call_id_authority(
        &mut self,
        original_request: &[u8],
        request: &[u8],
        first_payload: &[u8],
    ) {
        self.call_id_nonce.get_or_insert_with(|| {
            synthesized_call_id_nonce(original_request, request, first_payload)
        });
    }

    fn next_call_id(&mut self, name: &str) -> String {
        self.call_id_sequence = self.call_id_sequence.saturating_add(1);
        format!(
            "{name}-{}-{}",
            self.call_id_nonce.unwrap_or_default(),
            self.call_id_sequence
        )
    }
}

pub fn convert_gemini_response_to_openai_chat_stream(
    _model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
    state: &mut GeminiToChatStreamState,
) -> Vec<Vec<u8>> {
    state.initialize_call_id_authority(original_request, request, raw);
    if state.names.is_empty() {
        state.names = sanitized_name_map(original_request);
    }
    let payload = raw.strip_prefix(b"data:").map(trim_ascii).unwrap_or(raw);
    if payload == b"[DONE]" {
        return Vec::new();
    }
    let Ok(root) = serde_json::from_slice::<Value>(payload) else {
        return Vec::new();
    };
    if let Some(created) = root
        .get("createTime")
        .and_then(Value::as_str)
        .and_then(parse_time)
    {
        state.created = created;
    }
    let base = stream_shell(&root, state.created);
    let usage_seen = root.get("usageMetadata").is_some();
    let candidates = root.get("candidates").and_then(Value::as_array);
    if candidates.is_none() && usage_seen {
        return encode(base);
    }
    let mut output = Vec::new();
    for candidate in candidates.into_iter().flatten() {
        let index = candidate.get("index").and_then(Value::as_i64).unwrap_or(0);
        if let Some(reason) = candidate.get("finishReason").and_then(Value::as_str) {
            state
                .finish_reason
                .insert(index, reason.to_ascii_uppercase());
        }
        let mut chunk = base.clone();
        chunk["choices"][0]["index"] = Value::from(index);
        let mut role = false;
        let mut tool_calls = Vec::new();
        let mut images = Vec::new();
        for part in candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let has_signature = part
                .get("thoughtSignature")
                .or_else(|| part.get("thought_signature"))
                .and_then(Value::as_str)
                .is_some_and(|value| !value.is_empty());
            let has_payload = part.get("text").is_some()
                || part.get("functionCall").is_some()
                || part.get("inlineData").is_some()
                || part.get("inline_data").is_some();
            if has_signature && !has_payload {
                continue;
            }
            if let Some(text) = part.get("text") {
                role = true;
                let field = if part
                    .get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    "reasoning_content"
                } else {
                    "content"
                };
                chunk["choices"][0]["delta"][field] = Value::String(value_string(text));
            } else if let Some(call) = part.get("functionCall") {
                role = true;
                state.saw_tool.insert(index, true);
                let call_index = *state.function_index.entry(index).or_insert(0);
                state.function_index.insert(index, call_index + 1);
                let native_name = call.get("name").and_then(Value::as_str).unwrap_or("");
                let name = state
                    .names
                    .get(native_name)
                    .cloned()
                    .unwrap_or_else(|| native_name.to_owned());
                let arguments = call
                    .get("args")
                    .map(|value| serde_json::to_string(value).unwrap_or_default())
                    .unwrap_or_default();
                tool_calls.push(json!({
                    "id":state.next_call_id(&name),"index":call_index,"type":"function",
                    "function":{"name":name,"arguments":arguments}
                }));
            } else if let Some(data) = part.get("inlineData").or_else(|| part.get("inline_data")) {
                let bytes = data.get("data").and_then(Value::as_str).unwrap_or("");
                if bytes.is_empty() {
                    continue;
                }
                role = true;
                let mime = data
                    .get("mimeType")
                    .or_else(|| data.get("mime_type"))
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .unwrap_or("image/png");
                images.push(json!({"type":"image_url","index":images.len(),"image_url":{"url":format!("data:{mime};base64,{bytes}")}}));
            }
        }
        if role {
            chunk["choices"][0]["delta"]["role"] = Value::String("assistant".into());
        }
        if !tool_calls.is_empty() {
            chunk["choices"][0]["delta"]["tool_calls"] = Value::Array(tool_calls);
        }
        if !images.is_empty() {
            chunk["choices"][0]["delta"]["images"] = Value::Array(images);
        }
        if usage_seen {
            if let Some(reason) = state.finish_reason.get(&index) {
                let finish = if state.saw_tool.get(&index).copied().unwrap_or(false) {
                    "tool_calls"
                } else if reason == "MAX_TOKENS" {
                    "max_tokens"
                } else {
                    "stop"
                };
                chunk["choices"][0]["finish_reason"] = Value::String(finish.into());
                chunk["choices"][0]["native_finish_reason"] =
                    Value::String(reason.to_ascii_lowercase());
            }
        }
        output.extend(encode(chunk));
    }
    output
}

pub fn convert_gemini_response_to_openai_chat_non_stream(
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
) -> Vec<u8> {
    let mut state = GeminiToChatStreamState::default();
    convert_gemini_response_to_openai_chat_non_stream_with_state(
        original_request,
        request,
        raw,
        &mut state,
    )
}

/// Non-stream converter with explicit request-local call-ID authority.
pub fn convert_gemini_response_to_openai_chat_non_stream_with_state(
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
    state: &mut GeminiToChatStreamState,
) -> Vec<u8> {
    state.initialize_call_id_authority(original_request, request, raw);
    let root = serde_json::from_slice::<Value>(raw).unwrap_or(Value::Null);
    let names = sanitized_name_map(original_request);
    let created = root
        .get("createTime")
        .and_then(Value::as_str)
        .and_then(parse_time)
        .unwrap_or(0);
    let mut output = json!({
        "id":root.get("responseId").and_then(Value::as_str).unwrap_or(""),
        "object":"chat.completion","created":created,
        "model":root.get("modelVersion").and_then(Value::as_str).unwrap_or("model"),
        "choices":[]
    });
    if let Some(usage) = root.get("usageMetadata") {
        output["usage"] = usage_value(usage);
    }
    let mut choices = Vec::new();
    for candidate in root
        .get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let native = candidate
            .get("finishReason")
            .and_then(Value::as_str)
            .map(str::to_ascii_lowercase);
        let mut choice = json!({
            "index":candidate.get("index").and_then(Value::as_i64).unwrap_or(0),
            "message":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":null},
            "finish_reason":native,"native_finish_reason":native
        });
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut tools = Vec::new();
        let mut images = Vec::new();
        for part in candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(value) = part.get("text") {
                if part
                    .get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    reasoning.push_str(&value_string(value));
                } else {
                    text.push_str(&value_string(value));
                }
            } else if let Some(call) = part.get("functionCall") {
                let native_name = call.get("name").and_then(Value::as_str).unwrap_or("");
                let name = names
                    .get(native_name)
                    .map(String::as_str)
                    .unwrap_or(native_name);
                let arguments = call
                    .get("args")
                    .map(|value| serde_json::to_string(value).unwrap_or_default())
                    .unwrap_or_default();
                tools.push(json!({"id":state.next_call_id(name),"type":"function","function":{"name":name,"arguments":arguments}}));
            } else if let Some(data) = part.get("inlineData").or_else(|| part.get("inline_data")) {
                let bytes = data.get("data").and_then(Value::as_str).unwrap_or("");
                if !bytes.is_empty() {
                    let mime = data
                        .get("mimeType")
                        .or_else(|| data.get("mime_type"))
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty())
                        .unwrap_or("image/png");
                    images.push(json!({"type":"image_url","index":images.len(),"image_url":{"url":format!("data:{mime};base64,{bytes}")}}));
                }
            }
        }
        if !text.is_empty() {
            choice["message"]["content"] = Value::String(text);
        }
        if !reasoning.is_empty() {
            choice["message"]["reasoning_content"] = Value::String(reasoning);
        }
        if !tools.is_empty() {
            choice["message"]["tool_calls"] = Value::Array(tools);
            choice["finish_reason"] = Value::String("tool_calls".into());
            choice["native_finish_reason"] = Value::String("tool_calls".into());
        }
        if !images.is_empty() {
            choice["message"]["images"] = Value::Array(images);
        }
        choices.push(choice);
    }
    output["choices"] = Value::Array(choices);
    serde_json::to_vec(&output).unwrap_or_default()
}

fn stream_shell(root: &Value, created: i64) -> Value {
    let mut output = json!({
        "id":root.get("responseId").and_then(Value::as_str).unwrap_or(""),
        "object":"chat.completion.chunk","created":created,
        "model":root.get("modelVersion").and_then(Value::as_str).unwrap_or("model"),
        "choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":null},"finish_reason":null,"native_finish_reason":null}]
    });
    if let Some(usage) = root.get("usageMetadata") {
        output["usage"] = usage_value(usage);
    }
    output
}

fn usage_value(usage: &Value) -> Value {
    let mut output = json!({
        "completion_tokens":usage.get("candidatesTokenCount").and_then(Value::as_i64).unwrap_or(0),
        "prompt_tokens":usage.get("promptTokenCount").and_then(Value::as_i64).unwrap_or(0)
    });
    if let Some(total) = usage.get("totalTokenCount").and_then(Value::as_i64) {
        output["total_tokens"] = Value::from(total);
    }
    if let Some(reasoning) = usage
        .get("thoughtsTokenCount")
        .and_then(Value::as_i64)
        .filter(|value| *value > 0)
    {
        output["completion_tokens_details"] = json!({"reasoning_tokens":reasoning});
    }
    if let Some(cached) = usage
        .get("cachedContentTokenCount")
        .and_then(Value::as_i64)
        .filter(|value| *value > 0)
    {
        output["prompt_tokens_details"] = json!({"cached_tokens":cached});
    }
    output
}

fn parse_time(value: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|value| value.timestamp())
}

fn synthesized_call_id_nonce(
    original_request: &[u8],
    request: &[u8],
    first_payload: &[u8],
) -> u128 {
    let mut digest = Sha256::new();
    digest.update(b"ctox-gemini-chat-call-id-v1\0");
    for value in [original_request, request, first_payload] {
        digest.update((value.len() as u64).to_be_bytes());
        digest.update(value);
    }
    let digest = digest.finalize();
    u128::from_be_bytes(digest[..16].try_into().expect("SHA-256 prefix is 16 bytes"))
}

fn value_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        value => serde_json::to_string(value).unwrap_or_default(),
    }
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
        convert_gemini_response_to_openai_chat_non_stream,
        convert_gemini_response_to_openai_chat_stream, GeminiToChatStreamState,
    };
    use serde_json::Value;

    #[test]
    fn non_stream_preserves_candidates_reasoning_tools_images_and_usage() {
        let request = br#"{"tools":[{"name":"read file"}]}"#;
        let raw = br#"{"responseId":"r1","modelVersion":"gemini","createTime":"2026-08-03T12:34:56Z","usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5},"candidates":[{"index":1,"finishReason":"STOP","content":{"parts":[{"thought":true,"text":"why"},{"text":"answer"},{"functionCall":{"name":"read_file","args":{} }},{"inlineData":{"data":"AAAA"}}]}}]}"#;
        let output: Value = serde_json::from_slice(
            &convert_gemini_response_to_openai_chat_non_stream(request, b"", raw),
        )
        .unwrap();
        assert_eq!(output["id"], "r1");
        assert_eq!(output["choices"][0]["index"], 1);
        assert_eq!(output["choices"][0]["message"]["reasoning_content"], "why");
        assert_eq!(output["choices"][0]["message"]["content"], "answer");
        assert_eq!(
            output["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read file"
        );
        assert_eq!(output["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(output["usage"]["total_tokens"], 5);
    }

    #[test]
    fn stream_retains_tool_and_finish_state() {
        let mut state = GeminiToChatStreamState::default();
        let first = convert_gemini_response_to_openai_chat_stream(
            "fallback",
            b"{}",
            b"",
            br#"{"responseId":"r","candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"run","args":{}}}]}}]}"#,
            &mut state,
        );
        let first: Value = serde_json::from_slice(&first[0]).unwrap();
        assert_eq!(first["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        let final_chunk = convert_gemini_response_to_openai_chat_stream(
            "fallback",
            b"{}",
            b"",
            br#"{"usageMetadata":{"promptTokenCount":1},"candidates":[{"index":0,"finishReason":"STOP","content":{"parts":[]}}]}"#,
            &mut state,
        );
        let final_chunk: Value = serde_json::from_slice(&final_chunk[0]).unwrap();
        assert_eq!(final_chunk["choices"][0]["finish_reason"], "tool_calls");
    }
}
