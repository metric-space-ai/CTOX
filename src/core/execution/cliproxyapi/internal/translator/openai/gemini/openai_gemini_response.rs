// ref: internal/translator/openai/gemini/openai_gemini_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

//! OpenAI Chat Completions response to Gemini-native response conversion.

use std::collections::BTreeMap;

use serde_json::{json, Map, Number, Value};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OpenAiToGeminiState {
    tool_calls: BTreeMap<i64, ToolCallAccumulator>,
    content: String,
    is_first_chunk: bool,
}

impl OpenAiToGeminiState {
    #[must_use]
    pub fn accumulated_content(&self) -> &str {
        &self.content
    }
}

#[allow(clippy::too_many_arguments)]
pub fn convert_openai_response_to_gemini_stream_with_context(
    _context: &crate::sdk::translator::TranslationContext,
    _model_name: &str,
    _original_request: &[u8],
    _request: &[u8],
    raw_json: &[u8],
    state: &mut OpenAiToGeminiState,
) -> Vec<Vec<u8>> {
    convert_openai_response_to_gemini_stream(raw_json, state)
}

pub fn convert_openai_response_to_gemini_stream(
    raw_json: &[u8],
    state: &mut OpenAiToGeminiState,
) -> Vec<Vec<u8>> {
    if trim_ascii(raw_json) == b"[DONE]" {
        return Vec::new();
    }
    let payload = raw_json
        .strip_prefix(b"data:")
        .map(trim_ascii)
        .unwrap_or(raw_json);
    let Ok(root) = serde_json::from_slice::<Value>(payload) else {
        return Vec::new();
    };
    let Some(choices) = root.get("choices").and_then(Value::as_array) else {
        return Vec::new();
    };

    if choices.is_empty() {
        let Some(usage) = root.get("usage") else {
            return Vec::new();
        };
        let mut output = json!({"candidates":[],"usageMetadata":{}});
        if let Some(model) = root.get("model") {
            output["model"] = Value::String(gjson_string(model));
        }
        set_gemini_usage_metadata_from_openai_usage(&mut output, usage);
        return encode_one(output);
    }

    let mut results = Vec::new();
    for choice in choices {
        let mut template = base_gemini_response(&root);
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        let base = template.clone();

        if state.is_first_chunk {
            if let Some(role) = delta.get("role") {
                if gjson_string(role) == "assistant" {
                    template["candidates"][0]["content"]["role"] = Value::String("model".into());
                }
                state.is_first_chunk = false;
                results.extend(encode_one(template));
                continue;
            }
        }

        let mut chunk_outputs = Vec::new();
        if let Some(reasoning) = delta.get("reasoning_content") {
            for text in extract_reasoning_texts(reasoning) {
                if text.is_empty() {
                    continue;
                }
                let mut chunk = base.clone();
                chunk["candidates"][0]["content"]["parts"] =
                    Value::Array(vec![json!({"thought":true,"text":text})]);
                chunk_outputs.push(chunk);
            }
        }
        if let Some(content) = delta.get("content") {
            let text = gjson_string(content);
            if !text.is_empty() {
                state.content.push_str(&text);
                let mut chunk = base.clone();
                chunk["candidates"][0]["content"]["parts"] =
                    Value::Array(vec![json!({"text":text})]);
                chunk_outputs.push(chunk);
            }
        }
        if !chunk_outputs.is_empty() {
            for chunk in chunk_outputs {
                results.extend(encode_one(chunk));
            }
            continue;
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tool_call in tool_calls {
                let tool_type = tool_call.get("type").map(gjson_string).unwrap_or_default();
                if !tool_type.is_empty() && tool_type != "function" {
                    continue;
                }
                let Some(function) = tool_call.get("function") else {
                    continue;
                };
                let index = tool_call.get("index").and_then(gjson_i64).unwrap_or(0);
                let accumulator = state.tool_calls.entry(index).or_default();
                let id = tool_call.get("id").map(gjson_string).unwrap_or_default();
                if !id.is_empty() {
                    accumulator.id = id;
                }
                let name = function.get("name").map(gjson_string).unwrap_or_default();
                if !name.is_empty() {
                    accumulator.name = name;
                }
                let arguments = function
                    .get("arguments")
                    .map(gjson_string)
                    .unwrap_or_default();
                if !arguments.is_empty() {
                    accumulator.arguments.push_str(&arguments);
                }
            }
            continue;
        }

        if let Some(finish_reason) = choice.get("finish_reason") {
            template["candidates"][0]["finishReason"] = Value::String(
                map_openai_finish_reason_to_gemini(&gjson_string(finish_reason)).into(),
            );
            if !state.tool_calls.is_empty() {
                let parts = std::mem::take(&mut state.tool_calls)
                    .into_values()
                    .map(tool_accumulator_part)
                    .collect();
                template["candidates"][0]["content"]["parts"] = Value::Array(parts);
            }
            results.extend(encode_one(template));
            continue;
        }

        if let Some(usage) = root.get("usage") {
            set_gemini_usage_metadata_from_openai_usage(&mut template, usage);
            results.extend(encode_one(template));
        }
    }
    results
}

#[allow(clippy::too_many_arguments)]
pub fn convert_openai_response_to_gemini_non_stream_with_context(
    _context: &crate::sdk::translator::TranslationContext,
    _model_name: &str,
    _original_request: &[u8],
    _request: &[u8],
    raw_json: &[u8],
    _state: &mut OpenAiToGeminiState,
) -> Vec<u8> {
    convert_openai_response_to_gemini_non_stream(raw_json)
}

pub fn convert_openai_response_to_gemini_non_stream(raw_json: &[u8]) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(raw_json).unwrap_or(Value::Null);
    let mut output = base_gemini_response(&root);

    for choice in root
        .get("choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let index = choice.get("index").and_then(gjson_i64).unwrap_or(0);
        let message = choice.get("message").unwrap_or(&Value::Null);
        if message.get("role").map(gjson_string).as_deref() == Some("assistant") {
            output["candidates"][0]["content"]["role"] = Value::String("model".into());
        }

        let mut parts = Vec::new();
        if let Some(reasoning) = message.get("reasoning_content") {
            parts.extend(
                extract_reasoning_texts(reasoning)
                    .into_iter()
                    .filter(|text| !text.is_empty())
                    .map(|text| json!({"thought":true,"text":text})),
            );
        }
        if let Some(content) = message.get("content") {
            let text = gjson_string(content);
            if !text.is_empty() {
                parts.push(json!({"text":text}));
            }
        }
        for tool_call in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if tool_call.get("type").map(gjson_string).as_deref() != Some("function") {
                continue;
            }
            let function = tool_call.get("function").unwrap_or(&Value::Null);
            let mut call = Map::new();
            let id = tool_call.get("id").map(gjson_string).unwrap_or_default();
            if !id.is_empty() {
                call.insert("id".into(), Value::String(id));
            }
            call.insert(
                "name".into(),
                Value::String(function.get("name").map(gjson_string).unwrap_or_default()),
            );
            call.insert(
                "args".into(),
                parse_args_to_object(
                    &function
                        .get("arguments")
                        .map(gjson_string)
                        .unwrap_or_default(),
                ),
            );
            parts.push(json!({"functionCall":call}));
        }
        let existing_parts = output["candidates"][0]["content"]["parts"]
            .as_array_mut()
            .expect("Gemini response template carries a parts array");
        for (index, part) in parts.into_iter().enumerate() {
            if let Some(existing) = existing_parts.get_mut(index) {
                *existing = part;
            } else {
                existing_parts.push(part);
            }
        }
        if let Some(reason) = choice.get("finish_reason") {
            output["candidates"][0]["finishReason"] =
                Value::String(map_openai_finish_reason_to_gemini(&gjson_string(reason)).into());
        }
        output["candidates"][0]["index"] = Value::from(index);
    }
    if let Some(usage) = root.get("usage") {
        set_gemini_usage_metadata_from_openai_usage(&mut output, usage);
    }
    serde_json::to_vec(&output).unwrap_or_default()
}

pub fn gemini_token_count(count: i64) -> Vec<u8> {
    crate::internal::translator::common::gemini_token_count_json(count)
}

fn base_gemini_response(root: &Value) -> Value {
    let mut output = json!({
        "candidates":[{"content":{"parts":[],"role":"model"},"index":0}]
    });
    if let Some(model) = root.get("model") {
        output["model"] = Value::String(gjson_string(model));
    }
    output
}

fn tool_accumulator_part(accumulator: ToolCallAccumulator) -> Value {
    let mut function = Map::new();
    if !accumulator.id.is_empty() {
        function.insert("id".into(), Value::String(accumulator.id));
    }
    function.insert("name".into(), Value::String(accumulator.name));
    function.insert("args".into(), parse_args_to_object(&accumulator.arguments));
    json!({"functionCall":function})
}

fn map_openai_finish_reason_to_gemini(reason: &str) -> &'static str {
    match reason {
        "length" => "MAX_TOKENS",
        "content_filter" => "SAFETY",
        "stop" | "tool_calls" => "STOP",
        _ => "STOP",
    }
}

fn parse_args_to_object(arguments: &str) -> Value {
    let trimmed = arguments.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        return json!({});
    }
    if let Ok(Value::Object(object)) = serde_json::from_str::<Value>(trimmed) {
        return Value::Object(object);
    }
    tolerant_parse_json_object(trimmed).unwrap_or_else(|| json!({}))
}

fn tolerant_parse_json_object(input: &str) -> Option<Value> {
    let start = input.find('{')?;
    let end = input.rfind('}')?;
    if start >= end {
        return None;
    }
    let characters = input[start + 1..end].chars().collect::<Vec<_>>();
    let mut cursor = 0usize;
    let mut output = Map::new();
    while cursor < characters.len() {
        skip_delimiters(&characters, &mut cursor);
        if cursor >= characters.len() {
            break;
        }
        if characters[cursor] != '"' {
            skip_to_next_comma(&characters, &mut cursor);
            continue;
        }
        let (key_token, next) = parse_json_string_token(&characters, cursor)?;
        let key = serde_json::from_str::<String>(&key_token)
            .unwrap_or_else(|_| key_token.trim_matches('"').to_owned());
        cursor = next;
        skip_whitespace(&characters, &mut cursor);
        if characters.get(cursor) != Some(&':') {
            break;
        }
        cursor += 1;
        skip_whitespace(&characters, &mut cursor);
        if cursor >= characters.len() {
            break;
        }
        let (value, next) = parse_tolerant_value(&characters, cursor);
        output.insert(key, value);
        cursor = next;
        skip_whitespace(&characters, &mut cursor);
        if characters.get(cursor) == Some(&',') {
            cursor += 1;
        }
    }
    (!output.is_empty()).then_some(Value::Object(output))
}

fn parse_tolerant_value(characters: &[char], cursor: usize) -> (Value, usize) {
    if characters[cursor] == '"' {
        return parse_json_string_token(characters, cursor).map_or_else(
            || (Value::String(String::new()), characters.len()),
            |(token, next)| {
                (
                    Value::String(
                        serde_json::from_str::<String>(&token)
                            .unwrap_or_else(|_| token.trim_matches('"').to_owned()),
                    ),
                    next,
                )
            },
        );
    }
    if matches!(characters[cursor], '{' | '[') {
        return capture_bracketed(characters, cursor).map_or_else(
            || (Value::Null, characters.len()),
            |(segment, next)| {
                let value =
                    serde_json::from_str::<Value>(&segment).unwrap_or(Value::String(segment));
                (value, next)
            },
        );
    }
    let mut end = cursor;
    while end < characters.len() && characters[end] != ',' {
        end += 1;
    }
    let token = characters[cursor..end]
        .iter()
        .collect::<String>()
        .trim()
        .to_owned();
    let value = match token.as_str() {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "null" => Value::Null,
        _ => parse_number(&token).unwrap_or(Value::String(token)),
    };
    (value, end)
}

fn parse_json_string_token(characters: &[char], start: usize) -> Option<(String, usize)> {
    if characters.get(start) != Some(&'"') {
        return None;
    }
    let mut cursor = start + 1;
    let mut escaped = false;
    while cursor < characters.len() {
        let current = characters[cursor];
        if current == '\\' && !escaped {
            escaped = true;
            cursor += 1;
            continue;
        }
        if current == '"' && !escaped {
            return Some((characters[start..=cursor].iter().collect(), cursor + 1));
        }
        escaped = false;
        cursor += 1;
    }
    None
}

fn capture_bracketed(characters: &[char], start: usize) -> Option<(String, usize)> {
    let open = *characters.get(start)?;
    let close = match open {
        '{' => '}',
        '[' => ']',
        _ => return None,
    };
    let mut depth = 0usize;
    let mut cursor = start;
    let mut in_string = false;
    let mut escaped = false;
    while cursor < characters.len() {
        let current = characters[cursor];
        if in_string {
            if current == '\\' && !escaped {
                escaped = true;
                cursor += 1;
                continue;
            }
            if current == '"' && !escaped {
                in_string = false;
            }
            escaped = false;
            cursor += 1;
            continue;
        }
        if current == '"' {
            in_string = true;
        } else if current == open {
            depth += 1;
        } else if current == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some((characters[start..=cursor].iter().collect(), cursor + 1));
            }
        }
        cursor += 1;
    }
    None
}

fn parse_number(token: &str) -> Option<Value> {
    if let Ok(value) = token.parse::<i64>() {
        return Some(Value::from(value));
    }
    if let Ok(value) = token.parse::<u64>() {
        return Some(Value::from(value));
    }
    Number::from_f64(token.parse::<f64>().ok()?).map(Value::Number)
}

fn set_gemini_usage_metadata_from_openai_usage(output: &mut Value, usage: &Value) {
    let prompt = token_count_from_usage(usage, &["prompt_tokens", "input_tokens"]);
    let completion = token_count_from_usage(usage, &["completion_tokens", "output_tokens"]);
    let total = token_count_from_usage(usage, &["total_tokens"]);
    let metadata = ensure_object_field(output, "usageMetadata");
    if let Some(prompt) = prompt {
        metadata.insert("promptTokenCount".into(), Value::from(prompt));
    }
    if let Some(completion) = completion {
        metadata.insert("candidatesTokenCount".into(), Value::from(completion));
    }
    if let Some(total) = total.or_else(|| match (prompt, completion) {
        (None, None) => None,
        _ => Some(prompt.unwrap_or(0).saturating_add(completion.unwrap_or(0))),
    }) {
        metadata.insert("totalTokenCount".into(), Value::from(total));
    }
    if let Some(reasoning) = nested_token_count(
        usage,
        &[
            "/completion_tokens_details/reasoning_tokens",
            "/output_tokens_details/reasoning_tokens",
        ],
    )
    .filter(|value| *value > 0)
    {
        metadata.insert("thoughtsTokenCount".into(), Value::from(reasoning));
    }
    if let Some(cached) = nested_token_count(
        usage,
        &[
            "/prompt_tokens_details/cached_tokens",
            "/input_tokens_details/cached_tokens",
        ],
    )
    .filter(|value| *value > 0)
    {
        metadata.insert("cachedContentTokenCount".into(), Value::from(cached));
    }
}

fn token_count_from_usage(usage: &Value, paths: &[&str]) -> Option<i64> {
    paths
        .iter()
        .find_map(|path| usage.get(*path).and_then(gjson_i64))
}

fn nested_token_count(usage: &Value, pointers: &[&str]) -> Option<i64> {
    pointers
        .iter()
        .find_map(|pointer| usage.pointer(pointer).and_then(gjson_i64))
}

fn extract_reasoning_texts(value: &Value) -> Vec<String> {
    match value {
        Value::Array(values) => values.iter().flat_map(extract_reasoning_texts).collect(),
        Value::String(value) => vec![value.clone()],
        Value::Object(object) => object.get("text").map(gjson_string).into_iter().collect(),
        Value::Bool(_) | Value::Number(_) => Vec::new(),
        Value::Null => Vec::new(),
    }
}

fn ensure_object_field<'a>(output: &'a mut Value, key: &str) -> &'a mut Map<String, Value> {
    if !output.get(key).is_some_and(Value::is_object) {
        output[key] = json!({});
    }
    output[key]
        .as_object_mut()
        .expect("field was initialized as an object")
}

fn encode_one(value: Value) -> Vec<Vec<u8>> {
    vec![serde_json::to_vec(&value).unwrap_or_default()]
}

fn gjson_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
        .or_else(|| value.as_f64().map(|value| value as i64))
        .or_else(|| {
            value
                .as_str()
                .and_then(|value| value.parse::<f64>().ok())
                .map(|value| value as i64)
        })
}

fn gjson_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn skip_whitespace(characters: &[char], cursor: &mut usize) {
    while characters
        .get(*cursor)
        .is_some_and(|value| value.is_ascii_whitespace())
    {
        *cursor += 1;
    }
}

fn skip_delimiters(characters: &[char], cursor: &mut usize) {
    while characters
        .get(*cursor)
        .is_some_and(|value| value.is_ascii_whitespace() || *value == ',')
    {
        *cursor += 1;
    }
}

fn skip_to_next_comma(characters: &[char], cursor: &mut usize) {
    while characters.get(*cursor).is_some_and(|value| *value != ',') {
        *cursor += 1;
    }
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    bytes.trim_ascii()
}
