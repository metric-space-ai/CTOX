// ref: internal/translator/claude/gemini/claude_gemini_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::borrow::Cow;

use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::internal::thinking::{convert_budget_to_level, convert_level_to_budget};

pub fn convert_gemini_request_to_claude(model: &str, input: &[u8], stream: bool) -> Vec<u8> {
    let Ok(root) = serde_json::from_slice::<Value>(input) else {
        return input.to_vec();
    };
    let Some(object) = root.as_object() else {
        return input.to_vec();
    };
    let mut out = Map::new();
    out.insert("model".into(), Value::String(model.to_owned()));
    out.insert("max_tokens".into(), Value::from(32_000));
    out.insert("stream".into(), Value::Bool(stream));
    if let Some(service_tier) = object.get("service_tier").and_then(Value::as_str) {
        out.insert(
            "service_tier".into(),
            Value::String(service_tier.to_owned()),
        );
    }

    if let Some(config) = object.get("generationConfig").and_then(Value::as_object) {
        copy(config, &mut out, "maxOutputTokens", "max_tokens");
        copy(config, &mut out, "topP", "top_p");
        copy(config, &mut out, "stopSequences", "stop_sequences");
        if let Some(thinking) = config.get("thinkingConfig").and_then(Value::as_object) {
            apply_thinking_config(thinking, &mut out);
        }
    }
    let mut messages = Vec::new();
    let mut pending_tool_ids = Vec::new();
    if let Some(instruction) = object
        .get("system_instruction")
        .or_else(|| object.get("systemInstruction"))
    {
        let text = instruction
            .get("parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        if !text.is_empty() {
            messages.push(json!({"role":"user","content":[{"type":"text","text":text}]}));
        }
    }
    if let Some(contents) = object.get("contents").and_then(Value::as_array) {
        for content in contents {
            let role = if content.get("role").and_then(Value::as_str) == Some("model") {
                "assistant"
            } else {
                "user"
            };
            let parts: Vec<Value> = content
                .get("parts")
                .and_then(Value::as_array)
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|part| convert_part(part, role, &mut pending_tool_ids))
                        .collect()
                })
                .unwrap_or_default();
            if !parts.is_empty() {
                messages.push(json!({"role": role, "content": parts}));
            }
        }
    }
    out.insert("messages".into(), Value::Array(messages));
    if let Some(tools) = object.get("tools").and_then(Value::as_array) {
        let converted: Vec<Value> = tools
            .iter()
            .flat_map(|tool| {
                tool.get("functionDeclarations")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
            })
            .map(|function| {
                let schema = function
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| json!({"type":"object","properties":{}}));
                json!({
                    "name": function.get("name").and_then(Value::as_str).unwrap_or_default(),
                    "description": function.get("description").and_then(Value::as_str).unwrap_or_default(),
                    "input_schema": normalize_schema_value(schema)
                })
            })
            .collect();
        if !converted.is_empty() {
            out.insert("tools".into(), Value::Array(converted));
        }
    }
    if let Some(config) = object
        .get("tool_config")
        .and_then(|value| value.get("function_calling_config"))
        .or_else(|| {
            object
                .get("toolConfig")
                .and_then(|value| value.get("functionCallingConfig"))
        })
    {
        apply_tool_choice(config, &mut out);
    }
    serde_json::to_vec(&Value::Object(out)).unwrap_or_else(|_| input.to_vec())
}

fn apply_thinking_config(config: &Map<String, Value>, out: &mut Map<String, Value>) {
    if let Some(level) = config
        .get("thinkingLevel")
        .or_else(|| config.get("thinking_level"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|level| !level.is_empty())
    {
        match level.to_ascii_lowercase().as_str() {
            "none" => {
                out.insert("thinking".into(), json!({"type":"disabled"}));
            }
            "auto" => {
                out.insert("thinking".into(), json!({"type":"enabled"}));
            }
            level => {
                if let Some(budget) = convert_level_to_budget(level) {
                    out.insert(
                        "thinking".into(),
                        json!({"type":"enabled","budget_tokens":budget}),
                    );
                }
            }
        }
        return;
    }
    if let Some(budget) = config
        .get("thinkingBudget")
        .or_else(|| config.get("thinking_budget"))
        .and_then(Value::as_i64)
    {
        match budget {
            0 => {
                out.insert("thinking".into(), json!({"type":"disabled"}));
            }
            -1 => {
                out.insert("thinking".into(), json!({"type":"enabled"}));
            }
            value => {
                let normalized = convert_budget_to_level(value as isize)
                    .and_then(|level| convert_level_to_budget(level.as_str()))
                    .unwrap_or(value as isize);
                out.insert(
                    "thinking".into(),
                    json!({"type":"enabled","budget_tokens":normalized}),
                );
            }
        }
    }
}

fn apply_tool_choice(config: &Value, out: &mut Map<String, Value>) {
    let mode = config
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let choice = match mode {
        "AUTO" => Some(json!({"type":"auto"})),
        "NONE" => Some(json!({"type":"none"})),
        "ANY" => {
            let names = config
                .get("allowedFunctionNames")
                .or_else(|| config.get("allowed_function_names"))
                .and_then(Value::as_array);
            match names {
                Some(names) if names.len() == 1 => {
                    Some(json!({"type":"tool","name":names[0].as_str().unwrap_or_default()}))
                }
                _ => Some(json!({"type":"any"})),
            }
        }
        _ => None,
    };
    if let Some(choice) = choice {
        out.insert("tool_choice".into(), choice);
    }
}

fn copy(from: &Map<String, Value>, to: &mut Map<String, Value>, source: &str, target: &str) {
    if let Some(value) = from.get(source) {
        to.insert(target.to_owned(), value.clone());
    }
}

fn convert_part(part: &Value, role: &str, pending_tool_ids: &mut Vec<String>) -> Option<Value> {
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        return Some(json!({"type":"text","text":text}));
    }
    if let Some(call) = part
        .get("functionCall")
        .and_then(Value::as_object)
        .filter(|_| role == "assistant")
    {
        let id = call
            .get("id")
            .or_else(|| call.get("call_id"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("toolu_{}", Uuid::new_v4().simple()));
        pending_tool_ids.push(id.clone());
        return Some(json!({
            "type":"tool_use", "id":id,
            "name":call.get("name").and_then(Value::as_str).unwrap_or_default(),
            "input":call.get("args").cloned().unwrap_or_else(|| json!({}))
        }));
    }
    if let Some(result) = part.get("functionResponse").and_then(Value::as_object) {
        let explicit_id = result
            .get("id")
            .or_else(|| result.get("call_id"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let id = if let Some(id) = explicit_id {
            if let Some(index) = pending_tool_ids.iter().position(|pending| pending == &id) {
                pending_tool_ids.remove(index);
            }
            id
        } else if pending_tool_ids.is_empty() {
            format!("toolu_{}", Uuid::new_v4().simple())
        } else {
            pending_tool_ids.remove(0)
        };
        let content = result
            .get("response")
            .and_then(|response| response.get("result"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| {
                result
                    .get("response")
                    .map(Value::to_string)
                    .unwrap_or_default()
            });
        return Some(json!({
            "type":"tool_result", "tool_use_id":id,
            "content":content
        }));
    }
    let media = part
        .get("inlineData")
        .or_else(|| part.get("inline_data"))
        .or_else(|| part.get("fileData"))
        .or_else(|| part.get("file_data"))?;
    let mime = media
        .get("mimeType")
        .or_else(|| media.get("mime_type"))
        .and_then(Value::as_str)
        .unwrap_or("application/octet-stream");
    if let Some(uri) = media
        .get("fileUri")
        .or_else(|| media.get("file_uri"))
        .and_then(Value::as_str)
    {
        if mime.to_ascii_lowercase().starts_with("image/") {
            Some(json!({"type":"image","source":{"type":"url","url":uri}}))
        } else if mime.to_ascii_lowercase().starts_with("application/")
            || mime.to_ascii_lowercase().starts_with("text/")
        {
            Some(json!({"type":"document","source":{"type":"url","url":uri,"media_type":mime}}))
        } else {
            Some(json!({"type":"text","text":format!("File: {uri} (Type: {mime})")}))
        }
    } else if let Some(data) = media.get("data").and_then(Value::as_str) {
        if mime.to_ascii_lowercase().starts_with("image/") {
            Some(json!({"type":"image","source":{"type":"base64","media_type":mime,"data":data}}))
        } else if mime.to_ascii_lowercase().starts_with("application/")
            || mime.to_ascii_lowercase().starts_with("text/")
        {
            Some(
                json!({"type":"document","source":{"type":"base64","media_type":mime,"data":data}}),
            )
        } else {
            Some(json!({"type":"text","text":format!("Media content: inline data (Type: {mime})")}))
        }
    } else {
        None
    }
}

pub fn normalize_claude_tool_schema(input: &[u8]) -> Vec<u8> {
    let Ok(value) = serde_json::from_slice::<Value>(input) else {
        return input.to_vec();
    };
    if value.get("additionalProperties") == Some(&Value::Bool(false))
        && value.get("$schema").and_then(Value::as_str)
            == Some("http://json-schema.org/draft-07/schema#")
    {
        return input.to_vec();
    }
    serde_json::to_vec(&normalize_schema_value(value)).unwrap_or_else(|_| input.to_vec())
}

fn normalize_schema_value(mut value: Value) -> Value {
    let Some(object) = value.as_object_mut() else {
        return json!({"type":"object","properties":{}});
    };
    object.entry("type").or_insert_with(|| json!("object"));
    if !object
        .get("additionalProperties")
        .is_some_and(Value::is_boolean)
    {
        object.insert("additionalProperties".into(), Value::Bool(false));
    }
    if !object.get("$schema").is_some_and(Value::is_string) {
        object.insert(
            "$schema".into(),
            Value::String("http://json-schema.org/draft-07/schema#".into()),
        );
    }
    value
}

pub fn lowercase_claude_tool_schema_types(input: &[u8]) -> Cow<'_, [u8]> {
    let Ok(mut value) = serde_json::from_slice::<Value>(input) else {
        return Cow::Borrowed(input);
    };
    let mut changed = false;
    lowercase_types(&mut value, &mut changed);
    if changed {
        Cow::Owned(serde_json::to_vec(&value).unwrap_or_else(|_| input.to_vec()))
    } else {
        Cow::Borrowed(input)
    }
}

fn lowercase_types(value: &mut Value, changed: &mut bool) {
    match value {
        Value::Object(object) => {
            if let Some(kind) = object.get_mut("type") {
                let normalized = kind
                    .as_str()
                    .map(str::to_ascii_lowercase)
                    .unwrap_or_else(|| kind.to_string());
                if kind.as_str() != Some(&normalized) {
                    *kind = Value::String(normalized);
                    *changed = true;
                }
            }
            for child in object.values_mut() {
                lowercase_types(child, changed);
            }
        }
        Value::Array(items) => {
            for item in items {
                lowercase_types(item, changed);
            }
        }
        _ => {}
    }
}
