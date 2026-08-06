// ref: internal/translator/gemini/interactions/interactions_gemini_common.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Map, Value};

use crate::internal::translator::antigravity::interactions::convert_interactions_request_to_antigravity;

pub fn convert_interactions_request_to_gemini(
    model_name: &str,
    input: &[u8],
    stream: bool,
) -> Vec<u8> {
    let wrapped = convert_interactions_request_to_antigravity(model_name, input, stream);
    let root = serde_json::from_slice::<Value>(&wrapped).unwrap_or(Value::Null);
    let Some(mut request) = root.get("request").and_then(Value::as_object).cloned() else {
        return input.to_vec();
    };
    request.remove("safetySettings");
    if serde_json::from_slice::<Value>(input)
        .ok()
        .and_then(|root| root.get("model").cloned())
        .is_some()
        && !model_name.is_empty()
    {
        request.insert("model".into(), Value::String(model_name.to_owned()));
    }
    serde_json::to_vec(&Value::Object(request)).unwrap_or_else(|_| input.to_vec())
}

pub fn convert_gemini_request_to_interactions(
    model_name: &str,
    input: &[u8],
    stream: bool,
) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(input).unwrap_or(Value::Null);
    let mut out = Map::new();
    out.insert("model".into(), Value::String(model_name.to_owned()));
    out.insert("stream".into(), Value::Bool(stream));
    if let Some(system) = root
        .get("systemInstruction")
        .or_else(|| root.get("system_instruction"))
    {
        let text = system_text(system);
        if !text.is_empty() {
            out.insert("system_instruction".into(), Value::String(text));
        }
    }
    if let Some(config) = root.get("generationConfig") {
        out.insert("generation_config".into(), snake_case_value(config));
    }
    let tools = root
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|tool| {
            tool.get("functionDeclarations")
                .or_else(|| tool.get("function_declarations"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .map(|function| {
            let mut tool = json!({
                "type":"function",
                "name":function.get("name").cloned().unwrap_or(Value::String(String::new()))
            });
            if let Some(description) = function.get("description") {
                tool["description"] = description.clone();
            }
            if let Some(parameters) = function
                .get("parameters")
                .or_else(|| function.get("parametersJsonSchema"))
            {
                tool["parameters"] = parameters.clone();
            }
            tool
        })
        .collect::<Vec<_>>();
    if !tools.is_empty() {
        out.insert("tools".into(), Value::Array(tools));
    }
    let mut steps = Vec::new();
    for content in root
        .get("contents")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let role = content
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        for part in content
            .get("parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(call) = part.get("functionCall") {
                steps.push(json!({
                    "type":"function_call",
                    "name":call.get("name").cloned().unwrap_or(Value::String(String::new())),
                    "call_id":call.get("id").or_else(|| call.get("call_id")).cloned().unwrap_or(Value::String(String::new())),
                    "arguments":call.get("args").cloned().unwrap_or_else(|| json!({}))
                }));
            } else if let Some(response) = part.get("functionResponse") {
                steps.push(json!({
                    "type":"function_result",
                    "name":response.get("name").cloned().unwrap_or(Value::String(String::new())),
                    "call_id":response.get("id").or_else(|| response.get("call_id")).cloned().unwrap_or(Value::String(String::new())),
                    "result":response.get("response").cloned().unwrap_or_else(|| json!({}))
                }));
            } else if let Some(item) = part_content(part) {
                let kind = if role == "model" && part.get("thought") == Some(&Value::Bool(true)) {
                    "thought"
                } else if role == "model" {
                    "model_output"
                } else {
                    "user_input"
                };
                steps.push(json!({"type":kind,"content":[item]}));
            }
        }
    }
    out.insert("input".into(), Value::Array(steps));
    serde_json::to_vec(&Value::Object(out)).unwrap_or_else(|_| input.to_vec())
}

fn system_text(system: &Value) -> String {
    if let Some(text) = system
        .as_str()
        .or_else(|| system.get("text").and_then(Value::as_str))
    {
        return text.to_owned();
    }
    system
        .get("parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn part_content(part: &Value) -> Option<Value> {
    if let Some(text) = part.get("text") {
        return Some(json!({"type":"text","text":text}));
    }
    let inline = part.get("inlineData").or_else(|| part.get("inline_data"))?;
    let mime = inline
        .get("mimeType")
        .or_else(|| inline.get("mime_type"))
        .and_then(Value::as_str)
        .unwrap_or("application/octet-stream");
    let kind = mime.split('/').next().unwrap_or("file");
    Some(json!({
        "type":kind,"mime_type":mime,
        "data":inline.get("data").cloned().unwrap_or(Value::String(String::new()))
    }))
}

fn snake_case_value(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(snake_case_value).collect()),
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (snake_case(key), snake_case_value(value)))
                .collect(),
        ),
        value => value.clone(),
    }
}

fn snake_case(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        if character.is_ascii_uppercase() {
            if !out.is_empty() {
                out.push('_');
            }
            out.push(character.to_ascii_lowercase());
        } else {
            out.push(character);
        }
    }
    out
}
