// ref: internal/translator/openai/interactions/responses/interactions_openai_responses_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub fn convert_openai_responses_request_to_interactions(
    model_name: &str,
    input_raw_json: &[u8],
    stream: bool,
) -> Vec<u8> {
    let root = parse_object(input_raw_json);
    let mut out = json!({"model": request_model(model_name, &root), "input": []});
    if let Some(value) = root.get("stream").and_then(Value::as_bool) {
        out["stream"] = Value::Bool(value);
    } else if stream {
        out["stream"] = Value::Bool(true);
    }
    if let Some(instructions) = root.get("instructions") {
        out["system_instruction"] = Value::String(responses_instructions_text(instructions));
    }
    copy_string(
        &root,
        "previous_response_id",
        &mut out,
        "previous_interaction_id",
    );
    if let Some(input) = root.get("input") {
        out["input"] = responses_input_to_interactions(input);
    }
    if let Some(tools) = root.get("tools") {
        let converted = responses_tools_to_interactions(tools);
        if !converted.is_empty() {
            out["tools"] = Value::Array(converted);
        }
    }
    if let Some(choice) = root.get("tool_choice") {
        object_path(&mut out, &["generation_config"]).insert("tool_choice".into(), choice.clone());
    }
    if let Some(effort) = get_path(&root, &["reasoning", "effort"]).and_then(Value::as_str) {
        object_path(&mut out, &["generation_config"]).insert(
            "thinking_level".into(),
            Value::String(effort.trim().to_lowercase()),
        );
    }
    if let Some(summary) = get_path(&root, &["reasoning", "summary"]).and_then(Value::as_str) {
        object_path(&mut out, &["generation_config"])
            .insert("thinking_summaries".into(), Value::String(summary.into()));
    }
    if let Some(format) = root
        .get("response_format")
        .or_else(|| get_path(&root, &["text", "format"]))
    {
        out["response_format"] = format.clone();
    }
    encode(out)
}

pub fn convert_interactions_request_to_openai_responses(
    model_name: &str,
    input_raw_json: &[u8],
    stream: bool,
) -> Vec<u8> {
    let root = parse_object(input_raw_json);
    let mut out = json!({"model": request_model(model_name, &root), "input": []});
    if stream || root.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        out["stream"] = Value::Bool(true);
    }
    let instructions = interactions_system_instruction_text(&root);
    if !instructions.is_empty() {
        out["instructions"] = Value::String(instructions);
    }
    copy_string(
        &root,
        "previous_interaction_id",
        &mut out,
        "previous_response_id",
    );
    if let Some(input) = root.get("input") {
        out["input"] = interactions_input_to_responses(input);
    }
    if let Some(tools) = root.get("tools") {
        let converted = interactions_tools_to_responses(tools);
        if !converted.is_empty() {
            out["tools"] = Value::Array(converted);
        }
    }
    if let Some(choice) =
        get_path(&root, &["generation_config", "tool_choice"]).or_else(|| root.get("tool_choice"))
    {
        out["tool_choice"] = choice.clone();
    }
    if let Some(effort) = interactions_thinking_effort(&root) {
        object_path(&mut out, &["reasoning"]).insert("effort".into(), Value::String(effort));
    }
    if let Some(summary) =
        get_path(&root, &["generation_config", "thinking_summaries"]).and_then(Value::as_str)
    {
        object_path(&mut out, &["reasoning"])
            .insert("summary".into(), Value::String(summary.into()));
    }
    if let Some(modalities) = root.get("response_modalities") {
        out["modalities"] = modalities.clone();
    }
    copy_string(&root, "service_tier", &mut out, "service_tier");
    if let Some(format) = root.get("response_format") {
        object_path(&mut out, &["text"]).insert("format".into(), format.clone());
    }
    encode(out)
}

fn parse_object(raw: &[u8]) -> Value {
    serde_json::from_slice(raw).unwrap_or_else(|_| Value::Object(Map::new()))
}

fn encode(value: Value) -> Vec<u8> {
    serde_json::to_vec(&value).expect("serde_json::Value serialization cannot fail")
}

fn request_model(model_name: &str, root: &Value) -> String {
    if !model_name.trim().is_empty() {
        model_name.into()
    } else {
        string(root.get("model"))
    }
}

fn responses_instructions_text(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return text.into();
    }
    if let Some(text) = value.get("text").and_then(Value::as_str) {
        return text.into();
    }
    if let Some(parts) = value.get("content").and_then(Value::as_array) {
        return parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect();
    }
    string(Some(value))
}

fn interactions_system_instruction_text(root: &Value) -> String {
    let Some(value) = root.get("system_instruction") else {
        return String::new();
    };
    if let Some(text) = value.as_str() {
        return text.into();
    }
    if let Some(text) = value.get("text").and_then(Value::as_str) {
        return text.into();
    }
    value
        .get("parts")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default()
}

fn interactions_thinking_effort(root: &Value) -> Option<String> {
    [
        &["generation_config", "thinking_level"][..],
        &["generation_config", "thinkingConfig", "thinkingLevel"][..],
        &["generation_config", "thinkingConfig", "thinking_level"][..],
        &["generation_config", "thinking_config", "thinking_level"][..],
    ]
    .into_iter()
    .find_map(|path| get_path(root, path).and_then(Value::as_str))
    .map(|value| value.trim().to_lowercase())
}

fn responses_input_to_interactions(input: &Value) -> Value {
    let source: Vec<&Value> = match input {
        Value::Array(items) => items.iter().collect(),
        _ => vec![input],
    };
    let mut names = HashMap::new();
    let mut out = Vec::new();
    for item in source {
        if let Some(converted) = responses_input_item_to_interactions(item, &mut names) {
            out.push(converted);
        }
    }
    Value::Array(out)
}

fn responses_input_item_to_interactions(
    item: &Value,
    names: &mut HashMap<String, String>,
) -> Option<Value> {
    if let Some(text) = item.as_str() {
        return Some(interactions_text_step("user_input", text));
    }
    match item.get("type").and_then(Value::as_str).unwrap_or("") {
        "message" => {
            let role = item.get("role").and_then(Value::as_str).unwrap_or("");
            let step_type = if matches!(role, "assistant" | "model") {
                "model_output"
            } else {
                "user_input"
            };
            Some(json!({"type": step_type, "content": responses_content_to_interactions(item.get("content"))}))
        }
        "function_call" => {
            let call_id = first_non_empty(&[string(item.get("call_id")), string(item.get("id"))]);
            let name = string(item.get("name"));
            if !call_id.is_empty() && !name.is_empty() {
                names.insert(call_id.clone(), name.clone());
            }
            let mut out = json!({"type":"function_call", "name":name, "arguments": json_value(item.get("arguments"), json!({}))});
            if !call_id.is_empty() {
                out["call_id"] = Value::String(call_id);
            }
            Some(out)
        }
        "function_call_output" => {
            let call_id = first_non_empty(&[string(item.get("call_id")), string(item.get("id"))]);
            let name = first_non_empty(&[
                string(item.get("name")),
                names.get(&call_id).cloned().unwrap_or_default(),
            ]);
            let result = item.get("output").or_else(|| item.get("result"));
            let mut out = json!({"type":"function_result", "name":name, "result":json_value(result, json!({}))});
            if !call_id.is_empty() {
                out["call_id"] = Value::String(call_id);
            }
            Some(out)
        }
        kind @ ("input_text" | "output_text" | "text") => Some(interactions_text_step(
            if kind == "output_text" { "model_output" } else { "user_input" },
            item.get("text").and_then(Value::as_str).unwrap_or(""),
        )),
        kind @ ("input_image" | "output_image") => responses_content_part_to_interactions(item)
            .map(|part| json!({"type": if kind == "output_image" {"model_output"} else {"user_input"}, "content":[part]})),
        _ => item.get("content").map(|content| {
            json!({"type":"user_input", "content":responses_content_to_interactions(Some(content))})
        }),
    }
}

fn responses_content_to_interactions(content: Option<&Value>) -> Vec<Value> {
    match content {
        Some(Value::String(text)) => vec![json!({"type":"text", "text":text})],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(responses_content_part_to_interactions)
            .collect(),
        Some(Value::Object(_)) => content
            .and_then(responses_content_part_to_interactions)
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

pub(super) fn responses_content_part_to_interactions(part: &Value) -> Option<Value> {
    match part.get("type").and_then(Value::as_str).unwrap_or("") {
        "input_text" | "output_text" | "text" => {
            Some(json!({"type":"text", "text":string(part.get("text"))}))
        }
        "input_image" | "output_image" => Some(responses_image_part_to_interactions(part)),
        _ => part
            .get("text")
            .map(|text| json!({"type":"text", "text":string(Some(text))})),
    }
}

fn responses_image_part_to_interactions(part: &Value) -> Value {
    let url = first_non_empty(&[string(part.get("image_url")), string(part.get("url"))]);
    if let Some((mime_type, data)) = parse_data_url(&url) {
        return json!({"type":"image", "mime_type":mime_type, "data":data});
    }
    let data = string(part.get("data"));
    if !data.is_empty() {
        let mut out = json!({"type":"image", "data":data});
        if let Some(mime) = part.get("mime_type").and_then(Value::as_str) {
            out["mime_type"] = Value::String(mime.into());
        }
        return out;
    }
    if url.is_empty() {
        json!({"type":"image"})
    } else {
        json!({"type":"image", "image_url":url})
    }
}

fn interactions_text_step(step_type: &str, text: &str) -> Value {
    json!({"type":step_type, "content":[{"type":"text", "text":text}]})
}

fn responses_tools_to_interactions(tools: &Value) -> Vec<Value> {
    let Some(tools) = tools.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for tool in tools {
        match tool.get("type").and_then(Value::as_str).unwrap_or("") {
            "" | "function" => {
                if let Some(tool) = function_tool_to_interactions(tool, true) {
                    out.push(tool);
                }
            }
            "namespace" => {
                let children = tool
                    .get("children")
                    .or_else(|| tool.get("tools"))
                    .and_then(Value::as_array);
                let declarations: Vec<_> = children
                    .into_iter()
                    .flatten()
                    .filter_map(|tool| function_tool_to_interactions(tool, false))
                    .collect();
                if !declarations.is_empty() {
                    out.push(json!({"function_declarations":declarations}));
                }
            }
            _ => {}
        }
    }
    out
}

fn function_tool_to_interactions(tool: &Value, include_type: bool) -> Option<Value> {
    let name = first_non_empty(&[
        string(tool.get("name")),
        string(get_path(tool, &["function", "name"])),
    ]);
    if name.is_empty() {
        return None;
    }
    let mut out = Map::new();
    if include_type {
        out.insert("type".into(), Value::String("function".into()));
    }
    out.insert("name".into(), Value::String(name));
    copy_first(
        &mut out,
        "description",
        tool,
        &[&["description"], &["function", "description"]],
    );
    copy_first(
        &mut out,
        "parameters",
        tool,
        &[&["parameters"], &["function", "parameters"]],
    );
    Some(Value::Object(out))
}

fn interactions_input_to_responses(input: &Value) -> Value {
    let source: Vec<&Value> = match input {
        Value::Array(items) => items.iter().collect(),
        _ => vec![input],
    };
    Value::Array(
        source
            .into_iter()
            .filter_map(interactions_input_item_to_responses)
            .collect(),
    )
}

fn interactions_input_item_to_responses(item: &Value) -> Option<Value> {
    if let Some(text) = item.as_str() {
        return Some(interactions_text_message(text));
    }
    match item.get("type").and_then(Value::as_str).unwrap_or("") {
        "user_input" => Some(interactions_message_to_responses(item, "user")),
        "model_output" => Some(interactions_message_to_responses(item, "assistant")),
        "thought" => Some(interactions_thought_to_responses(item)),
        "function_call" => Some(interactions_function_call_to_responses(item)),
        "function_result" => Some(interactions_function_result_to_responses(item)),
        _ => None,
    }
}

fn interactions_text_message(text: &str) -> Value {
    json!({"type":"message", "role":"user", "content":[{"type":"input_text", "text":text}]})
}

fn interactions_message_to_responses(item: &Value, role: &str) -> Value {
    let content = match item.get("content") {
        Some(Value::String(text)) => vec![json!({
            "type": if role == "assistant" {"output_text"} else {"input_text"},
            "text":text,
        })],
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| interactions_content_part_to_responses(part, role))
            .collect(),
        _ => Vec::new(),
    };
    json!({"type":"message", "role":role, "content":content})
}

fn interactions_thought_to_responses(item: &Value) -> Value {
    let summary: Vec<_> = interactions_content_texts(item.get("content"))
        .into_iter()
        .map(|text| json!({"type":"summary_text", "text":text}))
        .collect();
    json!({"type":"reasoning", "summary":summary})
}

fn interactions_content_part_to_responses(part: &Value, role: &str) -> Option<Value> {
    let kind = part
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| part.get("text").map(|_| "text"))?;
    match kind {
        "text" => Some(json!({
            "type":if role == "assistant" {"output_text"} else {"input_text"},
            "text":string(part.get("text")),
        })),
        "image" => Some(json!({
            "type":if role == "assistant" {"output_image"} else {"input_image"},
            "image_url":interactions_media_data_url(part),
        })),
        "audio" => {
            let mime = string(part.get("mime_type"));
            let format = mime
                .split_once('/')
                .map_or(mime.as_str(), |(_, value)| value);
            Some(
                json!({"type":"output_text", "text":format!("Audio content: inline data (Format: {})", if format.is_empty() {"unknown"} else {format})}),
            )
        }
        "video" | "document" => {
            let mut out =
                json!({"type":if role == "assistant" {"output_file"} else {"input_file"}});
            let data = interactions_media_data_url(part);
            if !data.is_empty() {
                out["file_data"] = Value::String(data);
            }
            if let Some(name) = part.get("filename").and_then(Value::as_str) {
                out["filename"] = Value::String(name.into());
            }
            Some(out)
        }
        _ => None,
    }
}

pub(super) fn interactions_function_call_to_responses(item: &Value) -> Value {
    let call_id = first_non_empty(&[string(item.get("call_id")), string(item.get("id"))]);
    json!({
        "type":"function_call",
        "call_id":call_id,
        "name":string(item.get("name")),
        "arguments":json_string_value(item.get("arguments"), "{}"),
    })
}

fn interactions_function_result_to_responses(item: &Value) -> Value {
    let call_id = first_non_empty(&[string(item.get("call_id")), string(item.get("id"))]);
    let mut out = json!({
        "type":"function_call_output",
        "call_id":call_id,
        "output":json_string_value(item.get("result").or_else(|| item.get("output")), ""),
    });
    if let Some(name) = item.get("name").and_then(Value::as_str) {
        out["name"] = Value::String(name.into());
    }
    out
}

fn interactions_tools_to_responses(tools: &Value) -> Vec<Value> {
    let Some(tools) = tools.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for tool in tools {
        if let Some(value) = interactions_tool_to_responses(tool) {
            out.push(value);
        }
        if let Some(declarations) = tool.get("function_declarations").and_then(Value::as_array) {
            out.extend(
                declarations
                    .iter()
                    .filter_map(interactions_tool_to_responses),
            );
        }
    }
    out
}

fn interactions_tool_to_responses(tool: &Value) -> Option<Value> {
    let mut out = function_tool_to_interactions(tool, true)?;
    let Value::Object(ref mut map) = out else {
        unreachable!();
    };
    if !map.contains_key("parameters") {
        if let Some(schema) = tool.get("parametersJsonSchema") {
            map.insert("parameters".into(), schema.clone());
        }
    }
    Some(out)
}

pub(super) fn interactions_content_texts(content: Option<&Value>) -> Vec<String> {
    match content {
        Some(Value::String(text)) => vec![text.clone()],
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .or_else(|| get_path(part, &["content", "text"]))
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .map(str::to_owned)
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn interactions_media_data_url(part: &Value) -> String {
    let direct = first_non_empty(&[
        string(part.get("image_url")),
        string(part.get("file_data")),
        string(part.get("url")),
    ]);
    if !direct.is_empty() {
        return direct;
    }
    let data = string(part.get("data"));
    if data.is_empty() {
        return String::new();
    }
    let mime = first_non_empty(&[
        string(part.get("mime_type")),
        "application/octet-stream".into(),
    ]);
    format!("data:{mime};base64,{data}")
}

fn parse_data_url(value: &str) -> Option<(String, String)> {
    let value = value.strip_prefix("data:")?;
    let (header, data) = value.split_once(',')?;
    let mime = header
        .split(';')
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or("application/octet-stream");
    Some((mime.into(), data.into()))
}

fn json_value(value: Option<&Value>, fallback: Value) -> Value {
    match value {
        None => fallback,
        Some(Value::String(text)) => {
            serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.clone()))
        }
        Some(value) => value.clone(),
    }
}

fn json_string_value(value: Option<&Value>, fallback: &str) -> String {
    match value {
        None => fallback.into(),
        Some(Value::String(text)) => text.clone(),
        Some(value) => serde_json::to_string(value).unwrap_or_else(|_| fallback.into()),
    }
}

fn string(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Number(value)) => value.to_string(),
        Some(Value::Bool(value)) => value.to_string(),
        _ => String::new(),
    }
}

fn first_non_empty(values: &[String]) -> String {
    values
        .iter()
        .find(|value| !value.trim().is_empty())
        .cloned()
        .unwrap_or_default()
}

fn get_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(value, |current, key| current.get(key))
}

fn object_path<'a>(value: &'a mut Value, path: &[&str]) -> &'a mut Map<String, Value> {
    let mut current = value;
    for key in path {
        if !current.get(key).is_some_and(Value::is_object) {
            current[*key] = Value::Object(Map::new());
        }
        current = current.get_mut(key).expect("path element was inserted");
    }
    current.as_object_mut().expect("path contains objects")
}

fn copy_string(root: &Value, from: &str, out: &mut Value, to: &str) {
    if let Some(value) = root.get(from).and_then(Value::as_str) {
        out[to] = Value::String(value.into());
    }
}

fn copy_first(out: &mut Map<String, Value>, to: &str, root: &Value, paths: &[&[&str]]) {
    if let Some(value) = paths.iter().find_map(|path| get_path(root, path)) {
        out.insert(to.into(), value.clone());
    }
}
