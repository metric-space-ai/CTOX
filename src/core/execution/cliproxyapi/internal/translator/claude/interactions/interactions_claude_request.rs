// ref: internal/translator/claude/interactions/interactions_claude_request.go:1-461 @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Map, Value};

use crate::internal::thinking::convert_level_to_budget;
use crate::internal::util::sanitize_claude_tool_id;

pub fn convert_interactions_request_to_claude(model: &str, input: &[u8], stream: bool) -> Vec<u8> {
    let Ok(root) = serde_json::from_slice::<Value>(input) else {
        return input.to_vec();
    };
    let Some(object) = root.as_object() else {
        return input.to_vec();
    };
    let mut out = Map::new();
    out.insert("model".into(), Value::String(model.to_owned()));
    out.insert("max_tokens".into(), Value::from(32_000));
    if stream || object.get("stream").and_then(Value::as_bool) == Some(true) {
        out.insert("stream".into(), Value::Bool(true));
    }
    if let Some(system) =
        first(object, &["system_instruction", "systemInstruction"]).and_then(interactions_text)
    {
        out.insert("system".into(), Value::String(system));
    }
    copy_generation_config(object, &mut out);
    let mut messages = Vec::new();
    if let Some(value) = object.get("input") {
        append_input(&mut messages, value);
    }
    out.insert("messages".into(), Value::Array(messages));
    copy_tools(object, &mut out);
    serde_json::to_vec(&Value::Object(out)).unwrap_or_else(|_| input.to_vec())
}

fn copy_generation_config(root: &Map<String, Value>, out: &mut Map<String, Value>) {
    if let Some(config) =
        first(root, &["generation_config", "generationConfig"]).and_then(Value::as_object)
    {
        for (sources, target) in [
            (&["max_output_tokens", "maxOutputTokens"][..], "max_tokens"),
            (&["top_p", "topP"][..], "top_p"),
            (&["temperature"][..], "temperature"),
            (&["stop_sequences", "stopSequences"][..], "stop_sequences"),
        ] {
            if let Some(value) = first(config, sources) {
                out.insert(target.to_owned(), value.clone());
            }
        }
        if let Some(level) = first(config, &["thinking_level", "thinkingLevel", "reasoning"])
            .and_then(|value| {
                value
                    .as_str()
                    .or_else(|| value.get("effort").and_then(Value::as_str))
            })
        {
            set_thinking(out, level);
        }
        if let Some(choice) = first(config, &["tool_choice", "toolChoice"]) {
            set_tool_choice(out, choice);
        }
    }
    if let Some(reasoning) = root.get("reasoning") {
        if let Some(level) = reasoning
            .get("effort")
            .or_else(|| reasoning.get("thinking_level"))
            .and_then(Value::as_str)
        {
            set_thinking(out, level);
        }
    }
    if let Some(choice) = first(root, &["tool_choice", "toolChoice"]) {
        set_tool_choice(out, choice);
    }
}

fn set_thinking(out: &mut Map<String, Value>, level: &str) {
    let level = level.trim().to_ascii_lowercase();
    if level.is_empty() {
        return;
    }
    let thinking = match level.as_str() {
        "none" | "disabled" | "off" | "false" => json!({"type":"disabled"}),
        "auto" | "adaptive" => json!({"type":"adaptive"}),
        _ => match convert_level_to_budget(&level) {
            Some(0) => json!({"type":"disabled"}),
            Some(value) if value < 0 => json!({"type":"enabled"}),
            Some(value) => json!({"type":"enabled","budget_tokens":value}),
            None => {
                out.insert("output_config".into(), json!({"effort":level}));
                json!({"type":"adaptive"})
            }
        },
    };
    out.insert("thinking".into(), thinking);
}

fn set_tool_choice(out: &mut Map<String, Value>, choice: &Value) {
    let (kind, name) = if let Some(kind) = choice.as_str() {
        (kind.trim().to_ascii_lowercase(), None)
    } else {
        let kind = choice
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let name = choice
            .get("name")
            .or_else(|| choice.pointer("/function/name"))
            .and_then(Value::as_str);
        (kind, name)
    };
    let mapped = match kind.as_str() {
        "auto" => Some(json!({"type":"auto"})),
        "required" | "any" => Some(json!({"type":"any"})),
        "function" | "tool" if name.is_some_and(|name| !name.is_empty()) => {
            Some(json!({"type":"tool","name":name.unwrap_or_default()}))
        }
        _ => None,
    };
    if let Some(mapped) = mapped {
        out.insert("tool_choice".into(), mapped);
    }
}

fn append_input(messages: &mut Vec<Value>, input: &Value) {
    if let Some(text) = input.as_str() {
        append_step(
            messages,
            &json!({"type":"user_input","content":[{"type":"text","text":text}]}),
            "user",
        );
    } else if let Some(items) = input.as_array() {
        for item in items {
            append_input_item(messages, item);
        }
    } else if input.is_object() {
        append_input_item(messages, input);
    }
}

fn append_input_item(messages: &mut Vec<Value>, step: &Value) {
    if let Some(steps) = step.get("steps").and_then(Value::as_array) {
        let role = if assistant_role(step) {
            "assistant"
        } else {
            "user"
        };
        for nested in steps {
            append_step(messages, nested, role);
        }
        return;
    }
    if let Some(parts) = step.get("parts") {
        append_step(
            messages,
            &json!({
                "type": if assistant_role(step) {"model_output"} else {"user_input"},
                "content": parts,
            }),
            "user",
        );
        return;
    }
    match step.get("type").and_then(Value::as_str).unwrap_or_default() {
        "function_call" => append_function_call(messages, step),
        "function_result" => append_function_result(messages, step),
        "model_output" | "thought" => append_step(messages, step, "assistant"),
        _ => append_step(messages, step, "user"),
    }
}

fn append_step(messages: &mut Vec<Value>, step: &Value, default_role: &str) {
    let role = match step.get("role").and_then(Value::as_str) {
        Some("user") => "user",
        Some("assistant") => "assistant",
        _ => default_role,
    };
    let mut content = Vec::new();
    match step.get("content") {
        Some(Value::String(text)) => content.push(json!({"type":"text","text":text})),
        Some(Value::Array(parts)) => {
            content.extend(parts.iter().filter_map(|part| convert_content(part, role)))
        }
        _ => {
            if let Some(text) = step.get("text").and_then(Value::as_str) {
                content.push(json!({"type":"text","text":text}));
            }
        }
    }
    if !content.is_empty() {
        messages.push(json!({"role":role,"content":content}));
    }
}

fn convert_content(part: &Value, role: &str) -> Option<Value> {
    let kind = part
        .get("type")
        .and_then(Value::as_str)
        .filter(|kind| !kind.is_empty())
        .unwrap_or_else(|| {
            if part.get("text").is_some() {
                "text"
            } else {
                ""
            }
        });
    match kind {
        "text" => Some(json!({"type":"text","text":part.get("text")?.as_str()?})),
        "thinking" | "reasoning" if role == "assistant" => Some(json!({
            "type":"thinking", "thinking":interactions_text(part)?
        })),
        "image" => media_part(part, "image"),
        "document" | "file" => media_part(part, "document"),
        _ => interactions_text(part)
            .map(|text| json!({"type":"text","text":text}))
            .or_else(|| {
                (part.get("data").is_some() || part.get("file_data").is_some())
                    .then(|| json!({"type":"text","text":format!("[{kind} content omitted]")}))
            }),
    }
}

fn append_function_call(messages: &mut Vec<Value>, step: &Value) {
    let arguments = step
        .get("arguments")
        .or_else(|| step.get("args"))
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    messages.push(json!({"role":"assistant","content":[{
        "type":"tool_use", "id":tool_id(step),
        "name":step.get("name").and_then(Value::as_str).unwrap_or_default(),
        "input":arguments
    }]}));
}

fn append_function_result(messages: &mut Vec<Value>, step: &Value) {
    let result = step.get("result").or_else(|| step.get("output"));
    let content = match result {
        Some(Value::Array(parts)) => Value::Array(
            parts
                .iter()
                .filter_map(|part| convert_content(part, "user"))
                .collect(),
        ),
        Some(value) => Value::String(serde_json::to_string(value).unwrap_or_default()),
        None => Value::String(String::new()),
    };
    messages.push(json!({"role":"user","content":[{
        "type":"tool_result", "tool_use_id":tool_id(step), "content":content
    }]}));
}

fn copy_tools(root: &Map<String, Value>, out: &mut Map<String, Value>) {
    let Some(tools) = root.get("tools").and_then(Value::as_array) else {
        return;
    };
    let mut converted = Vec::new();
    for tool in tools {
        if let Some(declarations) = tool
            .get("function_declarations")
            .or_else(|| tool.get("functionDeclarations"))
            .and_then(Value::as_array)
        {
            converted.extend(declarations.iter().filter_map(convert_tool));
        } else if let Some(tool) = convert_tool(tool) {
            converted.push(tool);
        }
    }
    if !converted.is_empty() {
        out.insert("tools".into(), Value::Array(converted));
    }
}

fn convert_tool(tool: &Value) -> Option<Value> {
    let name = tool
        .get("name")
        .or_else(|| tool.pointer("/function/name"))
        .and_then(Value::as_str)?;
    if name.is_empty() {
        return None;
    }
    let mut out = Map::new();
    out.insert("name".into(), Value::String(name.to_owned()));
    if let Some(description) = tool
        .get("description")
        .or_else(|| tool.pointer("/function/description"))
        .and_then(Value::as_str)
    {
        out.insert("description".into(), Value::String(description.to_owned()));
    }
    let schema = [
        "parameters",
        "parametersJsonSchema",
        "parameters_json_schema",
        "input_schema",
    ]
    .into_iter()
    .find_map(|key| tool.get(key))
    .filter(|value| value.is_object())
    .cloned()
    .unwrap_or_else(|| json!({}));
    out.insert("input_schema".into(), schema);
    Some(Value::Object(out))
}

fn media_part(part: &Value, claude_type: &str) -> Option<Value> {
    let mime = ["mime_type", "mimeType", "media_type", "mediaType"]
        .into_iter()
        .find_map(|key| part.get(key).and_then(Value::as_str))
        .or_else(|| part.pointer("/source/media_type").and_then(Value::as_str))?;
    let data = ["data", "file_data", "fileData"]
        .into_iter()
        .find_map(|key| part.get(key).and_then(Value::as_str))
        .or_else(|| part.pointer("/source/data").and_then(Value::as_str))?;
    if mime.is_empty() || data.is_empty() {
        return None;
    }
    if claude_type == "image" && !mime.to_ascii_lowercase().starts_with("image/") {
        return Some(
            json!({"type":"text","text":format!("Media content: inline data (Type: {mime})")}),
        );
    }
    Some(json!({
        "type":claude_type,
        "source":{"type":"base64","media_type":mime,"data":data}
    }))
}

fn tool_id(step: &Value) -> String {
    for key in ["call_id", "id", "tool_use_id"] {
        if let Some(value) = step
            .get(key)
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
        {
            return sanitize_claude_tool_id(value);
        }
    }
    step.get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .map(|name| sanitize_claude_tool_id(&format!("toolu_{name}")))
        .unwrap_or_else(|| "toolu_interactions".to_owned())
}

fn interactions_text(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return Some(text.to_owned());
    }
    for key in ["text", "thinking"] {
        if let Some(text) = value.get(key).and_then(Value::as_str) {
            return Some(text.to_owned());
        }
    }
    if let Some(content) = value.get("content") {
        return interactions_text(content);
    }
    value
        .get("parts")
        .and_then(Value::as_array)
        .and_then(|parts| {
            let text = parts
                .iter()
                .filter_map(interactions_text)
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        })
}

fn assistant_role(value: &Value) -> bool {
    matches!(
        value.get("role").and_then(Value::as_str),
        Some("model" | "assistant")
    )
}

fn first<'a>(object: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| object.get(*key))
}
