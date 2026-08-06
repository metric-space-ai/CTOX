// ref: internal/translator/codex/interactions/interactions_codex_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};

use crate::internal::thinking::convert_budget_to_level;

pub fn convert_interactions_request_to_codex(
    model_name: &str,
    input_raw_json: &[u8],
    stream: bool,
) -> Vec<u8> {
    let Ok(root) = serde_json::from_slice::<Value>(input_raw_json) else {
        return input_raw_json.to_vec();
    };
    let mut out = json!({"model":model_name,"instructions":"","input":[]});
    if stream || root.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        out["stream"] = Value::Bool(true);
    }
    if let Some(system) = root
        .get("system_instruction")
        .or_else(|| root.get("systemInstruction"))
    {
        let text = system_text(system);
        if !text.is_empty() {
            out["instructions"] = Value::String(text);
        }
    }
    copy_generation_config(&root, &mut out);
    let mut items = Vec::new();
    if let Some(input) = root.get("input") {
        append_input(input, "user", &mut items);
    }
    out["input"] = Value::Array(items);
    copy_tools(&root, &mut out);
    for key in [
        "tool_choice",
        "parallel_tool_calls",
        "store",
        "metadata",
        "include",
        "truncation",
    ] {
        if let Some(value) = root.get(key) {
            out[key] = value.clone();
        }
    }
    if matches!(
        root.get("service_tier")
            .and_then(Value::as_str)
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("priority" | "fast")
    ) {
        out["service_tier"] = Value::String("priority".into());
    }
    serde_json::to_vec(&out).unwrap_or_else(|_| input_raw_json.to_vec())
}

fn system_text(system: &Value) -> String {
    system
        .as_str()
        .map(str::to_owned)
        .or_else(|| {
            system
                .get("text")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .or_else(|| {
            system.get("parts").and_then(Value::as_array).map(|parts| {
                parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .filter(|text| !text.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
        })
        .unwrap_or_default()
}

fn copy_generation_config(root: &Value, out: &mut Value) {
    let Some(config) = root
        .get("generation_config")
        .or_else(|| root.get("generationConfig"))
    else {
        if let Some(reasoning) = root.get("reasoning") {
            out["reasoning"] = reasoning.clone();
        }
        return;
    };
    if let Some(reasoning) = config.get("reasoning") {
        out["reasoning"] = reasoning.clone();
    }
    if let Some(effort) = reasoning_effort(config) {
        out["reasoning"]["effort"] = Value::String(effort);
    }
    if let Some(summary) = reasoning_summary(config) {
        out["reasoning"]["summary"] = Value::String(summary);
    }
    for (source, target) in [
        ("max_output_tokens", "max_output_tokens"),
        ("maxOutputTokens", "max_output_tokens"),
        ("max_tokens", "max_output_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("topP", "top_p"),
        ("parallel_tool_calls", "parallel_tool_calls"),
        ("parallelToolCalls", "parallel_tool_calls"),
        ("response_format", "response_format"),
        ("responseFormat", "response_format"),
        ("text", "text"),
        ("truncation", "truncation"),
        ("tool_choice", "tool_choice"),
        ("toolChoice", "tool_choice"),
        ("service_tier", "service_tier"),
        ("serviceTier", "service_tier"),
    ] {
        if let Some(value) = config.get(source) {
            out[target] = value.clone();
        }
    }
}

fn reasoning_effort(config: &Value) -> Option<String> {
    [
        "/thinking_level",
        "/thinkingLevel",
        "/thinking_config/thinking_level",
        "/thinking_config/thinkingLevel",
        "/thinkingConfig/thinking_level",
        "/thinkingConfig/thinkingLevel",
        "/reasoning/effort",
    ]
    .into_iter()
    .find_map(|path| config.pointer(path).and_then(Value::as_str))
    .map(str::trim)
    .filter(|value| !value.is_empty())
    .map(str::to_ascii_lowercase)
    .or_else(|| {
        [
            "/thinking_budget",
            "/thinkingBudget",
            "/thinking_config/thinking_budget",
            "/thinking_config/thinkingBudget",
            "/thinkingConfig/thinking_budget",
            "/thinkingConfig/thinkingBudget",
        ]
        .into_iter()
        .find_map(|path| config.pointer(path).and_then(Value::as_i64))
        .and_then(|budget| convert_budget_to_level(budget as isize))
        .map(|level| level.as_str().to_owned())
    })
}

fn reasoning_summary(config: &Value) -> Option<String> {
    [
        "/thinking_summaries",
        "/thinkingSummaries",
        "/reasoning/summary",
    ]
    .into_iter()
    .find_map(|path| config.pointer(path).and_then(Value::as_str))
    .map(str::trim)
    .map(str::to_ascii_lowercase)
    .filter(|value| matches!(value.as_str(), "auto" | "none"))
    .or_else(|| {
        [
            "/include_thoughts",
            "/includeThoughts",
            "/thinking_config/include_thoughts",
            "/thinkingConfig/includeThoughts",
        ]
        .into_iter()
        .find_map(|path| config.pointer(path).and_then(Value::as_bool))
        .map(|enabled| if enabled { "auto" } else { "none" }.into())
    })
}

fn append_input(input: &Value, default_role: &str, out: &mut Vec<Value>) {
    if let Some(text) = input.as_str() {
        append_message_part(default_role, text_part(default_role, text), out);
        return;
    }
    if let Some(array) = input.as_array() {
        for step in array {
            append_step(step, default_role, out);
        }
        return;
    }
    if let Some(steps) = input.get("steps").and_then(Value::as_array) {
        let role = role(input, default_role);
        for step in steps {
            append_step(step, role, out);
        }
    } else {
        append_step(input, default_role, out);
    }
}

fn append_step(step: &Value, default_role: &str, out: &mut Vec<Value>) {
    if let Some(text) = step.as_str() {
        append_message_part(default_role, text_part(default_role, text), out);
        return;
    }
    if let Some(steps) = step.get("steps").and_then(Value::as_array) {
        let role = role(step, default_role);
        for nested in steps {
            append_step(nested, role, out);
        }
        return;
    }
    match step
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "function_call" => out.push(json!({
            "type":"function_call",
            "name":shorten_tool_name(step.get("name").and_then(Value::as_str).unwrap_or_default()),
            "call_id":call_id(step),
            "arguments":json_string(step.get("arguments").or_else(|| step.get("args")), "{}"),
        })),
        "function_result" | "function_call_output" => out.push(json!({
            "type":"function_call_output",
            "call_id":call_id(step),
            "output":json_string(step.get("result").or_else(|| step.get("output")), ""),
        })),
        "thought" | "reasoning" => {
            let text = content_text(step.get("content"))
                .or_else(|| step.get("text").and_then(Value::as_str).map(str::to_owned));
            let mut item = json!({"type":"reasoning"});
            if let Some(text) = text.filter(|text| !text.is_empty()) {
                item["content"] = Value::String(text);
            }
            if let Some(id) = step.get("id") {
                item["id"] = Value::String(value_string(id));
            }
            out.push(item);
        }
        "model_output" | "assistant" => append_content(step.get("content"), "assistant", out),
        _ => {
            let role = role(step, default_role);
            if step.get("content").is_some() {
                append_content(step.get("content"), role, out);
            } else if let Some(text) = step.get("text").and_then(Value::as_str) {
                append_message_part(role, text_part(role, text), out);
            }
        }
    }
}

fn append_content(content: Option<&Value>, role: &str, out: &mut Vec<Value>) {
    let Some(content) = content else { return };
    if let Some(text) = content.as_str() {
        append_message_part(role, text_part(role, text), out);
    } else if let Some(parts) = content.as_array() {
        for part in parts {
            if let Some(part) = message_part(part, role) {
                append_message_part(role, part, out);
            }
        }
    } else if let Some(part) = message_part(content, role) {
        append_message_part(role, part, out);
    }
}

fn message_part(part: &Value, role: &str) -> Option<Value> {
    if let Some(text) = part.get("text") {
        return Some(text_part(role, &value_string(text)));
    }
    match part.get("type").and_then(Value::as_str).unwrap_or_default() {
        "image" => {
            let url = part
                .get("url")
                .or_else(|| part.get("file_uri"))
                .or_else(|| part.get("fileUri"))
                .and_then(Value::as_str)
                .map(str::to_owned)
                .or_else(|| data_url(part));
            url.map(|url| json!({"type":"input_image","image_url":url}))
        }
        "image_url" => Some(
            json!({"type":"input_image","image_url":part.pointer("/image_url/url").and_then(Value::as_str).unwrap_or_default()}),
        ),
        "audio" => {
            let data = part.get("data").and_then(Value::as_str)?;
            let mime = mime_type(part)?;
            let format = mime.split_once('/').map_or(mime, |(_, value)| value);
            Some(json!({"type":"input_audio","input_audio":{"data":data,"format":format}}))
        }
        "input_audio" => Some(
            json!({"type":"input_audio","input_audio":part.get("input_audio").cloned().unwrap_or_else(|| json!({}))}),
        ),
        "video" | "document" | "file" => file_part(part),
        _ => part
            .get("inline_data")
            .or_else(|| part.get("inlineData"))
            .and_then(inline_part),
    }
}

fn text_part(role: &str, text: &str) -> Value {
    json!({"type":if role == "assistant" {"output_text"} else {"input_text"},"text":text})
}

fn append_message_part(role: &str, part: Value, out: &mut Vec<Value>) {
    out.push(json!({"type":"message","role":role,"content":[part]}));
}

fn file_part(part: &Value) -> Option<Value> {
    if let Some(data) = part.pointer("/file/file_data").and_then(Value::as_str) {
        return Some(
            json!({"type":"input_file","file_data":data,"filename":part.pointer("/file/filename").and_then(Value::as_str).unwrap_or_default()}),
        );
    }
    let mime = mime_type(part).unwrap_or("application/octet-stream");
    if let Some(url) = part
        .get("file_uri")
        .or_else(|| part.get("fileUri"))
        .or_else(|| part.get("url"))
        .and_then(Value::as_str)
    {
        return Some(json!({"type":"input_file","file_url":url,"filename":file_name(mime)}));
    }
    part.get("data")
        .and_then(Value::as_str)
        .map(|data| json!({"type":"input_file","file_data":data,"filename":file_name(mime)}))
}

fn inline_part(inline: &Value) -> Option<Value> {
    let mime = mime_type(inline)?;
    let data = inline.get("data").and_then(Value::as_str)?;
    if mime.to_ascii_lowercase().starts_with("image/") {
        Some(json!({"type":"input_image","image_url":format!("data:{mime};base64,{data}")}))
    } else if mime.to_ascii_lowercase().starts_with("audio/") {
        let format = mime.split_once('/').map_or(mime, |(_, value)| value);
        Some(json!({"type":"input_audio","input_audio":{"data":data,"format":format}}))
    } else {
        Some(json!({"type":"input_file","file_data":data,"filename":file_name(mime)}))
    }
}

fn data_url(part: &Value) -> Option<String> {
    Some(format!(
        "data:{};base64,{}",
        mime_type(part)?,
        part.get("data")?.as_str()?
    ))
}

fn mime_type(part: &Value) -> Option<&str> {
    part.get("mime_type")
        .or_else(|| part.get("mimeType"))
        .and_then(Value::as_str)
}

fn file_name(mime: &str) -> String {
    let extension = mime.split_once('/').map_or("bin", |(_, value)| value);
    format!("attachment.{extension}")
}

fn copy_tools(root: &Value, out: &mut Value) {
    let Some(tools) = root.get("tools") else {
        return;
    };
    let Some(tools) = tools.as_array() else {
        out["tools"] = tools.clone();
        return;
    };
    let mut normalized = Vec::new();
    for tool in tools {
        let declarations: Vec<&Value> = tool
            .get("function_declarations")
            .or_else(|| tool.get("functionDeclarations"))
            .and_then(Value::as_array)
            .map(|values| values.iter().collect())
            .unwrap_or_else(|| vec![tool]);
        for declaration in declarations {
            if declaration.get("name").is_none() {
                continue;
            }
            let mut parameters = declaration
                .get("parameters")
                .or_else(|| declaration.get("parametersJsonSchema"))
                .cloned()
                .unwrap_or_else(|| json!({}));
            remove_schema(&mut parameters);
            let mut converted = json!({
                "type":"function",
                "name":shorten_tool_name(declaration.get("name").and_then(Value::as_str).unwrap_or_default()),
                "parameters":parameters,
            });
            if let Some(description) = declaration.get("description") {
                converted["description"] = description.clone();
            }
            normalized.push(converted);
        }
    }
    if !normalized.is_empty() {
        out["tools"] = Value::Array(normalized);
        if out.get("tool_choice").is_none() {
            out["tool_choice"] = Value::String("auto".into());
        }
    }
}

fn remove_schema(value: &mut Value) {
    match value {
        Value::Object(object) => {
            object.remove("$schema");
            object.values_mut().for_each(remove_schema);
        }
        Value::Array(values) => values.iter_mut().for_each(remove_schema),
        _ => {}
    }
}

fn role<'a>(value: &'a Value, default: &'a str) -> &'a str {
    match value.get("role").and_then(Value::as_str) {
        Some("model" | "assistant") => "assistant",
        Some("user") => "user",
        _ => default,
    }
}

fn call_id(step: &Value) -> String {
    ["call_id", "callId", "id"]
        .into_iter()
        .find_map(|key| step.get(key).and_then(Value::as_str))
        .unwrap_or_default()
        .to_owned()
}

fn json_string(value: Option<&Value>, fallback: &str) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(value) => serde_json::to_string(value).unwrap_or_else(|_| fallback.into()),
        None => fallback.into(),
    }
}

fn content_text(content: Option<&Value>) -> Option<String> {
    match content {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Array(parts)) => Some(
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    }
}

fn shorten_tool_name(name: &str) -> String {
    let mut end = name.len().min(64);
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    name[..end].to_owned()
}

fn value_string(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| match value {
            Value::Null => String::new(),
            other => other.to_string(),
        })
}
