// ref: internal/translator/codex/openai/chat-completions/codex_openai_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{HashMap, HashSet};

use serde_json::{json, Map, Value};

#[derive(Clone)]
struct PendingToolCall {
    call_id: String,
    source_call_id: String,
    call_type: String,
    consumed: bool,
}

pub fn convert_openai_chat_request_to_codex(
    model_name: &str,
    input: &[u8],
    stream: bool,
) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(input).unwrap_or(Value::Null);
    let tools = root
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let metadata = ToolMetadata::new(&tools);
    let mut output = json!({
        "instructions":"",
        "stream":stream,
        "reasoning":{"effort":root.get("reasoning_effort").cloned().unwrap_or_else(|| Value::String("medium".into()))},
        "parallel_tool_calls":true,
        "include":["reasoning.encrypted_content"],
        "model":model_name,
        "input":[]
    });

    output["input"] = Value::Array(convert_messages(&root, &metadata));
    adapt_text_format(&root, &mut output);
    if !tools.is_empty() {
        output["tools"] = Value::Array(convert_tools(&tools, &metadata));
    }
    if let Some(choice) = convert_tool_choice(&root, &metadata) {
        output["tool_choice"] = choice;
    }
    output["store"] = Value::Bool(false);
    serde_json::to_vec(&output).unwrap_or_default()
}

fn convert_messages(root: &Value, metadata: &ToolMetadata) -> Vec<Value> {
    let messages = root
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut output = Vec::with_capacity(messages.len());
    let mut pending = Vec::<PendingToolCall>::new();
    let mut ambiguous = HashSet::<String>::new();

    for (message_index, message) in messages.iter().enumerate() {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        if role == "tool" {
            let requested = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            if !requested.is_empty() && ambiguous.contains(requested) {
                continue;
            }
            let Some(call) = pending.iter_mut().find(|call| {
                !call.consumed
                    && (requested.is_empty()
                        || call.source_call_id == requested
                        || call.call_id == requested)
            }) else {
                continue;
            };
            call.consumed = true;
            output.push(json!({
                "type":if call.call_type == "custom" {"custom_tool_call_output"} else {"function_call_output"},
                "call_id":call.call_id,
                "output":tool_output_content(message.get("content"))
            }));
            continue;
        }

        pending.clear();
        ambiguous.clear();
        let content = message_content(role, message.get("content"));
        if role != "assistant" || !content.is_empty() {
            output.push(json!({
                "type":"message",
                "role":if role == "system" {"developer"} else {role},
                "content":content
            }));
        }
        if role != "assistant" {
            continue;
        }
        let calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut counts = HashMap::<String, usize>::new();
        let mut used = HashSet::<String>::new();
        for call in &calls {
            if resolve_tool_call(call, metadata).is_none() {
                continue;
            }
            let id = call.get("id").and_then(Value::as_str).unwrap_or("");
            if !id.is_empty() {
                *counts.entry(id.into()).or_default() += 1;
                used.insert(id.into());
            }
        }
        ambiguous.extend(
            counts
                .into_iter()
                .filter(|(_, count)| *count > 1)
                .map(|(id, _)| id),
        );
        for (call_index, call) in calls.iter().enumerate() {
            let Some((call_type, name, arguments)) = resolve_tool_call(call, metadata) else {
                continue;
            };
            let source = call
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            if !source.is_empty() && ambiguous.contains(&source) {
                continue;
            }
            let mut id = source.clone();
            if id.is_empty() {
                let base = format!("call_missing_{message_index}_{call_index}");
                id = base.clone();
                for suffix in 1.. {
                    if !used.contains(&id) {
                        break;
                    }
                    id = format!("{base}_{suffix}");
                }
                used.insert(id.clone());
            }
            pending.push(PendingToolCall {
                call_id: id.clone(),
                source_call_id: source,
                call_type: call_type.clone(),
                consumed: false,
            });
            let short = metadata.shorten(&name);
            output.push(if call_type == "custom" {
                json!({"type":"custom_tool_call","call_id":id,"name":short,"input":arguments})
            } else {
                json!({"type":"function_call","call_id":id,"name":short,"arguments":arguments})
            });
        }
    }
    output
}

fn message_content(role: &str, content: Option<&Value>) -> Vec<Value> {
    let mut output = Vec::new();
    match content {
        Some(Value::String(text)) if !text.is_empty() => output.push(json!({
            "type":if role == "assistant" {"output_text"} else {"input_text"},
            "text":text
        })),
        Some(Value::Array(parts)) => {
            for part in parts {
                match part.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" => output.push(json!({
                        "type":if role == "assistant" {"output_text"} else {"input_text"},
                        "text":part.get("text").and_then(Value::as_str).unwrap_or("")
                    })),
                    "image_url" if role == "user" => output.push(json!({
                        "type":"input_image",
                        "image_url":part.pointer("/image_url/url").and_then(Value::as_str).unwrap_or("")
                    })),
                    "file" if role == "user" => {
                        let data = part.pointer("/file/file_data").and_then(Value::as_str).unwrap_or("");
                        if !data.is_empty() {
                            let mut item = json!({"type":"input_file","file_data":data});
                            if let Some(name) = part.pointer("/file/filename").and_then(Value::as_str).filter(|value| !value.is_empty()) {
                                item["filename"] = Value::String(name.into());
                            }
                            output.push(item);
                        }
                    }
                    "input_audio" if role == "user" => {
                        let data = part.pointer("/input_audio/data").and_then(Value::as_str).unwrap_or("");
                        if !data.is_empty() {
                            let mut item = json!({"type":"input_audio","data":data});
                            if let Some(format) = part.pointer("/input_audio/format").and_then(Value::as_str).filter(|value| !value.is_empty()) {
                                item["format"] = Value::String(format.into());
                            }
                            output.push(item);
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    output
}

fn resolve_tool_call(call: &Value, metadata: &ToolMetadata) -> Option<(String, String, String)> {
    match call.get("type").and_then(Value::as_str).unwrap_or("") {
        "custom" => Some((
            "custom".into(),
            call.pointer("/custom/name")?.as_str()?.into(),
            call.pointer("/custom/input")
                .and_then(Value::as_str)
                .unwrap_or("")
                .into(),
        )),
        "function" => {
            let name = call.pointer("/function/name")?.as_str()?.to_owned();
            Some((
                if metadata.custom_names.contains(&name) {
                    "custom".into()
                } else {
                    "function".into()
                },
                name,
                call.pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .into(),
            ))
        }
        _ => None,
    }
}

fn tool_output_content(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(text)) => {
            if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                if has_tool_output_image(&parsed) {
                    return tool_output_content(Some(&parsed));
                }
            }
            Value::String(text.clone())
        }
        Some(Value::Array(parts)) => Value::Array(parts.iter().map(tool_output_part).collect()),
        Some(value) => Value::String(serde_json::to_string(value).unwrap_or_default()),
        None => Value::String(String::new()),
    }
}

fn tool_output_part(part: &Value) -> Value {
    match part.get("type").and_then(Value::as_str).unwrap_or("") {
        "text" | "input_text" | "output_text" => json!({
            "type":"input_text","text":part.get("text").and_then(Value::as_str).unwrap_or("")
        }),
        "image_url" | "input_image" => {
            let native = part.get("type").and_then(Value::as_str) == Some("input_image");
            let url = if native {
                part.get("image_url")
            } else {
                part.pointer("/image_url/url")
            }
            .and_then(Value::as_str)
            .unwrap_or("");
            let file_id = if native {
                part.get("file_id")
            } else {
                part.pointer("/image_url/file_id")
            }
            .and_then(Value::as_str)
            .unwrap_or("");
            if url.is_empty() && file_id.is_empty() {
                return fallback_part(part);
            }
            let mut value = json!({"type":"input_image"});
            if !url.is_empty() {
                value["image_url"] = Value::String(url.into());
            }
            if !file_id.is_empty() {
                value["file_id"] = Value::String(file_id.into());
            }
            let detail = if native {
                part.get("detail")
            } else {
                part.pointer("/image_url/detail")
            }
            .and_then(Value::as_str)
            .unwrap_or("");
            if !detail.is_empty() {
                value["detail"] = Value::String(detail.into());
            }
            value
        }
        "file" => {
            let mut value = json!({"type":"input_file"});
            let mut populated = false;
            for (source, target) in [
                ("file_id", "file_id"),
                ("file_data", "file_data"),
                ("file_url", "file_url"),
                ("filename", "filename"),
            ] {
                if let Some(raw) = part
                    .pointer(&format!("/file/{source}"))
                    .and_then(Value::as_str)
                    .filter(|raw| !raw.is_empty())
                {
                    value[target] = Value::String(raw.into());
                    populated = true;
                }
            }
            if populated {
                value
            } else {
                fallback_part(part)
            }
        }
        _ => fallback_part(part),
    }
}

fn fallback_part(value: &Value) -> Value {
    json!({"type":"input_text","text":serde_json::to_string(value).unwrap_or_default()})
}

fn has_tool_output_image(value: &Value) -> bool {
    value.as_array().is_some_and(|items| {
        items
            .iter()
            .any(|item| match item.get("type").and_then(Value::as_str) {
                Some("image_url") => {
                    item.pointer("/image_url/url")
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.is_empty())
                        || item
                            .pointer("/image_url/file_id")
                            .and_then(Value::as_str)
                            .is_some_and(|value| !value.is_empty())
                }
                Some("input_image") => {
                    item.get("image_url")
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.is_empty())
                        || item
                            .get("file_id")
                            .and_then(Value::as_str)
                            .is_some_and(|value| !value.is_empty())
                }
                _ => false,
            })
    })
}

fn adapt_text_format(root: &Value, output: &mut Value) {
    let response_format = root.get("response_format");
    let text = root.get("text");
    let mut settings = Map::new();
    if let Some(format) = response_format {
        match format.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => {
                settings.insert("format".into(), json!({"type":"text"}));
            }
            "json_schema" => {
                if let Some(schema) = format.get("json_schema") {
                    let mut value = json!({"type":"json_schema"});
                    for key in ["name", "strict", "schema"] {
                        if let Some(field) = schema.get(key) {
                            value[key] = field.clone();
                        }
                    }
                    settings.insert("format".into(), value);
                }
            }
            _ => {}
        }
    }
    if let Some(verbosity) = text.and_then(|value| value.get("verbosity")) {
        settings.insert("verbosity".into(), verbosity.clone());
    }
    if !settings.is_empty() {
        output["text"] = Value::Object(settings);
    }
}

fn convert_tools(tools: &[Value], metadata: &ToolMetadata) -> Vec<Value> {
    let mut output = Vec::new();
    for tool in tools {
        match tool.get("type").and_then(Value::as_str).unwrap_or("") {
            "custom" => {
                let mut item = tool.clone();
                let name = tool.get("name").and_then(Value::as_str).unwrap_or("");
                item["name"] = Value::String(metadata.shorten(name));
                output.push(item);
            }
            "function" => {
                let function = tool.get("function").unwrap_or(&Value::Null);
                let mut item = json!({"type":"function"});
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    item["name"] = Value::String(metadata.shorten(name));
                }
                for key in ["description", "parameters", "strict"] {
                    if let Some(value) = function.get(key) {
                        item[key] = value.clone();
                    }
                }
                output.push(item);
            }
            value if !value.is_empty() && tool.is_object() => output.push(tool.clone()),
            _ => {}
        }
    }
    output
}

fn convert_tool_choice(root: &Value, metadata: &ToolMetadata) -> Option<Value> {
    let choice = root.get("tool_choice")?;
    if let Some(choice) = choice.as_str() {
        return Some(Value::String(choice.into()));
    }
    let mut kind = choice.get("type")?.as_str()?.to_owned();
    if kind != "function" && kind != "custom" {
        return Some(choice.clone());
    }
    let name = if kind == "function" {
        choice.pointer("/function/name")
    } else {
        choice.get("name")
    }
    .and_then(Value::as_str)
    .unwrap_or("");
    if kind == "function" && metadata.custom_names.contains(name) {
        kind = "custom".into();
    }
    let mut output = json!({"type":kind});
    if !name.is_empty() {
        output["name"] = Value::String(metadata.shorten(name));
    }
    Some(output)
}

struct ToolMetadata {
    short_names: HashMap<String, String>,
    custom_names: HashSet<String>,
}

impl ToolMetadata {
    fn new(tools: &[Value]) -> Self {
        let mut names = Vec::new();
        let mut seen = HashSet::new();
        let mut custom_names = HashSet::new();
        let mut function_names = HashSet::new();
        for tool in tools {
            let (name, custom) = match tool.get("type").and_then(Value::as_str).unwrap_or("") {
                "function" => (
                    tool.pointer("/function/name")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                    false,
                ),
                "custom" => (tool.get("name").and_then(Value::as_str).unwrap_or(""), true),
                _ => continue,
            };
            if custom {
                custom_names.insert(name.to_owned());
            } else {
                function_names.insert(name.to_owned());
            }
            if !name.is_empty() && seen.insert(name.to_owned()) {
                names.push(name.to_owned());
            }
        }
        for name in function_names {
            custom_names.remove(&name);
        }
        Self {
            short_names: build_short_name_map(&names),
            custom_names,
        }
    }

    fn shorten(&self, name: &str) -> String {
        self.short_names
            .get(name)
            .cloned()
            .unwrap_or_else(|| shorten_name(name))
    }
}

pub(super) fn reverse_short_name_map(input: &[u8]) -> HashMap<String, String> {
    let root = serde_json::from_slice::<Value>(input).unwrap_or(Value::Null);
    let tools = root
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    ToolMetadata::new(&tools)
        .short_names
        .into_iter()
        .map(|(original, short)| (short, original))
        .collect()
}

fn build_short_name_map(names: &[String]) -> HashMap<String, String> {
    let mut output = HashMap::new();
    let mut used = HashSet::new();
    for name in names {
        let base = shorten_name(name);
        let mut candidate = base.clone();
        for suffix in 1.. {
            if !used.contains(&candidate) {
                break;
            }
            let suffix = format!("_{suffix}");
            candidate = format!("{}{}", truncate_bytes(&base, 64 - suffix.len()), suffix);
        }
        used.insert(candidate.clone());
        output.insert(name.clone(), candidate);
    }
    output
}

fn shorten_name(name: &str) -> String {
    if name.len() <= 64 {
        return name.into();
    }
    if let Some(last) = name
        .strip_prefix("mcp__")
        .and_then(|value| value.rsplit("__").next())
    {
        return truncate_bytes(&format!("mcp__{last}"), 64);
    }
    truncate_bytes(name, 64)
}

fn truncate_bytes(value: &str, limit: usize) -> String {
    if value.len() <= limit {
        return value.into();
    }
    let mut end = limit;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].into()
}

#[cfg(test)]
mod tests {
    use super::convert_openai_chat_request_to_codex;
    use serde_json::Value;

    #[test]
    fn maps_media_tools_history_and_structured_output() {
        let output: Value = serde_json::from_slice(&convert_openai_chat_request_to_codex(
            "gpt-5-codex",
            br#"{"reasoning_effort":"high","messages":[{"role":"system","content":"exact"},{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"ok"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"response_format":{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object"}}}}"#,
            true,
        )).unwrap();
        assert_eq!(output["model"], "gpt-5-codex");
        assert_eq!(output["reasoning"]["effort"], "high");
        assert_eq!(output["input"][0]["role"], "developer");
        assert_eq!(output["input"][1]["content"][0]["type"], "input_image");
        assert_eq!(output["input"][2]["type"], "function_call");
        assert_eq!(output["input"][3]["type"], "function_call_output");
        assert_eq!(output["text"]["format"]["type"], "json_schema");
        assert_eq!(output["store"], false);
    }
}
