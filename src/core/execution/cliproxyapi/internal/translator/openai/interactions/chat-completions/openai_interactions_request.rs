// ref: internal/translator/openai/interactions/chat-completions/openai_interactions_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

//! OpenAI chat-completions request body -> Interactions request body.
//!
//! Mirrors upstream's gjson/sjson byte-splice semantics on a typed
//! `serde_json::Value` builder while delegating file-data normalization to
//! `internal::translator::common::normalize_openai_file_data`.

use serde_json::{json, Map, Value};

use crate::internal::translator::common::{
    join_raw_array, normalize_openai_file_data, set_raw_array_items,
};

pub fn convert_openai_request_to_interactions(
    model_name: &str,
    input_raw_json: &[u8],
    stream: bool,
) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(input_raw_json).unwrap_or(Value::Null);
    let mut out = Map::new();
    out.insert(
        "model".into(),
        Value::String(first_nonempty(&[
            model_name,
            root.get("model").and_then(Value::as_str).unwrap_or(""),
        ])),
    );
    out.insert("input".into(), Value::Array(Vec::new()));
    if let Some(stream_value) = openai_request_stream_value(&root, stream) {
        out.insert("stream".into(), Value::Bool(stream_value));
    }
    append_openai_messages_to_interactions(&mut out, &root);
    copy_openai_chat_generation_config_to_interactions(&mut out, &root);
    append_openai_chat_tools_to_interactions(&mut out, &root);
    serde_json::to_vec(&Value::Object(out)).unwrap_or_default()
}

fn openai_request_stream_value(root: &Value, stream: bool) -> Option<bool> {
    if let Some(value) = root.get("stream").and_then(Value::as_bool) {
        return Some(value);
    }
    if stream {
        return Some(true);
    }
    None
}

fn append_openai_messages_to_interactions(out: &mut Map<String, Value>, root: &Value) {
    let Some(messages) = root.get("messages").and_then(Value::as_array) else {
        return;
    };
    let mut input_items: Vec<Vec<u8>> = Vec::with_capacity(messages.len());
    let mut system_builder = String::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        match role.as_str() {
            "system" | "developer" => {
                let text = openai_chat_content_text(message.get("content"));
                if !text.is_empty() {
                    if !system_builder.is_empty() {
                        system_builder.push('\n');
                    }
                    system_builder.push_str(&text);
                }
            }
            _ => append_openai_message_to_interactions(&mut input_items, message),
        }
    }
    if !system_builder.is_empty() {
        out.insert("system_instruction".into(), Value::String(system_builder));
    }
    if !input_items.is_empty() {
        out.insert(
            "input".into(),
            serde_json::from_slice::<Value>(&join_raw_array(&input_items))
                .unwrap_or(Value::Array(Vec::new())),
        );
        let mut encoded = serde_json::to_vec(&Value::Object(out.clone())).unwrap_or_default();
        encoded = set_raw_array_items(&encoded, "input", &input_items);
        if let Some(value) = serde_json::from_slice::<Value>(&encoded)
            .ok()
            .and_then(|v| v.as_object().cloned())
        {
            *out = value;
        }
    }
}

fn append_openai_message_to_interactions(items: &mut Vec<Vec<u8>>, message: &Value) {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match role.as_str() {
        "assistant" => {
            if let Some(reasoning) = message.get("reasoning_content") {
                for text in openai_reasoning_texts(reasoning) {
                    items.push(
                        serde_json::to_vec(&interactions_text_step("thought", &text))
                            .unwrap_or_default(),
                    );
                }
            }
            if let Some(step) = openai_chat_content_step("model_output", message.get("content")) {
                items.push(serde_json::to_vec(&step).unwrap_or_default());
            }
            if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
                for tool_call in tool_calls {
                    if let Some(step) = openai_tool_call_to_interactions_step(tool_call) {
                        items.push(serde_json::to_vec(&step).unwrap_or_default());
                    }
                }
            }
        }
        "tool" | "function" => {
            items.push(
                serde_json::to_vec(&openai_tool_result_to_interactions(message))
                    .unwrap_or_default(),
            );
        }
        _ => {
            if let Some(step) = openai_chat_content_step("user_input", message.get("content")) {
                items.push(serde_json::to_vec(&step).unwrap_or_default());
            }
        }
    }
}

fn openai_chat_content_step(step_type: &str, content: Option<&Value>) -> Option<Value> {
    let content = content?;
    if let Some(text) = content.as_str() {
        if text.is_empty() {
            return None;
        }
        let mut step = Map::new();
        step.insert("type".into(), Value::String(step_type.into()));
        step.insert(
            "content".into(),
            Value::Array(vec![json!({"type":"text","text":text})]),
        );
        return Some(Value::Object(step));
    }
    let mut content_items: Vec<Value> = Vec::new();
    let append_part = |part: &Value, items: &mut Vec<Value>| {
        if let Some(converted) = openai_chat_content_part_to_interactions(part) {
            items.push(converted);
        }
    };
    if let Some(parts) = content.as_array() {
        for part in parts {
            append_part(part, &mut content_items);
        }
    } else if content.is_object() {
        append_part(content, &mut content_items);
    }
    if content_items.is_empty() {
        return None;
    }
    let mut step = Map::new();
    step.insert("type".into(), Value::String(step_type.into()));
    step.insert("content".into(), Value::Array(content_items));
    Some(Value::Object(step))
}

fn openai_chat_content_part_to_interactions(part: &Value) -> Option<Value> {
    let mut part_type = part
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if part_type.is_empty() && part.get("text").is_some() {
        part_type = "text".into();
    }
    match part_type.as_str() {
        "text" | "input_text" | "output_text" => Some(json!({
            "type":"text",
            "text":part.get("text").and_then(Value::as_str).unwrap_or("")
        })),
        "image_url" | "input_image" | "image" => Some(openai_chat_image_part_to_interactions(part)),
        "input_audio" | "audio" => {
            let audio = part.get("input_audio");
            let data = first_nonempty(&[
                audio
                    .and_then(|value| value.get("data"))
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                part.get("data").and_then(Value::as_str).unwrap_or(""),
            ]);
            if data.is_empty() {
                return None;
            }
            let format = first_nonempty(&[
                audio
                    .and_then(|value| value.get("format"))
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                part.get("format").and_then(Value::as_str).unwrap_or(""),
            ]);
            let mut out = json!({"type":"audio","data":data});
            if !format.is_empty() {
                out["mime_type"] = Value::String(openai_input_audio_mime_type(&format));
            }
            Some(out)
        }
        "file" | "input_file" | "document" => {
            let file = part.get("file");
            let filename = first_nonempty(&[
                file.and_then(|value| value.get("filename"))
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                part.get("filename").and_then(Value::as_str).unwrap_or(""),
            ]);
            let fallback_mime = first_nonempty(&[
                file.and_then(|value| value.get("mime_type"))
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                file.and_then(|value| value.get("mimeType"))
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                part.get("mime_type").and_then(Value::as_str).unwrap_or(""),
                part.get("mimeType").and_then(Value::as_str).unwrap_or(""),
            ]);
            let file_data = first_nonempty(&[
                file.and_then(|value| value.get("file_data"))
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                part.get("file_data").and_then(Value::as_str).unwrap_or(""),
                part.get("data").and_then(Value::as_str).unwrap_or(""),
            ]);
            let file_url = first_nonempty(&[
                file.and_then(|value| value.get("file_url"))
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                part.get("file_url").and_then(Value::as_str).unwrap_or(""),
                part.get("url").and_then(Value::as_str).unwrap_or(""),
            ]);
            let mut out = Map::new();
            out.insert("type".into(), Value::String("document".into()));
            if !filename.is_empty() {
                out.insert("filename".into(), Value::String(filename.clone()));
            }
            let mut has_content = false;
            if let Some((mime_type, data)) =
                normalize_openai_file_data(&filename, &fallback_mime, &file_data)
            {
                out.insert("mime_type".into(), Value::String(mime_type));
                out.insert("data".into(), Value::String(data));
                has_content = true;
            }
            if !file_url.is_empty() {
                out.insert("file_url".into(), Value::String(file_url));
                has_content = true;
            }
            has_content.then_some(Value::Object(out))
        }
        _ => None,
    }
}

fn openai_chat_image_part_to_interactions(part: &Value) -> Value {
    let mut out = Map::new();
    out.insert("type".into(), Value::String("image".into()));
    let image_url = first_nonempty(&[
        part.pointer("/image_url/url")
            .and_then(Value::as_str)
            .unwrap_or(""),
        part.get("image_url").and_then(Value::as_str).unwrap_or(""),
        part.get("url").and_then(Value::as_str).unwrap_or(""),
    ]);
    if let Some((mime_type, data)) = openai_chat_parse_data_url(&image_url) {
        out.insert("mime_type".into(), Value::String(mime_type));
        out.insert("data".into(), Value::String(data));
        return Value::Object(out);
    }
    if let Some(data) = part.get("data").and_then(Value::as_str) {
        if !data.is_empty() {
            out.insert("data".into(), Value::String(data.to_owned()));
            if let Some(mime) = part.get("mime_type").and_then(Value::as_str) {
                out.insert("mime_type".into(), Value::String(mime.to_owned()));
            }
            return Value::Object(out);
        }
    }
    if !image_url.is_empty() {
        out.insert("image_url".into(), Value::String(image_url));
    }
    Value::Object(out)
}

fn openai_tool_result_to_interactions(message: &Value) -> Value {
    let mut out = Map::new();
    out.insert("type".into(), Value::String("function_result".into()));
    out.insert("result".into(), Value::String(String::new()));
    let call_id = first_nonempty(&[
        message
            .get("tool_call_id")
            .and_then(Value::as_str)
            .unwrap_or(""),
        message.get("id").and_then(Value::as_str).unwrap_or(""),
    ]);
    if !call_id.is_empty() {
        out.insert("id".into(), Value::String(call_id.clone()));
        out.insert("call_id".into(), Value::String(call_id));
    }
    if let Some(name) = message.get("name").and_then(Value::as_str) {
        if !name.is_empty() {
            out.insert("name".into(), Value::String(name.to_owned()));
        }
    }
    if let Some(content) = message.get("content") {
        match content {
            Value::String(text) => {
                out.insert("result".into(), Value::String(text.clone()));
            }
            other => {
                out.insert("result".into(), other.clone());
            }
        }
    }
    Value::Object(out)
}

fn copy_openai_chat_generation_config_to_interactions(out: &mut Map<String, Value>, root: &Value) {
    let mut config = Map::new();
    copy_openai_number(
        &mut config,
        "max_output_tokens",
        first_existing([root.get("max_completion_tokens"), root.get("max_tokens")]),
    );
    copy_openai_number(&mut config, "temperature", root.get("temperature"));
    copy_openai_number(&mut config, "top_p", root.get("top_p"));
    copy_openai_number(
        &mut config,
        "presence_penalty",
        root.get("presence_penalty"),
    );
    copy_openai_number(
        &mut config,
        "frequency_penalty",
        root.get("frequency_penalty"),
    );
    copy_openai_number(&mut config, "candidate_count", root.get("n"));
    if let Some(stop) = root.get("stop") {
        config.insert("stop_sequences".into(), stop.clone());
    }
    if let Some(choice) = root.get("tool_choice") {
        config.insert("tool_choice".into(), choice.clone());
    }
    if let Some(effort) = root.get("reasoning_effort").and_then(Value::as_str) {
        let trimmed = effort.trim();
        if !trimmed.is_empty() {
            config.insert(
                "thinking_level".into(),
                Value::String(trimmed.to_ascii_lowercase()),
            );
        }
    }
    if !config.is_empty() {
        out.insert("generation_config".into(), Value::Object(config));
    }
    if let Some(response_format) = root.get("response_format") {
        out.insert("response_format".into(), response_format.clone());
    }
    if let Some(modalities) = root.get("modalities") {
        out.insert("response_modalities".into(), modalities.clone());
    }
    if let Some(service_tier) = root.get("service_tier").and_then(Value::as_str) {
        out.insert(
            "service_tier".into(),
            Value::String(service_tier.to_owned()),
        );
    }
}

fn append_openai_chat_tools_to_interactions(out: &mut Map<String, Value>, root: &Value) {
    let Some(tools) = root.get("tools").and_then(Value::as_array) else {
        return;
    };
    let mut tool_items: Vec<Value> = Vec::new();
    for tool in tools {
        if let Some(converted) = openai_chat_tool_to_interactions(tool) {
            tool_items.push(converted);
        }
    }
    if !tool_items.is_empty() {
        out.insert("tools".into(), Value::Array(tool_items));
    }
}

fn openai_chat_tool_to_interactions(tool: &Value) -> Option<Value> {
    let tool_type = tool
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if !tool_type.is_empty() && tool_type != "function" {
        return None;
    }
    let name = first_nonempty(&[
        tool.pointer("/function/name")
            .and_then(Value::as_str)
            .unwrap_or(""),
        tool.get("name").and_then(Value::as_str).unwrap_or(""),
    ]);
    if name.is_empty() {
        return None;
    }
    let mut out = Map::new();
    out.insert("type".into(), Value::String("function".into()));
    out.insert("name".into(), Value::String(name));
    if let Some(desc) = first_existing([
        tool.get("description"),
        tool.pointer("/function/description"),
    ]) {
        if let Some(text) = desc.as_str() {
            out.insert("description".into(), Value::String(text.to_owned()));
        } else {
            out.insert("description".into(), desc.clone());
        }
    }
    if let Some(parameters) =
        first_existing([tool.get("parameters"), tool.pointer("/function/parameters")])
    {
        out.insert("parameters".into(), parameters.clone());
    }
    Some(Value::Object(out))
}

pub(super) fn openai_chat_content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Object(object)) => object
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        Some(Value::Array(parts)) => {
            let mut builder = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        builder.push_str(text);
                    }
                }
            }
            builder
        }
        _ => String::new(),
    }
}

pub(super) fn openai_input_audio_mime_type(format: &str) -> String {
    match format.trim().to_ascii_lowercase().as_str() {
        "wav" => "audio/wav".into(),
        "flac" => "audio/flac".into(),
        "opus" => "audio/opus".into(),
        "pcm16" => "audio/pcm".into(),
        _ => "audio/mpeg".into(),
    }
}

pub(super) fn openai_chat_parse_data_url(value: &str) -> Option<(String, String)> {
    let rest = value.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    if data.is_empty() {
        return None;
    }
    let (mime, encoding) = meta.split_once(';').unwrap_or((meta, ""));
    if !encoding.eq_ignore_ascii_case("base64") || mime.trim().is_empty() {
        return None;
    }
    Some((mime.trim().to_owned(), data.to_owned()))
}

pub(super) fn interactions_text_step(step_type: &str, text: &str) -> Value {
    let mut step = Map::new();
    step.insert("type".into(), Value::String(step_type.into()));
    step.insert(
        "content".into(),
        Value::Array(vec![json!({"type":"text","text":text})]),
    );
    Value::Object(step)
}

pub(super) fn openai_reasoning_texts(reasoning: &Value) -> Vec<String> {
    if let Some(text) = reasoning.as_str() {
        if !text.is_empty() {
            return vec![text.to_owned()];
        }
        return Vec::new();
    }
    if let Some(items) = reasoning.as_array() {
        let mut out = Vec::new();
        for item in items {
            let text = first_nonempty(&[
                item.get("text").and_then(Value::as_str).unwrap_or(""),
                item.get("content").and_then(Value::as_str).unwrap_or(""),
            ]);
            if !text.is_empty() {
                out.push(text);
            }
        }
        return out;
    }
    Vec::new()
}

pub(super) fn openai_tool_call_to_interactions_step(tool_call: &Value) -> Option<Value> {
    if let Some(tool_type) = tool_call.get("type").and_then(Value::as_str) {
        if !tool_type.is_empty() && tool_type != "function" {
            return None;
        }
    }
    let function = tool_call.get("function")?;
    let mut step = Map::new();
    step.insert("type".into(), Value::String("function_call".into()));
    step.insert("name".into(), Value::String(String::new()));
    step.insert("arguments".into(), json!({}));
    if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
        if !id.is_empty() {
            step.insert("id".into(), Value::String(id.to_owned()));
            step.insert("call_id".into(), Value::String(id.to_owned()));
        }
    }
    if let Some(name) = function.get("name").and_then(Value::as_str) {
        step.insert("name".into(), Value::String(name.to_owned()));
    }
    if let Some(arguments) = function.get("arguments") {
        set_raw_json_value(&mut step, "arguments", arguments, b"{}".to_vec());
    }
    Some(Value::Object(step))
}

fn copy_openai_number(target: &mut Map<String, Value>, key: &str, value: Option<&Value>) {
    if let Some(value) = value {
        target.insert(key.into(), value.clone());
    }
}

fn first_existing<'a>(values: impl IntoIterator<Item = Option<&'a Value>>) -> Option<&'a Value> {
    values.into_iter().flatten().next()
}

pub(super) fn first_nonempty(values: &[&str]) -> String {
    for value in values {
        if !value.trim().is_empty() {
            return (*value).to_owned();
        }
    }
    String::new()
}

fn set_raw_json_value(out: &mut Map<String, Value>, path: &str, value: &Value, fallback: Vec<u8>) {
    if value == &Value::Null {
        if let Ok(fallback) = serde_json::from_slice::<Value>(&fallback) {
            out.insert(path.into(), fallback);
        }
        return;
    }
    if let Value::String(text) = value {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
                out.insert(path.into(), parsed);
                return;
            }
        }
        out.insert(path.into(), Value::String(text.clone()));
        return;
    }
    out.insert(path.into(), value.clone());
}
