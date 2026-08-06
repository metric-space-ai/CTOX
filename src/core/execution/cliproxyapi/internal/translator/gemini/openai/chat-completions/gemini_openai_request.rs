// ref: internal/translator/gemini/openai/chat-completions/gemini_openai_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::internal::signature::compatible_gemini_signature;
use crate::internal::util::gemini_schema::clean_json_schema_for_gemini;

pub(super) const GEMINI_THOUGHT_BYPASS: &str = "skip_thought_signature_validator";
const SAFETY: &[(&str, &str)] = &[
    ("HARM_CATEGORY_HARASSMENT", "OFF"),
    ("HARM_CATEGORY_HATE_SPEECH", "OFF"),
    ("HARM_CATEGORY_SEXUALLY_EXPLICIT", "OFF"),
    ("HARM_CATEGORY_DANGEROUS_CONTENT", "OFF"),
    ("HARM_CATEGORY_CIVIC_INTEGRITY", "BLOCK_NONE"),
];

pub fn convert_openai_chat_request_to_gemini(
    model_name: &str,
    input: &[u8],
    _stream: bool,
) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(input).unwrap_or(Value::Null);
    let mut output = Map::new();
    output.insert("contents".into(), Value::Array(Vec::new()));
    output.insert("model".into(), Value::String(model_name.to_owned()));

    let mut generation = root
        .get("generationConfig")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(effort) = root
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        generation.insert(
            "thinkingConfig".into(),
            if effort.eq_ignore_ascii_case("auto") {
                json!({"thinkingBudget":-1})
            } else {
                json!({"thinkingLevel":effort.to_ascii_lowercase()})
            },
        );
    }
    for (source, target) in [
        ("temperature", "temperature"),
        ("top_p", "topP"),
        ("top_k", "topK"),
    ] {
        if root.get(source).is_some_and(Value::is_number) {
            generation.insert(target.into(), root[source].clone());
        }
    }
    if root.get("max_tokens").is_some_and(Value::is_number) {
        generation.insert("maxOutputTokens".into(), root["max_tokens"].clone());
    } else if root
        .get("max_completion_tokens")
        .is_some_and(Value::is_number)
    {
        generation.insert(
            "maxOutputTokens".into(),
            root["max_completion_tokens"].clone(),
        );
    }
    if root
        .get("n")
        .and_then(Value::as_i64)
        .is_some_and(|value| value > 1)
    {
        generation.insert("candidateCount".into(), root["n"].clone());
    }
    apply_response_format(&root, &mut generation);
    if let Some(modalities) = root.get("modalities").and_then(Value::as_array) {
        let values = modalities
            .iter()
            .filter_map(Value::as_str)
            .filter_map(|value| match value.to_ascii_lowercase().as_str() {
                "text" => Some(Value::String("TEXT".into())),
                "image" => Some(Value::String("IMAGE".into())),
                _ => None,
            })
            .collect::<Vec<_>>();
        if !values.is_empty() {
            generation.insert("responseModalities".into(), Value::Array(values));
        }
    }
    if let Some(image) = root.get("image_config").and_then(Value::as_object) {
        let mut config = Map::new();
        if let Some(value) = image.get("aspect_ratio").and_then(Value::as_str) {
            config.insert("aspectRatio".into(), Value::String(value.into()));
        }
        if let Some(value) = image.get("image_size").and_then(Value::as_str) {
            config.insert("imageSize".into(), Value::String(value.into()));
        }
        if !config.is_empty() {
            generation.insert("imageConfig".into(), Value::Object(config));
        }
    }
    if !generation.is_empty() {
        output.insert("generationConfig".into(), Value::Object(generation));
    }

    if let Some(messages) = root.get("messages").and_then(Value::as_array) {
        let (system, contents) = convert_messages(messages);
        if !system.is_empty() {
            output.insert(
                "systemInstruction".into(),
                json!({"role":"user","parts":system}),
            );
        }
        output.insert("contents".into(), Value::Array(contents));
    }
    if let Some(tools) = convert_tools(&root) {
        output.insert("tools".into(), tools);
    }
    output.insert(
        "safetySettings".into(),
        Value::Array(
            SAFETY
                .iter()
                .map(|(category, threshold)| json!({"category":category,"threshold":threshold}))
                .collect(),
        ),
    );
    serde_json::to_vec(&Value::Object(output)).unwrap_or_default()
}

fn convert_messages(messages: &[Value]) -> (Vec<Value>, Vec<Value>) {
    let mut call_names = HashMap::new();
    let mut tool_outputs = HashMap::new();
    for message in messages {
        if message.get("role").and_then(Value::as_str) == Some("assistant") {
            for call in message
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if call.get("type").and_then(Value::as_str) == Some("function") {
                    let id = string_at(call, "id");
                    let name = call
                        .pointer("/function/name")
                        .map(value_string)
                        .unwrap_or_default();
                    if !id.is_empty() && !name.is_empty() {
                        call_names.insert(id, name);
                    }
                }
            }
        } else if message.get("role").and_then(Value::as_str) == Some("tool") {
            let id = string_at(message, "tool_call_id");
            if !id.is_empty() {
                tool_outputs.insert(id, message.get("content").cloned().unwrap_or(Value::Null));
            }
        }
    }

    let mut system = Vec::new();
    let mut contents = Vec::new();
    for message in messages {
        let role = string_at(message, "role");
        let content = message.get("content");
        if matches!(role.as_str(), "system" | "developer") && messages.len() > 1 {
            append_text_parts(content, &mut system, false);
            continue;
        }
        if role == "user"
            || (matches!(role.as_str(), "system" | "developer") && messages.len() == 1)
        {
            let mut parts = Vec::new();
            append_user_parts(content, &mut parts);
            contents.push(json!({"role":"user","parts":parts}));
            continue;
        }
        if role != "assistant" {
            continue;
        }
        let mut parts = Vec::new();
        if let Some(reasoning) = message
            .get("reasoning_content")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            parts.push(
                json!({"text":reasoning,"thought":true,"thoughtSignature":GEMINI_THOUGHT_BYPASS}),
            );
        }
        append_assistant_parts(content, &mut parts);
        let calls = message.get("tool_calls").and_then(Value::as_array);
        if let Some(calls) = calls {
            let mut ids = Vec::new();
            for call in calls {
                if call.get("type").and_then(Value::as_str) != Some("function") {
                    continue;
                }
                let name = sanitize_function_name(
                    call.pointer("/function/name")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                );
                if name.is_empty() {
                    continue;
                }
                let arguments = call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .and_then(|raw| serde_json::from_str::<Value>(raw).ok());
                let mut function = json!({"name":name});
                if let Some(arguments) = arguments {
                    function["args"] = arguments;
                }
                parts.push(json!({
                    "functionCall":function,
                    "thoughtSignature":tool_signature(call)
                }));
                let id = string_at(call, "id");
                if !id.is_empty() {
                    ids.push(id);
                }
            }
            if !parts.is_empty() {
                contents.push(json!({"role":"model","parts":parts}));
            }
            let responses = ids
                .iter()
                .filter_map(|id| call_names.get(id).map(|name| (id, name)))
                .map(|(id, name)| {
                    let result = tool_outputs
                        .get(id)
                        .map(|value| serde_json::to_string(value).unwrap_or_default())
                        .unwrap_or_else(|| "{}".into());
                    json!({"functionResponse":{"name":sanitize_function_name(name),"response":{"result":result}}})
                })
                .collect::<Vec<_>>();
            if !responses.is_empty() {
                contents.push(json!({"role":"user","parts":responses}));
            }
        } else if !parts.is_empty() {
            contents.push(json!({"role":"model","parts":parts}));
        }
    }
    if contents
        .last()
        .and_then(|value| value.get("role"))
        .and_then(Value::as_str)
        == Some("model")
    {
        contents.pop();
    }
    (system, contents)
}

fn append_text_parts(content: Option<&Value>, output: &mut Vec<Value>, skip_empty: bool) {
    match content {
        Some(Value::String(text)) if !skip_empty || !text.is_empty() => {
            output.push(json!({"text":text}))
        }
        Some(Value::Object(object))
            if object.get("type").and_then(Value::as_str) == Some("text") =>
        {
            let text = object.get("text").map(value_string).unwrap_or_default();
            if !skip_empty || !text.is_empty() {
                output.push(json!({"text":text}));
            }
        }
        Some(Value::Array(parts)) => {
            for part in parts {
                let text = part.get("text").map(value_string).unwrap_or_default();
                if !skip_empty || !text.is_empty() {
                    output.push(json!({"text":text}));
                }
            }
        }
        _ => {}
    }
}

fn append_user_parts(content: Option<&Value>, output: &mut Vec<Value>) {
    if matches!(content, Some(Value::String(_))) {
        append_text_parts(content, output, false);
        return;
    }
    for part in content.and_then(Value::as_array).into_iter().flatten() {
        match part.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => append_text_parts(Some(part), output, true),
            "image_url" => append_data_url(part.pointer("/image_url/url"), output, true),
            "video_url" => append_data_url(part.pointer("/video_url/url"), output, false),
            "file" => {
                if let Some((mime, data)) = normalize_file(part.get("file")) {
                    output.push(json!({"inlineData":{"mime_type":mime,"data":data}}));
                }
            }
            "input_audio" => {
                let data = part
                    .pointer("/input_audio/data")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if !data.is_empty() {
                    let format = part
                        .pointer("/input_audio/format")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    output.push(json!({"inlineData":{"mime_type":audio_mime(format),"data":data}}));
                }
            }
            _ => {}
        }
    }
}

fn append_assistant_parts(content: Option<&Value>, output: &mut Vec<Value>) {
    if let Some(Value::String(text)) = content {
        if !text.is_empty() {
            output.push(json!({"text":text}));
        }
        return;
    }
    for part in content.and_then(Value::as_array).into_iter().flatten() {
        match part.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => append_text_parts(Some(part), output, true),
            "image_url" => append_data_url(part.pointer("/image_url/url"), output, true),
            _ => {}
        }
    }
}

fn append_data_url(value: Option<&Value>, output: &mut Vec<Value>, signed: bool) {
    let Some((mime, data)) = value.and_then(Value::as_str).and_then(data_url) else {
        return;
    };
    let mut part = json!({"inlineData":{"mime_type":mime,"data":data}});
    if signed {
        part["thoughtSignature"] = Value::String(GEMINI_THOUGHT_BYPASS.into());
    }
    output.push(part);
}

fn convert_tools(root: &Value) -> Option<Value> {
    let mut functions = Vec::new();
    let mut others = Vec::new();
    for tool in root.get("tools")?.as_array()? {
        if tool.get("type").and_then(Value::as_str) == Some("function") {
            let Some(function) = tool.get("function").and_then(Value::as_object) else {
                continue;
            };
            let mut function = function.clone();
            function.remove("strict");
            let parameters = function
                .remove("parameters")
                .map(|value| clean_gemini_tool_schema_with_root_placeholder_parity(&value))
                .unwrap_or_else(|| json!({"type":"object","properties":{}}));
            function.insert("parametersJsonSchema".into(), parameters);
            let name = function.get("name").map(value_string).unwrap_or_default();
            function.insert("name".into(), Value::String(sanitize_function_name(&name)));
            functions.push(Value::Object(function));
        }
        for (source, target) in [
            ("google_search", "googleSearch"),
            ("code_execution", "codeExecution"),
            ("url_context", "urlContext"),
        ] {
            if let Some(value) = tool.get(source) {
                others.push(json!({target:value}));
            }
        }
    }
    let mut tools = Vec::new();
    if !functions.is_empty() {
        tools.push(json!({"functionDeclarations":functions}));
    }
    tools.extend(others);
    (!tools.is_empty()).then_some(Value::Array(tools))
}

fn apply_response_format(root: &Value, generation: &mut Map<String, Value>) {
    let Some(format) = root.get("response_format") else {
        return;
    };
    match format
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "json_object" => {
            generation.insert(
                "responseMimeType".into(),
                Value::String("application/json".into()),
            );
        }
        "json_schema" => {
            generation.insert(
                "responseMimeType".into(),
                Value::String("application/json".into()),
            );
            generation.remove("responseSchema");
            if let Some(schema) = format.pointer("/json_schema/schema") {
                generation.insert("responseJsonSchema".into(), schema.clone());
            }
        }
        _ => {}
    }
}

pub(super) fn sanitized_name_map(root: &[u8]) -> HashMap<String, String> {
    let value = serde_json::from_slice::<Value>(root).unwrap_or(Value::Null);
    value
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            tool.pointer("/function/name")
                .or_else(|| tool.get("name"))
                .map(value_string)
        })
        .map(|name| (sanitize_function_name(&name), name))
        .collect()
}

/// Pinned Go keeps root `_` and `reason` placeholders while its recursive
/// cleaner removes the same transport-only fields below the tool root.
fn clean_gemini_tool_schema_with_root_placeholder_parity(schema: &Value) -> Value {
    let mut cleaned = clean_json_schema_for_gemini(schema);
    let (Some(source), Some(target)) = (schema.as_object(), cleaned.as_object_mut()) else {
        return cleaned;
    };
    let Some(source_properties) = source.get("properties").and_then(Value::as_object) else {
        return cleaned;
    };
    let target_properties = target
        .entry("properties")
        .or_insert_with(|| Value::Object(Map::new()));
    let Some(target_properties) = target_properties.as_object_mut() else {
        return cleaned;
    };
    for name in ["_", "reason"] {
        if let Some(property) = source_properties.get(name) {
            target_properties.insert(name.to_owned(), clean_json_schema_for_gemini(property));
        }
    }
    if let Some(required) = source.get("required").and_then(Value::as_array) {
        let retained = required
            .iter()
            .filter(|name| {
                name.as_str()
                    .is_some_and(|name| target_properties.contains_key(name))
            })
            .cloned()
            .collect::<Vec<_>>();
        if retained.is_empty() {
            target.remove("required");
        } else {
            target.insert("required".into(), Value::Array(retained));
        }
    }
    cleaned
}

pub(super) fn sanitize_function_name(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let mut value = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || "_.:-".contains(character) {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if !value
        .as_bytes()
        .first()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
    {
        if value.len() >= 64 {
            value.truncate(63);
        }
        value.insert(0, '_');
    }
    value.truncate(value.len().min(64));
    value
}

fn tool_signature(tool: &Value) -> String {
    for pointer in [
        "/extra_content/google/thought_signature",
        "/function/extra_content/google/thought_signature",
        "/thoughtSignature",
        "/thought_signature",
    ] {
        if let Some(raw) = tool.pointer(pointer).and_then(Value::as_str) {
            return compatible_gemini_signature(raw)
                .unwrap_or_else(|| GEMINI_THOUGHT_BYPASS.to_owned());
        }
    }
    GEMINI_THOUGHT_BYPASS.to_owned()
}

fn normalize_file(file: Option<&Value>) -> Option<(String, String)> {
    let file = file?;
    let raw = file.get("file_data")?.as_str()?;
    if let Some((mime, data)) = data_url(raw) {
        return Some((mime.into(), data.into()));
    }
    let filename = file.get("filename").and_then(Value::as_str).unwrap_or("");
    let mime = match filename
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "pdf" => "application/pdf",
        "txt" | "md" | "csv" | "json" => "text/plain",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "mp4" => "video/mp4",
        _ => return None,
    };
    Some((mime.into(), raw.into()))
}

fn data_url(raw: &str) -> Option<(&str, &str)> {
    raw.strip_prefix("data:")?
        .split_once(";base64,")
        .or_else(|| raw.strip_prefix("data:")?.split_once(','))
}

fn audio_mime(format: &str) -> String {
    match format {
        "" | "wav" => "audio/wav".into(),
        "mp3" => "audio/mpeg".into(),
        "ogg" => "audio/ogg".into(),
        "flac" => "audio/flac".into(),
        "aac" => "audio/aac".into(),
        "webm" => "audio/webm".into(),
        "pcm16" => "audio/pcm".into(),
        "g711_ulaw" | "g711_alaw" => "audio/basic".into(),
        other => format!("audio/{other}"),
    }
}

fn string_at(value: &Value, key: &str) -> String {
    value.get(key).map(value_string).unwrap_or_default()
}

fn value_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        value => serde_json::to_string(value).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::convert_openai_chat_request_to_gemini;
    use serde_json::Value;

    #[test]
    fn maps_controls_media_tools_and_history() {
        let output: Value = serde_json::from_slice(&convert_openai_chat_request_to_gemini(
            "gemini-3",
            br#"{"reasoning_effort":"high","max_tokens":9,"messages":[{"role":"system","content":"exact"},{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"read file","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"ok"},{"role":"user","content":"next"}],"tools":[{"type":"function","function":{"name":"read file"}}]}"#,
            true,
        ))
        .unwrap();
        assert_eq!(output["model"], "gemini-3");
        assert_eq!(
            output["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "high"
        );
        assert_eq!(output["generationConfig"]["maxOutputTokens"], 9);
        assert_eq!(
            output["contents"][0]["parts"][0]["inlineData"]["data"],
            "AAAA"
        );
        assert_eq!(
            output["contents"][1]["parts"][0]["functionCall"]["name"],
            "read_file"
        );
        assert_eq!(output["contents"][2]["role"], "user");
        assert_eq!(
            output["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["type"],
            "object"
        );
    }
}
