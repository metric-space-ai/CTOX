// ref: internal/translator/openai/gemini/openai_gemini_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

//! Gemini-native request to OpenAI Chat Completions conversion.
//!
//! The source implementation generates random tool-call IDs. CTOX keeps that
//! authority request-local: IDs are derived from the immutable request plus a
//! per-conversion sequence. This preserves uniqueness inside one request and
//! avoids process-global counters or ambient entropy.

use serde_json::{json, Map, Number, Value};
use sha2::{Digest, Sha256};

use crate::internal::thinking::convert_budget_to_level;

const TOOL_ID_ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

#[derive(Debug)]
struct ToolCallIdAuthority {
    seed: [u8; 32],
    sequence: u64,
}

impl ToolCallIdAuthority {
    fn new(model_name: &str, input: &[u8]) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"ctox:openai-gemini:tool-call-id:v1\0");
        hash.update(model_name.as_bytes());
        hash.update(b"\0");
        hash.update(input);
        Self {
            seed: hash.finalize().into(),
            sequence: 0,
        }
    }

    fn next_id(&mut self) -> String {
        self.sequence = self.sequence.saturating_add(1);
        let mut hash = Sha256::new();
        hash.update(self.seed);
        hash.update(self.sequence.to_be_bytes());
        let digest = hash.finalize();
        let mut id = String::with_capacity(29);
        id.push_str("call_");
        for byte in digest.iter().take(24) {
            id.push(TOOL_ID_ALPHABET[usize::from(*byte) % TOOL_ID_ALPHABET.len()] as char);
        }
        id
    }
}

pub fn convert_gemini_request_to_openai(
    model_name: &str,
    input_raw_json: &[u8],
    stream: bool,
) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(input_raw_json).unwrap_or(Value::Null);
    let mut output = Map::new();
    output.insert("model".into(), Value::String(model_name.to_owned()));
    output.insert("messages".into(), Value::Array(Vec::new()));

    if let Some(generation) = root.get("generationConfig") {
        apply_generation_config(generation, &mut output);
    }
    output.insert("stream".into(), Value::Bool(stream));
    if let Some(service_tier) = root.get("service_tier").and_then(Value::as_str) {
        output.insert(
            "service_tier".into(),
            Value::String(service_tier.to_owned()),
        );
    }

    let mut id_authority = ToolCallIdAuthority::new(model_name, input_raw_json);
    output.insert(
        "messages".into(),
        Value::Array(convert_messages(&root, &mut id_authority)),
    );

    if let Some(tools) = convert_tools(root.get("tools")) {
        output.insert("tools".into(), tools);
    }
    if let Some(tool_choice) = convert_tool_choice(&root) {
        output.insert("tool_choice".into(), tool_choice);
    }

    serde_json::to_vec(&Value::Object(output)).unwrap_or_default()
}

fn apply_generation_config(generation: &Value, output: &mut Map<String, Value>) {
    let Some(generation) = generation.as_object() else {
        return;
    };
    for (source, target, integer) in [
        ("temperature", "temperature", false),
        ("maxOutputTokens", "max_tokens", true),
        ("topP", "top_p", false),
        ("topK", "top_k", true),
        ("candidateCount", "n", true),
    ] {
        let Some(value) = generation.get(source) else {
            continue;
        };
        if let Some(number) = numeric_value(value, integer) {
            output.insert(target.into(), number);
        }
    }

    if let Some(stops) = generation.get("stopSequences").and_then(Value::as_array) {
        let values = stops
            .iter()
            .map(|value| Value::String(gjson_string(value)))
            .collect::<Vec<_>>();
        if !values.is_empty() {
            output.insert("stop".into(), Value::Array(values));
        }
    }

    if let Some(modalities) = generation
        .get("responseModalities")
        .and_then(Value::as_array)
    {
        let modalities = modalities
            .iter()
            .filter_map(|value| value.as_str().map(str::trim))
            .filter_map(|value| match value.to_ascii_lowercase().as_str() {
                "text" => Some(Value::String("text".into())),
                "image" => Some(Value::String("image".into())),
                "audio" => Some(Value::String("audio".into())),
                _ => None,
            })
            .collect::<Vec<_>>();
        if !modalities.is_empty() {
            output.insert("modalities".into(), Value::Array(modalities));
        }
    }

    let Some(thinking) = generation.get("thinkingConfig").and_then(Value::as_object) else {
        return;
    };
    if let Some(level) = thinking
        .get("thinkingLevel")
        .or_else(|| thinking.get("thinking_level"))
    {
        let effort = gjson_string(level).trim().to_ascii_lowercase();
        if !effort.is_empty() {
            output.insert("reasoning_effort".into(), Value::String(effort));
        }
        return;
    }
    if let Some(budget) = thinking
        .get("thinkingBudget")
        .or_else(|| thinking.get("thinking_budget"))
        .and_then(gjson_i64)
        .and_then(|value| isize::try_from(value).ok())
        .and_then(convert_budget_to_level)
    {
        output.insert(
            "reasoning_effort".into(),
            Value::String(budget.as_str().to_owned()),
        );
    }
}

fn convert_messages(root: &Value, ids: &mut ToolCallIdAuthority) -> Vec<Value> {
    let mut messages = Vec::new();
    if let Some(system) = root
        .get("systemInstruction")
        .or_else(|| root.get("system_instruction"))
    {
        let parts = system
            .get("parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .flat_map(openai_parts_from_gemini_part)
            .collect::<Vec<_>>();
        if !parts.is_empty() {
            messages.push(json!({"role":"system","content":parts}));
        }
    }

    let mut tool_call_ids = Vec::new();
    let mut consumed = 0usize;
    for content in root
        .get("contents")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let role = match content.get("role").and_then(Value::as_str) {
            Some("model") => "assistant",
            Some(role) => role,
            None => "",
        };
        let mut text = String::new();
        let mut content_items = Vec::new();
        let mut only_text_content = true;
        let mut tool_calls = Vec::new();

        for part in content
            .get("parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(value) = part.get("text") {
                let value = gjson_string(value);
                text.push_str(&value);
                content_items.push(json!({"type":"text","text":value}));
            }
            for media in openai_media_parts_from_gemini_part(part) {
                only_text_content = false;
                content_items.push(media);
            }

            if let Some(call) = part.get("functionCall") {
                let id = explicit_gemini_tool_id(call).unwrap_or_else(|| ids.next_id());
                tool_call_ids.push(id.clone());
                let arguments = call
                    .get("args")
                    .map(raw_json_string)
                    .unwrap_or_else(|| "{}".into());
                tool_calls.push(json!({
                    "id":id,
                    "type":"function",
                    "function":{
                        "name":call.get("name").map(gjson_string).unwrap_or_default(),
                        "arguments":arguments
                    }
                }));
            }

            if let Some(response) = part.get("functionResponse") {
                let explicit = explicit_gemini_tool_id(response);
                let id = if let Some(id) = explicit {
                    if tool_call_ids.get(consumed) == Some(&id) {
                        consumed += 1;
                    }
                    id
                } else if let Some(id) = tool_call_ids.get(consumed).cloned() {
                    consumed += 1;
                    id
                } else {
                    ids.next_id()
                };
                let response = response.get("response");
                let content = response
                    .and_then(|value| value.get("content"))
                    .or(response)
                    .map(raw_json_string)
                    .unwrap_or_default();
                messages.push(json!({"role":"tool","tool_call_id":id,"content":content}));
            }
        }

        let message_content = if content_items.is_empty() || only_text_content {
            Value::String(text)
        } else {
            Value::Array(content_items)
        };
        let mut message = json!({"role":role,"content":message_content});
        if !tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(tool_calls);
        }
        messages.push(message);
    }
    messages
}

fn openai_parts_from_gemini_part(part: &Value) -> Vec<Value> {
    let mut output = Vec::new();
    if let Some(text) = part.get("text") {
        output.push(json!({"type":"text","text":gjson_string(text)}));
    }
    output.extend(openai_media_parts_from_gemini_part(part));
    output
}

fn openai_media_parts_from_gemini_part(part: &Value) -> Vec<Value> {
    openai_content_part_from_gemini_inline_data(part)
        .into_iter()
        .chain(openai_content_part_from_gemini_file_data(part))
        .collect()
}

fn openai_content_part_from_gemini_inline_data(part: &Value) -> Option<Value> {
    let data = part.get("inlineData").or_else(|| part.get("inline_data"))?;
    let mime = data
        .get("mimeType")
        .or_else(|| data.get("mime_type"))
        .map(gjson_string)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "application/octet-stream".into());
    let bytes = data.get("data").map(gjson_string)?;
    if bytes.is_empty() {
        return None;
    }
    let data_url = format!("data:{mime};base64,{bytes}");
    let lower = mime.to_ascii_lowercase();
    Some(if lower.starts_with("image/") {
        json!({"type":"image_url","image_url":{"url":data_url}})
    } else if lower.starts_with("audio/") {
        json!({"type":"input_audio","input_audio":{"data":bytes,"format":openai_input_audio_format_from_mime(&mime)}})
    } else if lower.starts_with("video/") {
        json!({"type":"video_url","video_url":{"url":data_url}})
    } else {
        json!({"type":"file","file":{"filename":openai_file_name_from_mime(&mime),"file_data":bytes}})
    })
}

fn openai_content_part_from_gemini_file_data(part: &Value) -> Option<Value> {
    let data = part.get("fileData").or_else(|| part.get("file_data"))?;
    let uri = data
        .get("fileUri")
        .or_else(|| data.get("file_uri"))
        .map(gjson_string)?;
    if uri.is_empty() {
        return None;
    }
    let mime = data
        .get("mimeType")
        .or_else(|| data.get("mime_type"))
        .map(gjson_string)
        .unwrap_or_default();
    let lower = mime.to_ascii_lowercase();
    Some(if lower.starts_with("image/") {
        json!({"type":"image_url","image_url":{"url":uri}})
    } else if lower.starts_with("video/") {
        json!({"type":"video_url","video_url":{"url":uri}})
    } else if lower.starts_with("application/") || lower.starts_with("text/") {
        json!({"type":"file","file":{"filename":openai_file_name_from_mime(&mime),"file_url":uri}})
    } else {
        let suffix = if mime.is_empty() {
            String::new()
        } else {
            format!(" (Type: {mime})")
        };
        json!({"type":"text","text":format!("File: {uri}{suffix}")})
    })
}

fn convert_tools(tools: Option<&Value>) -> Option<Value> {
    let mut output = Vec::new();
    for tool in tools.and_then(Value::as_array).into_iter().flatten() {
        for declaration in tool
            .get("functionDeclarations")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let mut function = Map::new();
            function.insert(
                "name".into(),
                Value::String(
                    declaration
                        .get("name")
                        .map(gjson_string)
                        .unwrap_or_default(),
                ),
            );
            function.insert(
                "description".into(),
                Value::String(
                    declaration
                        .get("description")
                        .map(gjson_string)
                        .unwrap_or_default(),
                ),
            );
            if let Some(parameters) = declaration
                .get("parameters")
                .or_else(|| declaration.get("parametersJsonSchema"))
            {
                function.insert("parameters".into(), parameters.clone());
            }
            output.push(json!({"type":"function","function":function}));
        }
    }
    (!output.is_empty()).then_some(Value::Array(output))
}

fn convert_tool_choice(root: &Value) -> Option<Value> {
    let config = root.pointer("/toolConfig/functionCallingConfig")?;
    match config.get("mode").map(gjson_string).as_deref() {
        Some("NONE") => Some(Value::String("none".into())),
        Some("AUTO") => Some(Value::String("auto".into())),
        Some("ANY") => {
            let names = config.get("allowedFunctionNames").and_then(Value::as_array);
            if let Some([name]) = names.map(Vec::as_slice) {
                Some(json!({"type":"function","function":{"name":gjson_string(name)}}))
            } else {
                Some(Value::String("required".into()))
            }
        }
        _ => None,
    }
}

fn explicit_gemini_tool_id(value: &Value) -> Option<String> {
    value
        .get("id")
        .or_else(|| value.get("call_id"))
        .map(gjson_string)
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn openai_input_audio_format_from_mime(mime: &str) -> &'static str {
    match mime.trim().to_ascii_lowercase().as_str() {
        "audio/wav" | "audio/wave" | "audio/x-wav" => "wav",
        "audio/flac" => "flac",
        "audio/opus" | "audio/ogg" => "opus",
        "audio/pcm" | "audio/l16" => "pcm16",
        _ => "mp3",
    }
}

fn openai_file_name_from_mime(mime: &str) -> &'static str {
    match mime.trim().to_ascii_lowercase().as_str() {
        "application/pdf" => "document.pdf",
        "text/plain" => "document.txt",
        "text/csv" => "document.csv",
        "application/json" => "document.json",
        "application/xml" | "text/xml" => "document.xml",
        value if value.starts_with("video/") => "video",
        _ => "document",
    }
}

fn numeric_value(value: &Value, integer: bool) -> Option<Value> {
    if integer {
        gjson_i64(value).map(Value::from)
    } else {
        let number = match value {
            Value::Number(value) => value.clone(),
            Value::String(value) => Number::from_f64(value.parse().ok()?)?,
            _ => return None,
        };
        Some(Value::Number(number))
    }
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

fn raw_json_string(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}
