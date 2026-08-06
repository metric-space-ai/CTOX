// ref: internal/translator/codex/gemini/codex_gemini_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::{json, Map, Value};

use crate::internal::thinking::convert_budget_to_level;

pub fn convert_gemini_request_to_codex(model_name: &str, input: &[u8], _stream: bool) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(input).unwrap_or(Value::Null);
    let short_names = declared_tool_names(&root);
    let mut pending = VecDeque::new();
    let mut output = json!({"model":model_name,"instructions":"","input":[]});

    if normalize_service_tier(root.get("service_tier")).is_some() {
        output["service_tier"] = Value::String("priority".to_owned());
    }
    let mut items = Vec::new();
    append_system_instruction(&root, &mut items);
    append_contents(&root, &short_names, &mut pending, &mut items);
    output["input"] = Value::Array(items);

    if let Some(tools) = convert_tools(&root, &short_names) {
        output["tools"] = Value::Array(tools);
        output["tool_choice"] = Value::String("auto".to_owned());
    }
    output["parallel_tool_calls"] = Value::Bool(true);
    apply_tool_choice(&root, &short_names, &mut output);
    output["reasoning"] = json!({"effort":reasoning_effort(&root)});
    output["stream"] = Value::Bool(true);
    output["store"] = Value::Bool(false);
    output["include"] = json!(["reasoning.encrypted_content"]);
    if let Some(tools) = output.get_mut("tools") {
        lowercase_schema_types(tools);
    }
    serde_json::to_vec(&output).unwrap_or_default()
}

fn append_system_instruction(root: &Value, items: &mut Vec<Value>) {
    let parts = root
        .pointer("/system_instruction/parts")
        .or_else(|| root.pointer("/systemInstruction/parts"))
        .and_then(Value::as_array);
    let content = parts
        .into_iter()
        .flatten()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .map(|text| json!({"type":"input_text","text":text}))
        .collect::<Vec<_>>();
    if !content.is_empty() {
        items.push(json!({"type":"message","role":"developer","content":content}));
    }
}

fn append_contents(
    root: &Value,
    short_names: &HashMap<String, String>,
    pending: &mut VecDeque<String>,
    items: &mut Vec<Value>,
) {
    let Some(contents) = root.get("contents").and_then(Value::as_array) else {
        return;
    };
    for item in contents {
        let role = match item.get("role").and_then(Value::as_str).unwrap_or("") {
            "model" => "assistant",
            value => value,
        };
        let Some(parts) = item.get("parts").and_then(Value::as_array) else {
            continue;
        };
        for part in parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                let kind = if role == "assistant" {
                    "output_text"
                } else {
                    "input_text"
                };
                items.push(message(role, json!({"type":kind,"text":text})));
                continue;
            }
            if let Some(content) = content_from_inline_data(part) {
                items.push(message(role, content));
                continue;
            }
            if let Some(content) = content_from_file_data(part) {
                items.push(message(role, content));
                continue;
            }
            if let Some(call) = part.get("functionCall") {
                let name = call
                    .get("name")
                    .and_then(Value::as_str)
                    .map(|name| shortened(short_names, name))
                    .unwrap_or_default();
                let call_id = call_id(call).unwrap_or_else(random_call_id);
                pending.push_back(call_id.clone());
                let arguments = call
                    .get("args")
                    .map(|value| value.to_string())
                    .unwrap_or_default();
                items.push(json!({"type":"function_call","name":name,"arguments":arguments,"call_id":call_id}));
                continue;
            }
            if let Some(response) = part.get("functionResponse") {
                let custom = call_id(response);
                let call_id = if let Some(custom) = custom {
                    if let Some(index) = pending.iter().position(|id| id == &custom) {
                        pending.remove(index);
                    }
                    custom
                } else {
                    pending.pop_front().unwrap_or_else(random_call_id)
                };
                let output = response
                    .pointer("/response/result")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| response.get("response").map(Value::to_string))
                    .unwrap_or_default();
                items
                    .push(json!({"type":"function_call_output","call_id":call_id,"output":output}));
            }
        }
    }
}

fn message(role: &str, content: Value) -> Value {
    json!({"type":"message","role":role,"content":[content]})
}

fn call_id(value: &Value) -> Option<String> {
    ["id", "call_id"]
        .into_iter()
        .find_map(|key| value.get(key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn random_call_id() -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut random = [0_u8; 24];
    if getrandom::fill(&mut random).is_err() {
        random = [0x5a; 24];
    }
    let suffix = random
        .into_iter()
        .map(|byte| ALPHABET[usize::from(byte) % ALPHABET.len()] as char)
        .collect::<String>();
    format!("call_{suffix}")
}

fn declared_tool_names(root: &Value) -> HashMap<String, String> {
    let names = root
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|tool| {
            tool.get("functionDeclarations")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|function| function.get("name").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    build_short_name_map(&names)
}

fn convert_tools(root: &Value, names: &HashMap<String, String>) -> Option<Vec<Value>> {
    let mut result = Vec::new();
    for tool in root.get("tools")?.as_array()? {
        for function in tool.get("functionDeclarations")?.as_array()? {
            let mut converted = Map::new();
            converted.insert("type".to_owned(), Value::String("function".to_owned()));
            if let Some(name) = function.get("name").and_then(Value::as_str) {
                converted.insert("name".to_owned(), Value::String(shortened(names, name)));
            }
            if let Some(description) = function.get("description") {
                converted.insert("description".to_owned(), description.clone());
            }
            if let Some(parameters) = function
                .get("parameters")
                .or_else(|| function.get("parametersJsonSchema"))
            {
                converted.insert("parameters".to_owned(), clean_parameters(parameters));
            }
            converted.insert("strict".to_owned(), Value::Bool(false));
            result.push(Value::Object(converted));
        }
    }
    Some(result)
}

pub(super) fn clean_parameters(parameters: &Value) -> Value {
    let mut cleaned = parameters.clone();
    if let Some(object) = cleaned.as_object_mut() {
        object.remove("$schema");
        if object.get("additionalProperties") != Some(&Value::Bool(false)) {
            object.insert("additionalProperties".to_owned(), Value::Bool(false));
        }
    }
    cleaned
}

fn apply_tool_choice(root: &Value, names: &HashMap<String, String>, output: &mut Value) {
    let Some(config) = root.pointer("/toolConfig/functionCallingConfig") else {
        return;
    };
    match config.get("mode").and_then(Value::as_str).unwrap_or("") {
        "NONE" => output["tool_choice"] = Value::String("none".to_owned()),
        "AUTO" => output["tool_choice"] = Value::String("auto".to_owned()),
        "ANY" => {
            let allowed = config.get("allowedFunctionNames").and_then(Value::as_array);
            output["tool_choice"] = if let Some([name]) = allowed.map(Vec::as_slice) {
                let name = name.as_str().unwrap_or_default();
                json!({"type":"function","name":shortened(names, name)})
            } else {
                Value::String("required".to_owned())
            };
        }
        _ => {}
    }
}

fn reasoning_effort(root: &Value) -> String {
    let generation = root.get("generationConfig").unwrap_or(&Value::Null);
    let level = generation
        .get("thinkingLevel")
        .or_else(|| generation.get("thinking_level"))
        .or_else(|| generation.pointer("/thinkingConfig/thinkingLevel"))
        .or_else(|| generation.pointer("/thinkingConfig/thinking_level"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if let Some(level) = level {
        return level.to_ascii_lowercase();
    }
    let budget = generation
        .pointer("/thinkingConfig/thinkingBudget")
        .or_else(|| generation.pointer("/thinkingConfig/thinking_budget"))
        .and_then(Value::as_i64);
    budget
        .and_then(|value| isize::try_from(value).ok())
        .and_then(convert_budget_to_level)
        .map(|level| level.to_string())
        .unwrap_or_else(|| "medium".to_owned())
}

fn normalize_service_tier(value: Option<&Value>) -> Option<()> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .filter(|value| matches!(value.as_str(), "priority" | "fast"))
        .map(|_| ())
}

fn content_from_inline_data(part: &Value) -> Option<Value> {
    let data = part.get("inlineData").or_else(|| part.get("inline_data"))?;
    let mime = data
        .get("mimeType")
        .or_else(|| data.get("mime_type"))?
        .as_str()?;
    let bytes = data.get("data")?.as_str()?;
    let lower = mime.to_ascii_lowercase();
    if lower.starts_with("image/") {
        Some(json!({"type":"input_image","image_url":format!("data:{mime};base64,{bytes}")}))
    } else if lower.starts_with("audio/") {
        Some(json!({"type":"input_audio","input_audio":{"data":bytes,"format":audio_format(mime)}}))
    } else {
        Some(json!({"type":"input_file","file_data":bytes,"filename":file_name(mime)}))
    }
}

fn content_from_file_data(part: &Value) -> Option<Value> {
    let data = part.get("fileData").or_else(|| part.get("file_data"))?;
    let uri = data
        .get("fileUri")
        .or_else(|| data.get("file_uri"))?
        .as_str()?;
    let mime = data
        .get("mimeType")
        .or_else(|| data.get("mime_type"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let lower = mime.to_ascii_lowercase();
    if lower.starts_with("image/") {
        Some(json!({"type":"input_image","image_url":uri}))
    } else if ["video/", "application/", "text/"]
        .iter()
        .any(|prefix| lower.starts_with(prefix))
    {
        Some(json!({"type":"input_file","file_url":uri,"filename":file_name(mime)}))
    } else {
        let suffix = if mime.is_empty() {
            String::new()
        } else {
            format!(" (Type: {mime})")
        };
        Some(json!({"type":"input_text","text":format!("File: {uri}{suffix}")}))
    }
}

fn audio_format(mime: &str) -> &'static str {
    match mime.trim().to_ascii_lowercase().as_str() {
        "audio/wav" | "audio/wave" | "audio/x-wav" => "wav",
        "audio/flac" => "flac",
        "audio/opus" | "audio/ogg" => "opus",
        "audio/pcm" | "audio/l16" => "pcm16",
        _ => "mp3",
    }
}

fn file_name(mime: &str) -> &'static str {
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

fn shortened(names: &HashMap<String, String>, name: &str) -> String {
    names
        .get(name)
        .cloned()
        .unwrap_or_else(|| shorten_name(name))
}

fn shorten_name(name: &str) -> String {
    const LIMIT: usize = 64;
    if name.len() <= LIMIT {
        return name.to_owned();
    }
    if name.starts_with("mcp__") {
        if let Some(index) = name.rfind("__") {
            let candidate = format!("mcp__{}", &name[index + 2..]);
            return candidate.chars().take(LIMIT).collect();
        }
    }
    name.chars().take(LIMIT).collect()
}

pub(super) fn build_short_name_map(names: &[String]) -> HashMap<String, String> {
    let mut result = HashMap::new();
    let mut used = HashSet::new();
    for name in names {
        let base = shorten_name(name);
        let mut candidate = base.clone();
        let mut suffix = 1;
        while used.contains(&candidate) {
            let tail = format!("_{suffix}");
            let keep = 64_usize.saturating_sub(tail.len());
            candidate = format!("{}{}", base.chars().take(keep).collect::<String>(), tail);
            suffix += 1;
        }
        used.insert(candidate.clone());
        result.insert(name.clone(), candidate);
    }
    result
}

fn lowercase_schema_types(value: &mut Value) {
    match value {
        Value::Array(values) => values.iter_mut().for_each(lowercase_schema_types),
        Value::Object(object) => {
            if let Some(Value::String(kind)) = object.get_mut("type") {
                kind.make_ascii_lowercase();
            }
            object.values_mut().for_each(lowercase_schema_types);
        }
        _ => {}
    }
}
