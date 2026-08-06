// ref: internal/translator/antigravity/interactions/interactions_antigravity_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{HashMap, HashSet};

use serde_json::{json, Map, Value};

use crate::internal::translator::common::normalize_openai_file_data;
use crate::internal::util::{map_sanitized_function_name, sanitized_function_name_map};

pub fn convert_interactions_request_to_antigravity(
    model_name: &str,
    input: &[u8],
    stream: bool,
) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(input).unwrap_or(Value::Null);
    let names = sanitized_function_name_map(input);
    let mut request = Map::new();
    request.insert(
        "contents".into(),
        Value::Array(convert_input(root.get("input"))),
    );
    if stream || root.get("stream").and_then(Value::as_bool) == Some(true) {
        request.insert("stream".into(), Value::Bool(true));
    }
    copy_system(&root, &mut request);
    copy_generation_config(&root, &mut request);
    copy_tools(&root, &mut request, &names);
    attach_safety_settings(&mut request);
    let raw = serde_json::to_vec(&json!({
        "project":"", "request":Value::Object(request), "model":model_name,
    }))
    .unwrap_or_default();
    rewrite_interactions_function_names(raw, &names)
}

fn copy_system(root: &Value, request: &mut Map<String, Value>) {
    let Some(system) = root.get("system_instruction") else {
        return;
    };
    let instruction = if let Some(text) = system.as_str() {
        json!({"parts":[{"text":text}]})
    } else if system.get("parts").is_none() {
        system
            .get("text")
            .map(|text| json!({"parts":[{"text":string_value(text)}]}))
            .unwrap_or_else(|| system.clone())
    } else {
        system.clone()
    };
    request.insert("systemInstruction".into(), instruction);
}

fn copy_generation_config(root: &Value, request: &mut Map<String, Value>) {
    let mut generation = if let Some(config) = root.get("generation_config") {
        camel_case_value(config)
    } else {
        root.get("generationConfig")
            .cloned()
            .unwrap_or_else(|| json!({}))
    };
    if !generation.is_object() {
        generation = json!({});
    }
    for key in ["thinkingLevel", "thinkingBudget", "includeThoughts"] {
        if let Some(value) = generation.as_object_mut().and_then(|map| map.remove(key)) {
            generation["thinkingConfig"][key] = value;
        }
    }
    if let Some(summary) = generation
        .as_object_mut()
        .and_then(|map| map.remove("thinkingSummaries"))
    {
        if let Some(include) = thinking_summary(&summary) {
            generation["thinkingConfig"]["includeThoughts"] = Value::Bool(include);
        }
    }
    generation
        .as_object_mut()
        .map(|map| map.remove("toolChoice"));

    if let Some(reasoning) = root.get("reasoning") {
        let effort = reasoning
            .get("effort")
            .or_else(|| reasoning.get("thinking_level"))
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or("")
            .to_ascii_lowercase();
        if effort == "auto" {
            generation["thinkingConfig"]["thinkingBudget"] = Value::from(-1);
        } else if !effort.is_empty() {
            generation["thinkingConfig"]["thinkingLevel"] = Value::String(effort);
        }
        if let Some(include) = reasoning.get("summary").and_then(thinking_summary) {
            generation["thinkingConfig"]["includeThoughts"] = Value::Bool(include);
        }
    }
    copy_modalities(root, &mut generation);
    if generation.as_object().is_some_and(|map| !map.is_empty()) {
        request.insert("generationConfig".into(), generation);
    }
    copy_tool_choice(root, request);
}

fn copy_modalities(root: &Value, generation: &mut Value) {
    let Some(modalities) = root
        .get("response_modalities")
        .or_else(|| root.get("responseModalities"))
        .and_then(Value::as_array)
    else {
        return;
    };
    let mapped = modalities
        .iter()
        .filter_map(Value::as_str)
        .filter_map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "text" => Some("TEXT"),
            "image" => Some("IMAGE"),
            "audio" => Some("AUDIO"),
            _ => None,
        })
        .map(|value| Value::String(value.into()))
        .collect::<Vec<_>>();
    if !mapped.is_empty() {
        generation["responseModalities"] = Value::Array(mapped);
    }
}

fn copy_tool_choice(root: &Value, request: &mut Map<String, Value>) {
    let choice = root
        .get("tool_choice")
        .or_else(|| root.pointer("/generation_config/tool_choice"))
        .or_else(|| root.pointer("/generationConfig/toolChoice"));
    let Some(choice) = choice else { return };
    let (kind, name) = if let Some(kind) = choice.as_str() {
        (kind.trim().to_ascii_lowercase(), None)
    } else {
        let kind = choice
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        let name = match kind.as_str() {
            "function" => choice.pointer("/function/name"),
            "tool" => choice.get("name"),
            _ => None,
        }
        .map(string_value)
        .filter(|value| !value.trim().is_empty());
        (kind, name)
    };
    let mode = match kind.as_str() {
        "none" => "NONE",
        "auto" => "AUTO",
        "required" | "any" | "function" | "tool" => "ANY",
        _ => return,
    };
    let mut config = json!({"functionCallingConfig":{"mode":mode}});
    if let Some(name) = name {
        config["functionCallingConfig"]["allowedFunctionNames"] = json!([name]);
    }
    request.insert("toolConfig".into(), config);
}

fn convert_input(input: Option<&Value>) -> Vec<Value> {
    let mut contents = Vec::new();
    let Some(input) = input else { return contents };
    match input {
        Value::String(text) => push_text(&mut contents, "user", text, false),
        Value::Array(items) => {
            for item in items {
                append_step(&mut contents, item, "user");
            }
        }
        Value::Object(map) if map.get("steps").and_then(Value::as_array).is_some() => {
            let role = content_role(
                map.get("role").and_then(Value::as_str).unwrap_or(""),
                "user",
            );
            if let Some(steps) = map.get("steps").and_then(Value::as_array) {
                for step in steps {
                    append_step(&mut contents, step, role);
                }
            }
        }
        _ => append_step(&mut contents, input, "user"),
    }
    contents
}

fn append_step(contents: &mut Vec<Value>, step: &Value, default_role: &str) {
    if let Some(text) = step.as_str() {
        push_text(contents, default_role, text, false);
        return;
    }
    if let Some(steps) = step.get("steps").and_then(Value::as_array) {
        let role = content_role(
            step.get("role").and_then(Value::as_str).unwrap_or(""),
            default_role,
        );
        for child in steps {
            append_step(contents, child, role);
        }
        return;
    }
    match step.get("type").and_then(Value::as_str).unwrap_or("") {
        "model_output" => append_step_content(contents, "model", step, false),
        "thought" => append_step_content(contents, "model", step, true),
        "function_call" => append_function_call(contents, step),
        "function_result" => append_function_result(contents, step),
        "user_input" | "" => {
            if step.get("parts").is_some() {
                append_native_content(contents, step, default_role);
            } else {
                append_content_list(contents, default_role, step.get("content"));
            }
        }
        _ => {
            if step.get("parts").is_some() {
                append_native_content(contents, step, default_role);
            } else if step.get("content").is_some() {
                append_content_list(contents, default_role, step.get("content"));
            } else if let Some(text) = step.get("text") {
                push_text(contents, default_role, &string_value(text), false);
            }
        }
    }
}

fn append_native_content(contents: &mut Vec<Value>, step: &Value, default_role: &str) {
    let parts = step
        .get("parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(native_part)
        .collect::<Vec<_>>();
    if !parts.is_empty() {
        contents.push(json!({
            "role":content_role(step.get("role").and_then(Value::as_str).unwrap_or(""), default_role),
            "parts":parts,
        }));
    }
}

fn append_step_content(contents: &mut Vec<Value>, role: &str, step: &Value, thought: bool) {
    let Some(content) = step.get("content") else {
        return;
    };
    let parts = match content {
        Value::Array(items) => items
            .iter()
            .filter_map(|part| content_part(part, thought))
            .collect(),
        Value::Object(_) => content_part(content, thought).into_iter().collect(),
        Value::String(text) => vec![text_part(text, thought)],
        _ => Vec::new(),
    };
    if !parts.is_empty() {
        contents.push(json!({"role":role,"parts":parts}));
    }
}

fn append_content_list(contents: &mut Vec<Value>, role: &str, content: Option<&Value>) {
    let Some(content) = content else { return };
    match content {
        Value::Array(parts) => {
            for part in parts.iter().filter_map(|part| content_part(part, false)) {
                contents.push(json!({"role":content_role(role,"user"),"parts":[part]}));
            }
        }
        Value::Object(_) => {
            if let Some(part) = content_part(content, false) {
                contents.push(json!({"role":content_role(role,"user"),"parts":[part]}));
            }
        }
        Value::String(text) => push_text(contents, role, text, false),
        _ => {}
    }
}

fn append_function_call(contents: &mut Vec<Value>, step: &Value) {
    let mut call = json!({
        "name":string_value(step.get("name").unwrap_or(&Value::Null)),
        "args":step.get("arguments").cloned().unwrap_or_else(|| json!({})),
    });
    if let Some(id) = step.get("call_id").or_else(|| step.get("id")) {
        call["id"] = Value::String(string_value(id));
    }
    contents.push(json!({"role":"model","parts":[{"functionCall":call}]}));
}

fn append_function_result(contents: &mut Vec<Value>, step: &Value) {
    let mut response = json!({
        "name":string_value(step.get("name").unwrap_or(&Value::Null)),
        "response":step.get("result").cloned().unwrap_or_else(|| json!({})),
    });
    if let Some(id) = step.get("call_id").or_else(|| step.get("id")) {
        response["id"] = Value::String(string_value(id));
    }
    contents.push(json!({"role":"user","parts":[{"functionResponse":response}]}));
}

fn content_part(content: &Value, thought: bool) -> Option<Value> {
    if let Some(text) = content.get("text") {
        return Some(text_part(&string_value(text), thought));
    }
    if let Some(inline) = content
        .get("inline_data")
        .or_else(|| content.get("inlineData"))
    {
        return inline_data_part(inline);
    }
    match content
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "image" | "audio" | "video" | "document" => media_part(content),
        "image_url" => content
            .pointer("/image_url/url")
            .and_then(Value::as_str)
            .and_then(data_url_part),
        "input_audio" => {
            let format = content
                .pointer("/input_audio/format")
                .and_then(Value::as_str)
                .unwrap_or("");
            let data = content
                .pointer("/input_audio/data")
                .and_then(Value::as_str)
                .unwrap_or("");
            inline_data(&audio_mime_type(format), data)
        }
        "file" => {
            let filename = content
                .pointer("/file/filename")
                .and_then(Value::as_str)
                .unwrap_or("");
            let data = content
                .pointer("/file/file_data")
                .and_then(Value::as_str)
                .unwrap_or("");
            normalize_openai_file_data(filename, "", data)
                .and_then(|(mime, data)| inline_data(&mime, &data))
        }
        _ => None,
    }
}

fn media_part(content: &Value) -> Option<Value> {
    let mime = content
        .get("mime_type")
        .or_else(|| content.get("mimeType"))
        .map(string_value)
        .unwrap_or_default();
    if let Some(data) = content.get("data").and_then(Value::as_str) {
        return inline_data(&mime, data);
    }
    if let Some(uri) = content
        .get("file_uri")
        .or_else(|| content.get("fileUri"))
        .and_then(Value::as_str)
    {
        return file_data(&mime, uri);
    }
    content
        .get("url")
        .and_then(Value::as_str)
        .and_then(data_url_part)
}

fn native_part(part: &Value) -> Option<Value> {
    if part.get("text").is_some()
        || part.get("functionCall").is_some()
        || part.get("functionResponse").is_some()
    {
        return Some(part.clone());
    }
    if let Some(inline) = part.get("inlineData").or_else(|| part.get("inline_data")) {
        return inline_data_part(inline);
    }
    part.get("fileData")
        .or_else(|| part.get("file_data"))
        .and_then(file_data_part)
}

fn text_part(text: &str, thought: bool) -> Value {
    if thought {
        json!({"text":text,"thought":true})
    } else {
        json!({"text":text})
    }
}

fn inline_data_part(inline: &Value) -> Option<Value> {
    let mime = inline
        .get("mimeType")
        .or_else(|| inline.get("mime_type"))
        .map(string_value)
        .unwrap_or_default();
    let data = inline.get("data").map(string_value).unwrap_or_default();
    inline_data(&mime, &data)
}

fn inline_data(mime: &str, data: &str) -> Option<Value> {
    (!mime.is_empty() && !data.is_empty())
        .then(|| json!({"inlineData":{"mimeType":mime,"data":data}}))
}

fn file_data_part(file: &Value) -> Option<Value> {
    let mime = file
        .get("mimeType")
        .or_else(|| file.get("mime_type"))
        .map(string_value)
        .unwrap_or_default();
    let uri = file
        .get("fileUri")
        .or_else(|| file.get("file_uri"))
        .map(string_value)
        .unwrap_or_default();
    file_data(&mime, &uri)
}

fn file_data(mime: &str, uri: &str) -> Option<Value> {
    (!mime.is_empty() && !uri.is_empty())
        .then(|| json!({"fileData":{"mimeType":mime,"fileUri":uri}}))
}

fn data_url_part(url: &str) -> Option<Value> {
    let payload = url.strip_prefix("data:")?;
    let (mime, encoded) = payload.split_once(';')?;
    let data = encoded.strip_prefix("base64,")?;
    inline_data(mime, data)
}

fn push_text(contents: &mut Vec<Value>, role: &str, text: &str, thought: bool) {
    contents.push(json!({"role":content_role(role,"user"),"parts":[text_part(text,thought)]}));
}

fn content_role<'a>(role: &'a str, default: &'a str) -> &'a str {
    match role.trim().to_ascii_lowercase().as_str() {
        "model" | "assistant" => "model",
        "user" => "user",
        _ if default == "model" => "model",
        _ => "user",
    }
}

fn audio_mime_type(format: &str) -> String {
    match format.trim().to_ascii_lowercase().as_str() {
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "opus" => "audio/opus",
        "pcm16" => "audio/pcm",
        _ => "audio/mpeg",
    }
    .to_owned()
}

fn thinking_summary(value: &Value) -> Option<bool> {
    match value.as_str()?.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(true),
        "none" => Some(false),
        _ => None,
    }
}

fn camel_case_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (camel_case(key), camel_case_value(value)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(camel_case_value).collect()),
        _ => value.clone(),
    }
}

fn camel_case(value: &str) -> String {
    let mut parts = value.split('_');
    let mut output = parts.next().unwrap_or("").to_owned();
    for part in parts.filter(|part| !part.is_empty()) {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            output.extend(first.to_uppercase());
            output.extend(chars);
        }
    }
    output
}

fn copy_tools(root: &Value, request: &mut Map<String, Value>, names: &HashMap<String, String>) {
    let Some(tools) = root.get("tools") else {
        return;
    };
    let Some(tools) = tools.as_array() else {
        request.insert("tools".into(), tools.clone());
        return;
    };
    let mut declarations = Vec::new();
    let mut other = Vec::new();
    let mut seen = HashSet::new();
    for tool in tools {
        let declaration_nodes = tool
            .get("functionDeclarations")
            .or_else(|| tool.get("function_declarations"))
            .and_then(Value::as_array);
        if let Some(nodes) = declaration_nodes {
            for node in nodes {
                if let Some(declaration) = function_declaration(node, names) {
                    let name = declaration
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if name.is_empty() || seen.insert(name.to_owned()) {
                        declarations.push(declaration);
                    }
                }
            }
        } else if tool.get("type").and_then(Value::as_str) == Some("function")
            || tool.get("name").is_some()
        {
            if let Some(declaration) = function_declaration(tool, names) {
                let name = declaration
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if name.is_empty() || seen.insert(name.to_owned()) {
                    declarations.push(declaration);
                }
            }
        } else {
            other.push(tool.clone());
        }
    }
    let mut result = Vec::new();
    if !declarations.is_empty() {
        result.push(json!({"functionDeclarations":declarations}));
    }
    result.extend(other);
    if !result.is_empty() {
        request.insert("tools".into(), Value::Array(result));
    }
}

fn function_declaration(declaration: &Value, names: &HashMap<String, String>) -> Option<Value> {
    let function = declaration
        .get("function")
        .filter(|value| value.is_object())
        .unwrap_or(declaration);
    let original = function.get("name").map(string_value).unwrap_or_default();
    if original.trim().is_empty() {
        return None;
    }
    let mut output = json!({
        "name":map_sanitized_function_name(names,&original),
        "parametersJsonSchema":{"type":"object","properties":{}},
    });
    if let Some(description) = function.get("description") {
        output["description"] = Value::String(string_value(description));
    }
    if let Some(parameters) = function
        .get("parametersJsonSchema")
        .or_else(|| function.get("parameters"))
    {
        output["parametersJsonSchema"] = parameters.clone();
    }
    for key in ["response", "responseJsonSchema"] {
        if let Some(value) = function.get(key) {
            output[key] = value.clone();
        }
    }
    Some(output)
}

fn attach_safety_settings(request: &mut Map<String, Value>) {
    if request.contains_key("safetySettings") {
        return;
    }
    request.insert(
        "safetySettings".into(),
        json!([
            {"category":"HARM_CATEGORY_HARASSMENT","threshold":"OFF"},
            {"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"OFF"},
            {"category":"HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold":"OFF"},
            {"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"OFF"},
            {"category":"HARM_CATEGORY_CIVIC_INTEGRITY","threshold":"BLOCK_NONE"},
        ]),
    );
}

pub(crate) fn rewrite_interactions_function_names(
    input: Vec<u8>,
    names: &HashMap<String, String>,
) -> Vec<u8> {
    let Ok(mut root) = serde_json::from_slice::<Value>(&input) else {
        return input;
    };
    let mut changed = false;
    for part in root
        .pointer_mut("/request/contents")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|content| content.get_mut("parts"))
        .filter_map(Value::as_array_mut)
        .flatten()
    {
        for field in ["functionCall", "functionResponse"] {
            let Some(call) = part.get_mut(field).and_then(Value::as_object_mut) else {
                continue;
            };
            let Some(name) = call.get("name") else {
                continue;
            };
            let original = string_value(name);
            let mapped = map_sanitized_function_name(names, &original);
            if !name.is_string() || mapped != original {
                call.insert("name".into(), Value::String(mapped));
                changed = true;
            }
        }
    }
    if let Some(allowed) = root
        .pointer_mut("/request/toolConfig/functionCallingConfig/allowedFunctionNames")
        .and_then(Value::as_array_mut)
    {
        for name in allowed {
            let original = string_value(name);
            let mapped = map_sanitized_function_name(names, &original);
            if !name.is_string() || mapped != original {
                *name = Value::String(mapped);
                changed = true;
            }
        }
    }
    if changed {
        serde_json::to_vec(&root).unwrap_or(input)
    } else {
        input
    }
}

fn string_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        _ => value.to_string(),
    }
}
