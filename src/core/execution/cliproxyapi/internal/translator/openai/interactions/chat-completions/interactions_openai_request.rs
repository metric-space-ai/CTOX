// ref: internal/translator/openai/interactions/chat-completions/interactions_openai_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

//! Interactions request body -> OpenAI chat-completions request body.
//!
//! Upstream uses `gjson`/`sjson` to splice raw byte sub-trees; this port
//! preserves the same field-omission and order semantics through a typed
//! `serde_json::Value` builder and the `internal::translator::common` helpers.

use std::fmt::Write as _;

use serde_json::{json, Map, Value};

use crate::internal::translator::common::{
    join_raw_array, new_raw_array_items, normalize_openai_file_data, set_raw_array_items,
};

pub fn convert_interactions_request_to_openai(
    model_name: &str,
    input_raw_json: &[u8],
    stream: bool,
) -> Vec<u8> {
    let root = parse_root(input_raw_json);
    let mut out = Map::new();
    out.insert(
        "model".into(),
        Value::String(first_nonempty(&[
            model_name,
            root.get("model").and_then(Value::as_str).unwrap_or(""),
        ])),
    );
    out.insert("messages".into(), Value::Array(Vec::new()));
    if stream || root.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        out.insert("stream".into(), Value::Bool(true));
    }
    let system_text = interactions_text(root.get("system_instruction"));
    let input_len = root
        .get("input")
        .and_then(Value::as_array)
        .map(|value| value.len() as i64)
        .unwrap_or_default();
    let capacity = input_len + if system_text.is_empty() { 0 } else { 1 };
    let mut message_items = new_raw_array_items(capacity).unwrap_or_default();
    append_interactions_system_to_openai(&mut message_items, &root);
    append_interactions_input_to_openai_messages(&mut message_items, &root);
    out = apply_raw_messages(out, message_items);
    out = copy_interactions_tools_to_openai(out, &root);
    out = copy_interactions_generation_config_to_openai(out, &root);
    out = copy_interactions_openai_top_level(out, &root);
    serde_json::to_vec(&Value::Object(out)).unwrap_or_default()
}

fn apply_raw_messages(mut out: Map<String, Value>, items: Vec<Vec<u8>>) -> Map<String, Value> {
    if items.is_empty() {
        return out;
    }
    // Preserve the `{"model":"","messages":[]}` slot shape that upstream builds
    // byte-for-byte; `set_raw_array_items` then splices each item verbatim so
    // surrounding field order and raw tool-argument bytes survive the round
    // trip.
    out.insert(
        "messages".into(),
        serde_json::from_slice::<Value>(&join_raw_array(&items)).unwrap_or(Value::Null),
    );
    let mut encoded = serde_json::to_vec(&Value::Object(out.clone())).unwrap_or_default();
    encoded = set_raw_array_items(&encoded, "messages", &items);
    serde_json::from_slice::<Value>(&encoded)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or(out)
}

fn parse_root(input: &[u8]) -> Value {
    serde_json::from_slice(input).unwrap_or(Value::Null)
}

fn append_interactions_system_to_openai(items: &mut Vec<Vec<u8>>, root: &Value) {
    let text = interactions_text(root.get("system_instruction"));
    if text.is_empty() {
        return;
    }
    let mut msg = Map::new();
    msg.insert("role".into(), Value::String("system".into()));
    msg.insert("content".into(), Value::String(text));
    items.push(serde_json::to_vec(&Value::Object(msg)).unwrap_or_default());
}

fn append_interactions_input_to_openai_messages(items: &mut Vec<Vec<u8>>, root: &Value) {
    let Some(input) = root.get("input") else {
        return;
    };
    if let Some(text) = input.as_str() {
        let mut msg = Map::new();
        msg.insert("role".into(), Value::String("user".into()));
        msg.insert("content".into(), Value::String(text.to_owned()));
        items.push(serde_json::to_vec(&Value::Object(msg)).unwrap_or_default());
        return;
    }
    if let Some(array) = input.as_array() {
        for step in array {
            append_interactions_step_to_openai(items, step, "user");
        }
        return;
    }
    if input.is_object() {
        append_interactions_step_to_openai(items, input, "user");
    }
}

fn append_interactions_step_to_openai(items: &mut Vec<Vec<u8>>, step: &Value, default_role: &str) {
    match step.get("type").and_then(Value::as_str).unwrap_or_default() {
        "user_input" => append_interactions_message_to_openai(items, step, "user"),
        "model_output" => append_interactions_message_to_openai(items, step, "assistant"),
        "thought" => append_interactions_thought_to_openai(items, step),
        "function_call" => append_interactions_function_call_to_openai(items, step),
        "function_result" => append_interactions_function_result_to_openai(items, step),
        _ => {
            if let Some(text) = step.as_str() {
                let mut msg = Map::new();
                msg.insert("role".into(), Value::String(default_role.into()));
                msg.insert("content".into(), Value::String(text.to_owned()));
                items.push(serde_json::to_vec(&Value::Object(msg)).unwrap_or_default());
            }
        }
    }
}

fn append_interactions_message_to_openai(items: &mut Vec<Vec<u8>>, step: &Value, role: &str) {
    let mut msg = Map::new();
    msg.insert("role".into(), Value::String(role.into()));
    msg.insert("content".into(), Value::String(String::new()));
    if let Some(content) = step.get("content") {
        if let Some(text) = content.as_str() {
            msg.insert("content".into(), Value::String(text.to_owned()));
        } else {
            let encoded = append_interactions_content_to_openai_message(content, role);
            if let Some(object) = encoded.as_object() {
                for (key, value) in object {
                    msg.insert(key.clone(), value.clone());
                }
            }
        }
    }
    items.push(serde_json::to_vec(&Value::Object(msg)).unwrap_or_default());
}

fn append_interactions_thought_to_openai(items: &mut Vec<Vec<u8>>, step: &Value) {
    let mut msg = Map::new();
    msg.insert("role".into(), Value::String("assistant".into()));
    msg.insert("content".into(), Value::String(String::new()));
    msg.insert(
        "reasoning_content".into(),
        Value::String(interactions_text(step.get("content"))),
    );
    items.push(serde_json::to_vec(&Value::Object(msg)).unwrap_or_default());
}

fn append_interactions_content_to_openai_message(content: &Value, role: &str) -> Value {
    if let Some(text) = content.as_str() {
        return json!({"content":text});
    }
    let mut converted_items: Vec<Vec<u8>> = Vec::new();
    let mut text_only = true;
    let mut text_builder = String::new();
    let mut append_part = |part: &Value, items: &mut Vec<Vec<u8>>, text_builder: &mut String| {
        let Some((converted, is_text)) = interactions_content_part_to_openai(part, role) else {
            return;
        };
        if is_text {
            text_builder.push_str(converted.get("text").and_then(Value::as_str).unwrap_or(""));
        } else {
            text_only = false;
        }
        items.push(serde_json::to_vec(&converted).unwrap_or_default());
    };
    if let Some(parts) = content.as_array() {
        for part in parts {
            append_part(part, &mut converted_items, &mut text_builder);
        }
    } else if content.is_object() {
        append_part(content, &mut converted_items, &mut text_builder);
    }
    if converted_items.is_empty() {
        return Value::Null;
    }
    if text_only {
        json!({"content":text_builder})
    } else {
        let array = serde_json::from_slice::<Value>(&join_raw_array(&converted_items))
            .unwrap_or(Value::Array(Vec::new()));
        json!({"content":array})
    }
}

fn interactions_content_part_to_openai(part: &Value, role: &str) -> Option<(Value, bool)> {
    let mut part_type = part.get("type").and_then(Value::as_str).unwrap_or("");
    if part_type.is_empty() && part.get("text").is_some() {
        part_type = "text";
    }
    match part_type {
        "text" => Some((
            json!({"type":"text","text":part.get("text").and_then(Value::as_str).unwrap_or("")}),
            true,
        )),
        "image" => Some((
            json!({
                "type":"image_url",
                "image_url":{"url":interactions_media_data_url(part, "application/octet-stream")}
            }),
            false,
        )),
        "audio" => Some((
            json!({
                "type":"input_audio",
                "input_audio":{
                    "data":part.get("data").and_then(Value::as_str).unwrap_or(""),
                    "format":openai_input_audio_format_from_mime(
                        part.get("mime_type").and_then(Value::as_str).unwrap_or("")
                    )
                }
            }),
            false,
        )),
        "video" => Some((
            json!({
                "type":"video_url",
                "video_url":{"url":interactions_media_data_url(part, "video/mp4")}
            }),
            false,
        )),
        "document" | "file" => {
            let filename = first_nonempty(&[
                part.get("filename").and_then(Value::as_str).unwrap_or(""),
                &openai_file_name_from_mime(
                    part.get("mime_type").and_then(Value::as_str).unwrap_or(""),
                ),
            ]);
            let mut file = Map::new();
            file.insert("filename".into(), Value::String(filename));
            file.insert(
                "file_data".into(),
                Value::String(
                    part.get("data")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                ),
            );
            let url = first_nonempty(&[
                part.get("file_url").and_then(Value::as_str).unwrap_or(""),
                part.get("url").and_then(Value::as_str).unwrap_or(""),
            ]);
            if !url.is_empty() {
                file.remove("file_data");
                file.insert("file_url".into(), Value::String(url));
            }
            Some((json!({"type":"file","file":Value::Object(file)}), false))
        }
        _ => {
            let _ = role;
            None
        }
    }
}

fn append_interactions_function_call_to_openai(items: &mut Vec<Vec<u8>>, step: &Value) {
    let call_id = first_nonempty(&[
        step.get("call_id").and_then(Value::as_str).unwrap_or(""),
        step.get("id").and_then(Value::as_str).unwrap_or(""),
        "call_0",
    ]);
    let mut tool_call = Map::new();
    tool_call.insert("id".into(), Value::String(call_id));
    tool_call.insert("type".into(), Value::String("function".into()));
    let mut function = Map::new();
    function.insert(
        "name".into(),
        Value::String(
            step.get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
        ),
    );
    function.insert(
        "arguments".into(),
        Value::String(json_string_value(step.get("arguments"), "{}")),
    );
    tool_call.insert("function".into(), Value::Object(function));
    let tool_call_bytes = serde_json::to_vec(&Value::Object(tool_call)).unwrap_or_default();
    let mut msg = Map::new();
    msg.insert("role".into(), Value::String("assistant".into()));
    msg.insert("content".into(), Value::String(String::new()));
    msg.insert(
        "tool_calls".into(),
        serde_json::from_slice::<Value>(&join_raw_array(std::slice::from_ref(&tool_call_bytes)))
            .unwrap_or(Value::Array(Vec::new())),
    );
    let mut msg_bytes = serde_json::to_vec(&Value::Object(msg)).unwrap_or_default();
    msg_bytes = set_raw_array_items(&msg_bytes, "tool_calls", &[tool_call_bytes]);
    items.push(msg_bytes);
}

fn append_interactions_function_result_to_openai(items: &mut Vec<Vec<u8>>, step: &Value) {
    let call_id = first_nonempty(&[
        step.get("call_id").and_then(Value::as_str).unwrap_or(""),
        step.get("id").and_then(Value::as_str).unwrap_or(""),
    ]);
    let result = step.get("result").or_else(|| step.get("output"));
    let mut msg = Map::new();
    msg.insert("role".into(), Value::String("tool".into()));
    msg.insert("tool_call_id".into(), Value::String(call_id));
    msg.insert(
        "content".into(),
        Value::String(json_string_value(result, "")),
    );
    items.push(serde_json::to_vec(&Value::Object(msg)).unwrap_or_default());
}

fn copy_interactions_tools_to_openai(
    mut out: Map<String, Value>,
    root: &Value,
) -> Map<String, Value> {
    let Some(tools) = root.get("tools").and_then(Value::as_array) else {
        return out;
    };
    let mut tool_items: Vec<Vec<u8>> = Vec::new();
    for tool in tools {
        if let Some(converted) = openai_tool_from_interactions_tool(tool) {
            tool_items.push(serde_json::to_vec(&converted).unwrap_or_default());
        }
        let declarations = first_existing([
            tool.get("function_declarations"),
            tool.get("functionDeclarations"),
        ]);
        if let Some(decls) = declarations.and_then(Value::as_array) {
            for decl in decls {
                if let Some(converted) = openai_tool_from_interactions_tool(decl) {
                    tool_items.push(serde_json::to_vec(&converted).unwrap_or_default());
                }
            }
        }
    }
    if !tool_items.is_empty() {
        out.insert(
            "tools".into(),
            serde_json::from_slice::<Value>(&join_raw_array(&tool_items))
                .unwrap_or(Value::Array(Vec::new())),
        );
    }
    out
}

fn copy_interactions_generation_config_to_openai(
    mut out: Map<String, Value>,
    root: &Value,
) -> Map<String, Value> {
    let gen = first_existing([root.get("generation_config"), root.get("generationConfig")])
        .cloned()
        .unwrap_or(Value::Null);
    copy_number(
        &mut out,
        "temperature",
        first_existing([gen.get("temperature"), root.get("temperature")]),
    );
    copy_number(
        &mut out,
        "max_tokens",
        first_existing([
            gen.get("max_output_tokens"),
            gen.get("maxOutputTokens"),
            root.get("max_tokens"),
            root.get("max_completion_tokens"),
        ]),
    );
    copy_number(
        &mut out,
        "top_p",
        first_existing([gen.get("top_p"), gen.get("topP"), root.get("top_p")]),
    );
    copy_number(
        &mut out,
        "top_k",
        first_existing([gen.get("top_k"), gen.get("topK")]),
    );
    copy_number(
        &mut out,
        "n",
        first_existing([
            gen.get("candidate_count"),
            gen.get("candidateCount"),
            root.get("n"),
        ]),
    );
    // Upstream uses `sjson.SetRawBytes` whenever the JSON pointer exists
    // (including explicit `null`), so a literal `null` is preserved rather
    // than dropped on the floor. The presence test in Go gjson is the source
    // of truth here.
    if let Some(stop) = first_existing([
        gen.get("stop_sequences"),
        gen.get("stopSequences"),
        root.get("stop"),
    ]) {
        out.insert("stop".into(), stop.clone());
    }
    if let Some(tool_choice) = first_existing([gen.get("tool_choice"), root.get("tool_choice")]) {
        out.insert("tool_choice".into(), tool_choice.clone());
    }
    if let Some(effort) = interactions_reasoning_effort(root, &gen) {
        out.insert("reasoning_effort".into(), Value::String(effort));
    }
    if let Some(modalities) = root.get("response_modalities") {
        out.insert("modalities".into(), modalities.clone());
    }
    out
}

fn copy_interactions_openai_top_level(
    mut out: Map<String, Value>,
    root: &Value,
) -> Map<String, Value> {
    if let Some(format) = root.get("response_format") {
        out.insert("response_format".into(), format.clone());
    }
    if let Some(service_tier) = root.get("service_tier").and_then(Value::as_str) {
        if !service_tier.is_empty() {
            out.insert(
                "service_tier".into(),
                Value::String(service_tier.to_owned()),
            );
        }
    }
    for key in ["parallel_tool_calls", "seed", "user"] {
        if let Some(value) = root.get(key) {
            out.insert(key.into(), value.clone());
        }
    }
    serde_json::from_slice::<Value>(
        &serde_json::to_vec(&Value::Object(out.clone())).unwrap_or_default(),
    )
    .ok()
    .and_then(|value| value.as_object().cloned())
    .unwrap_or(out)
}

fn openai_tool_from_interactions_tool(tool: &Value) -> Option<Value> {
    let name = first_nonempty(&[
        tool.get("name").and_then(Value::as_str).unwrap_or(""),
        tool.pointer("/function/name")
            .and_then(Value::as_str)
            .unwrap_or(""),
    ]);
    if name.is_empty() {
        return None;
    }
    let mut out = Map::new();
    out.insert("type".into(), Value::String("function".into()));
    let mut function = Map::new();
    function.insert("name".into(), Value::String(name));
    if let Some(desc) = first_existing([
        tool.get("description"),
        tool.pointer("/function/description"),
    ]) {
        // Upstream `sjson.SetBytes(out, "function.description", desc.String())`
        // reinterprets the gjson-rendered string as JSON. For string-typed
        // values that means the field is set as a JSON string; for compound
        // values (rare) the upstream `String()` returns a JSON fragment that
        // `sjson` parses back. Mirror the parse-back by inserting the value
        // unchanged whenever it is not already a JSON string.
        if let Some(text) = desc.as_str() {
            function.insert("description".into(), Value::String(text.to_owned()));
        } else {
            function.insert("description".into(), desc.clone());
        }
    }
    if let Some(params) = first_existing([
        tool.get("parameters"),
        tool.pointer("/function/parameters"),
        tool.get("parametersJsonSchema"),
    ]) {
        function.insert("parameters".into(), params.clone());
    }
    out.insert("function".into(), Value::Object(function));
    Some(Value::Object(out))
}

pub(super) fn interactions_text(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    if let Some(text) = value.as_str() {
        return text.to_owned();
    }
    if let Some(text) = value.get("text").and_then(Value::as_str) {
        return text.to_owned();
    }
    for path in ["content", "parts"] {
        let Some(parts) = value.get(path).and_then(Value::as_array) else {
            continue;
        };
        let mut builder = String::new();
        for part in parts {
            let text = first_nonempty(&[
                part.get("text").and_then(Value::as_str).unwrap_or(""),
                part.pointer("/content/text")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            ]);
            builder.push_str(&text);
        }
        return builder;
    }
    String::new()
}

fn interactions_reasoning_effort(root: &Value, gen: &Value) -> Option<String> {
    let candidates: [Option<&Value>; 6] = [
        gen.get("reasoning_effort"),
        gen.get("thinking_level"),
        gen.get("thinkingLevel"),
        gen.pointer("/thinking_config/thinking_level"),
        gen.pointer("/thinkingConfig/thinkingLevel"),
        root.get("reasoning_effort"),
    ];
    for candidate in candidates.into_iter().flatten() {
        if let Some(text) = candidate.as_str() {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_ascii_lowercase());
            }
        }
    }
    None
}

pub(super) fn interactions_media_data_url(part: &Value, fallback_mime_type: &str) -> String {
    let direct = first_nonempty(&[
        part.get("image_url").and_then(Value::as_str).unwrap_or(""),
        part.get("file_data").and_then(Value::as_str).unwrap_or(""),
        part.get("url").and_then(Value::as_str).unwrap_or(""),
    ]);
    if !direct.is_empty() {
        return direct;
    }
    let data = part.get("data").and_then(Value::as_str).unwrap_or("");
    if data.is_empty() {
        return String::new();
    }
    let mime = first_nonempty(&[
        part.get("mime_type").and_then(Value::as_str).unwrap_or(""),
        fallback_mime_type,
    ]);
    let mut output = String::with_capacity(mime.len() + data.len() + 16);
    let _ = write!(output, "data:{mime};base64,{data}");
    output
}

fn openai_input_audio_format_from_mime(mime_type: &str) -> &'static str {
    match mime_type.trim().to_ascii_lowercase().as_str() {
        "audio/wav" | "audio/wave" | "audio/x-wav" => "wav",
        "audio/flac" => "flac",
        "audio/opus" | "audio/ogg" => "opus",
        "audio/pcm" | "audio/l16" => "pcm16",
        _ => "mp3",
    }
}

fn openai_file_name_from_mime(mime_type: &str) -> String {
    let normalized = mime_type.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "application/pdf" => "document.pdf".to_owned(),
        "text/plain" => "document.txt".to_owned(),
        "text/csv" => "document.csv".to_owned(),
        "application/json" => "document.json".to_owned(),
        other => {
            if let Some((_, suffix)) = other.split_once('/') {
                if !suffix.is_empty() {
                    return format!("document.{}", suffix.replace('+', "."));
                }
            }
            "document.bin".to_owned()
        }
    }
}

fn copy_number(out: &mut Map<String, Value>, path: &str, value: Option<&Value>) {
    if let Some(value) = value {
        out.insert(path.into(), value.clone());
    }
}

pub(super) fn json_string_value(value: Option<&Value>, fallback: &str) -> String {
    match value {
        None => fallback.to_owned(),
        // Matches upstream `gjson.Result.String()` for string-typed values
        // (returns the unquoted content).
        Some(Value::String(text)) => text.clone(),
        // Non-string existing values are encoded via their JSON serialization,
        // matching upstream's `value.Raw` behaviour. The downstream caller
        // (OpenAI chat tool-call arguments) treats the result as a JSON string
        // payload, so the serialized text is what the consumer expects.
        Some(value) => serde_json::to_string(value).unwrap_or_else(|_| fallback.to_owned()),
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

// Touched by the file-data regression test in `openai_interactions_file_data_test`
// so the upstream normalization behavior is exercised end-to-end from this leaf.
#[allow(dead_code)]
pub(super) fn document_file_data(
    filename: &str,
    fallback_mime_type: &str,
    file_data: &str,
) -> Option<(String, String)> {
    normalize_openai_file_data(filename, fallback_mime_type, file_data)
}
