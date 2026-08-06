// ref: internal/translator/gemini/openai/responses/gemini_openai-responses_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{HashMap, HashSet};

use serde_json::{json, Map, Value};

use super::signature_carrier::{
    normalize_carriers, CarrierDirection, CarrierTarget, DIRECTION_FIELD, SIGNATURE_FIELD,
    SUMMARY_FIELD, TARGET_FIELD,
};
use crate::internal::signature::compatible_gemini_signature;
use crate::internal::util::gemini_schema::clean_json_schema_for_gemini;

const GEMINI_THOUGHT_SIGNATURE_BYPASS: &str = "skip_thought_signature_validator";
const SAFETY_CATEGORIES: &[(&str, &str)] = &[
    ("HARM_CATEGORY_HARASSMENT", "OFF"),
    ("HARM_CATEGORY_HATE_SPEECH", "OFF"),
    ("HARM_CATEGORY_SEXUALLY_EXPLICIT", "OFF"),
    ("HARM_CATEGORY_DANGEROUS_CONTENT", "OFF"),
    ("HARM_CATEGORY_CIVIC_INTEGRITY", "BLOCK_NONE"),
];

/// Converts the bounded, already differential-gated OpenAI Responses request
/// surface into native Gemini `generateContent` JSON.
///
/// Covers ordinary/system messages, role splitting, images/audio, function
/// calls/results, carrier-aware signed reasoning, tools, generation controls,
/// structured output, default safety and trailing assistant-prefill removal.
pub fn convert_openai_responses_request_to_gemini(
    _model_name: &str,
    input_raw_json: &[u8],
    _stream: bool,
) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(input_raw_json).unwrap_or(Value::Null);
    let mut output = Map::new();
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();

    if let Some(instructions) = root.get("instructions") {
        system_parts.push(json!({"text": value_string(instructions)}));
    }

    match root.get("input") {
        Some(Value::String(text)) => {
            contents.push(content("user", vec![json!({"text": text})]));
        }
        Some(Value::Array(items)) => {
            let (normalized, _) = normalize_carriers(items);
            let bound = bind_leading_carriers(&normalized);
            contents = convert_input_items(&bound, &mut system_parts);
        }
        _ => {}
    }

    coalesce_adjacent_model_contents(&mut contents);
    sanitize_model_function_signatures(&mut contents);
    strip_trailing_model_prefill(&mut contents);
    output.insert("contents".to_owned(), Value::Array(contents));

    if !system_parts.is_empty() {
        output.insert(
            "systemInstruction".to_owned(),
            json!({"parts": system_parts}),
        );
    }
    if let Some(tools) = convert_tools(&root) {
        output.insert("tools".to_owned(), tools);
    }
    if let Some(generation) = generation_config(&root) {
        output.insert("generationConfig".to_owned(), generation);
    }
    output.insert(
        "safetySettings".to_owned(),
        Value::Array(
            SAFETY_CATEGORIES
                .iter()
                .map(|(category, threshold)| json!({"category":category,"threshold":threshold}))
                .collect(),
        ),
    );

    serde_json::to_vec(&Value::Object(output)).unwrap_or_default()
}

fn convert_input_items(items: &[Value], system_parts: &mut Vec<Value>) -> Vec<Value> {
    let mut contents = Vec::new();
    let mut function_names = HashMap::new();
    for item in items {
        if item.get("type").and_then(Value::as_str) == Some("function_call") {
            function_names
                .entry(value_string_at(item, "call_id"))
                .or_insert_with(|| value_string_at(item, "name"));
        }
    }

    let mut pending_call_ids = Vec::new();
    let mut consumed_outputs = HashSet::new();
    let mut index = 0;
    while index < items.len() {
        if consumed_outputs.contains(&index) {
            index += 1;
            continue;
        }
        let item = &items[index];
        let kind = item_kind(item);
        let role = value_string_at(item, "role");
        match kind.as_str() {
            "message" => {
                if role.eq_ignore_ascii_case("system") || role.eq_ignore_ascii_case("developer") {
                    pending_call_ids.clear();
                    append_system_content(item.get("content"), system_parts);
                } else {
                    if assistant_visible_text(item).is_none() {
                        pending_call_ids.clear();
                    }
                    append_message_contents(item, &role, &mut contents);
                }
            }
            "function_call" => {
                contents.push(function_call_content(item));
                let call_id = value_string_at(item, "call_id");
                if !call_id.trim().is_empty() {
                    pending_call_ids.push(call_id);
                }
            }
            "function_call_output" => {
                let end = items[index..]
                    .iter()
                    .take_while(|candidate| item_kind(candidate) == "function_call_output")
                    .count()
                    + index;
                let ordered = order_function_outputs(&items[index..end], &mut pending_call_ids);
                consumed_outputs.extend(index..end);
                let response_parts = ordered
                    .into_iter()
                    .map(|output| function_response_part(output, &function_names))
                    .collect::<Vec<_>>();
                if !response_parts.is_empty() {
                    contents.push(content("user", response_parts));
                }
            }
            "reasoning" => {
                let thought = item
                    .pointer("/summary/0/text")
                    .map(value_string)
                    .unwrap_or_default();
                let signature = item
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .and_then(compatible_gemini_signature);
                if !thought.is_empty() {
                    let mut part = json!({"text":thought,"thought":true});
                    if let Some(signature) = signature {
                        part.as_object_mut()
                            .unwrap()
                            .insert("thoughtSignature".to_owned(), Value::String(signature));
                    }
                    contents.push(content("model", vec![part]));
                } else if let Some(signature) = signature {
                    contents.push(content(
                        "model",
                        vec![json!({"text":"","thoughtSignature":signature})],
                    ));
                }
            }
            _ => {}
        }
        index += 1;
    }
    contents
}

fn item_kind(item: &Value) -> String {
    item.get("type")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| item.get("role").is_some().then(|| "message".to_owned()))
        .unwrap_or_default()
}

fn bind_leading_carriers(items: &[Value]) -> Vec<Value> {
    let mut bound = Vec::with_capacity(items.len());
    let mut index = 0;
    while index < items.len() {
        let item = &items[index];
        let signature = (item_kind(item) == "reasoning")
            .then(|| item.get("encrypted_content").and_then(Value::as_str))
            .flatten()
            .and_then(compatible_gemini_signature);
        let explicit_direction = item
            .get(DIRECTION_FIELD)
            .and_then(Value::as_str)
            .map(str::to_owned);
        let target = item.get(TARGET_FIELD).and_then(Value::as_str);
        if let Some(signature) = signature.as_ref() {
            let can_bind_previous = explicit_direction.as_deref()
                == Some(CarrierDirection::Previous.as_str())
                || explicit_direction.is_none();
            if can_bind_previous {
                let previous_target = bound.last().and_then(semantic_item_target);
                let target_matches = previous_target.is_some_and(|actual| {
                    target.is_none()
                        || target == Some(CarrierTarget::Any.as_str())
                        || target == Some(actual.as_str())
                });
                let output_matches = previous_target != Some(CarrierTarget::Function)
                    || bound.last().is_some_and(|previous| {
                        matching_output_follows(
                            items,
                            index + 1,
                            &value_string_at(previous, "call_id"),
                        )
                    });
                if target_matches && output_matches {
                    if let Some(previous) = bound.last_mut() {
                        attach_carrier_signature(previous, signature, item);
                        index += 1;
                        continue;
                    }
                }
            }
        }
        let can_bind_next = explicit_direction.as_deref()
            != Some(CarrierDirection::Previous.as_str())
            && explicit_direction.as_deref() != Some(CarrierDirection::Standalone.as_str());
        if let (Some(signature), true, Some(next)) =
            (signature, can_bind_next, items.get(index + 1))
        {
            let next_target = match item_kind(next).as_str() {
                "function_call" => Some(CarrierTarget::Function),
                "message" if assistant_visible_text(next).is_some() => Some(CarrierTarget::Text),
                _ => None,
            };
            let target_matches = next_target.is_some_and(|actual| {
                target.is_none()
                    || target == Some(CarrierTarget::Any.as_str())
                    || target == Some(actual.as_str())
            });
            if target_matches {
                let mut attached = next.clone();
                attach_carrier_signature(&mut attached, &signature, item);
                bound.push(attached);
                index += 2;
                continue;
            }
        }
        bound.push(item.clone());
        index += 1;
    }
    bound
}

fn semantic_item_target(item: &Value) -> Option<CarrierTarget> {
    match item_kind(item).as_str() {
        "function_call" => Some(CarrierTarget::Function),
        "message" if assistant_visible_text(item).is_some() => Some(CarrierTarget::Text),
        _ => None,
    }
}

fn attach_carrier_signature(target: &mut Value, signature: &str, carrier: &Value) {
    if let Some(object) = target.as_object_mut() {
        object.insert(
            SIGNATURE_FIELD.to_owned(),
            Value::String(signature.to_owned()),
        );
        if let Some(summary) = carrier.pointer("/summary/0/text").and_then(Value::as_str) {
            if !summary.is_empty() {
                object.insert(SUMMARY_FIELD.to_owned(), Value::String(summary.to_owned()));
            }
        }
    }
}

fn matching_output_follows(items: &[Value], start: usize, call_id: &str) -> bool {
    if call_id.trim().is_empty() {
        return false;
    }
    for item in items.iter().skip(start) {
        match item_kind(item).as_str() {
            "reasoning" | "function_call" => continue,
            "function_call_output" => {
                if value_string_at(item, "call_id") == call_id {
                    return true;
                }
            }
            _ => return false,
        }
    }
    false
}

fn append_system_content(content: Option<&Value>, system_parts: &mut Vec<Value>) {
    match content {
        Some(Value::Array(parts)) => {
            for part in parts {
                system_parts.push(json!({"text": value_string_at(part, "text")}));
            }
        }
        Some(Value::String(text)) => system_parts.push(json!({"text": text})),
        _ => {}
    }
}

fn append_message_contents(item: &Value, item_role: &str, contents: &mut Vec<Value>) {
    match item.get("content") {
        Some(Value::Array(parts)) => {
            let mut current_role = String::new();
            let mut current_parts = Vec::new();
            for part in parts {
                let content_type = part
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("input_text");
                let role = effective_role(item_role, content_type);
                if !current_role.is_empty() && current_role != role {
                    contents.push(content(&current_role, std::mem::take(&mut current_parts)));
                }
                current_role = role;
                if let Some(mut converted) = convert_message_part(part, content_type) {
                    if content_type == "output_text" {
                        if let Some(signature) = item
                            .get(SIGNATURE_FIELD)
                            .and_then(Value::as_str)
                            .and_then(compatible_gemini_signature)
                        {
                            converted
                                .as_object_mut()
                                .unwrap()
                                .insert("thoughtSignature".to_owned(), Value::String(signature));
                        }
                    }
                    current_parts.push(converted);
                }
            }
            if !current_role.is_empty() && !current_parts.is_empty() {
                contents.push(content(&current_role, current_parts));
            }
        }
        Some(Value::String(text)) => {
            contents.push(content(
                &effective_role(item_role, "input_text"),
                vec![json!({"text":text})],
            ));
        }
        _ => {}
    }
}

fn effective_role(item_role: &str, content_type: &str) -> String {
    if content_type == "output_text" {
        return "model".to_owned();
    }
    match item_role.to_ascii_lowercase().as_str() {
        "assistant" | "model" => "model".to_owned(),
        "" => "user".to_owned(),
        other => other.to_owned(),
    }
}

fn convert_message_part(part: &Value, kind: &str) -> Option<Value> {
    match kind {
        "input_text" | "output_text" => part
            .get("text")
            .map(|text| json!({"text": value_string(text)})),
        "input_image" => {
            let source = part
                .get("image_url")
                .or_else(|| part.get("url"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let (mime, data) = data_url(source)?;
            (!data.is_empty()).then(|| json!({"inline_data":{"mime_type":mime,"data":data}}))
        }
        "input_audio" => {
            let data = part.get("data").and_then(Value::as_str).unwrap_or_default();
            if data.is_empty() {
                return None;
            }
            let format = part
                .get("format")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let mime = match format {
                "" | "wav" => "audio/wav".to_owned(),
                "mp3" => "audio/mpeg".to_owned(),
                "ogg" => "audio/ogg".to_owned(),
                "flac" => "audio/flac".to_owned(),
                "aac" => "audio/aac".to_owned(),
                "webm" => "audio/webm".to_owned(),
                "pcm16" => "audio/pcm".to_owned(),
                "g711_ulaw" | "g711_alaw" => "audio/basic".to_owned(),
                other => format!("audio/{other}"),
            };
            Some(json!({"inline_data":{"mime_type":mime,"data":data}}))
        }
        _ => None,
    }
}

fn content(role: &str, parts: Vec<Value>) -> Value {
    json!({"role":role,"parts":parts})
}

fn function_call_content(item: &Value) -> Value {
    let arguments = item
        .get("arguments")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .unwrap_or_else(|| json!({}));
    let signature = item
        .get(SIGNATURE_FIELD)
        .and_then(Value::as_str)
        .and_then(compatible_gemini_signature)
        .unwrap_or_else(|| GEMINI_THOUGHT_SIGNATURE_BYPASS.to_owned());
    let mut parts = Vec::with_capacity(2);
    if let Some(summary) = item.get(SUMMARY_FIELD).and_then(Value::as_str) {
        if !summary.is_empty() {
            parts.push(json!({"text":summary,"thought":true}));
        }
    }
    parts.push(json!({
        "functionCall": {
            "name": sanitize_function_name(&value_string_at(item, "name")),
            "args": arguments,
            "id": value_string_at(item, "call_id")
        },
        "thoughtSignature": signature
    }));
    content("model", parts)
}

fn function_response_part(item: &Value, names: &HashMap<String, String>) -> Value {
    let call_id = value_string_at(item, "call_id");
    let name = names.get(&call_id).map(String::as_str).unwrap_or("unknown");
    let mut response = Map::new();
    if let Some(raw) = item.get("output").and_then(Value::as_str) {
        if !raw.is_empty() && raw != "null" {
            response.insert(
                "result".to_owned(),
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned())),
            );
        }
    }
    json!({"functionResponse":{
        "name":sanitize_function_name(name),
        "response":Value::Object(response),
        "id":call_id
    }})
}

fn order_function_outputs<'a>(outputs: &'a [Value], pending: &mut Vec<String>) -> Vec<&'a Value> {
    let mut ordered = Vec::with_capacity(outputs.len());
    let mut used = vec![false; outputs.len()];
    let mut remaining = Vec::new();
    for pending_id in pending.iter() {
        if let Some((index, output)) = outputs.iter().enumerate().find(|(index, output)| {
            !used[*index] && value_string_at(output, "call_id") == *pending_id
        }) {
            used[index] = true;
            ordered.push(output);
        } else {
            remaining.push(pending_id.clone());
        }
    }
    for (index, output) in outputs.iter().enumerate() {
        if !used[index] {
            ordered.push(output);
        }
    }
    *pending = remaining;
    ordered
}

fn assistant_visible_text(item: &Value) -> Option<String> {
    if item_kind(item) != "message" {
        return None;
    }
    match item.get("content") {
        Some(Value::String(text))
            if matches!(
                value_string_at(item, "role")
                    .trim()
                    .to_ascii_lowercase()
                    .as_str(),
                "assistant" | "model"
            ) =>
        {
            Some(text.clone())
        }
        Some(Value::Array(parts)) => {
            let output = parts
                .iter()
                .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
                .map(|part| value_string_at(part, "text"))
                .collect::<Vec<_>>();
            (!output.is_empty()).then(|| output.join("\n"))
        }
        _ => None,
    }
}

fn coalesce_adjacent_model_contents(contents: &mut Vec<Value>) {
    let mut coalesced: Vec<Value> = Vec::with_capacity(contents.len());
    for mut item in contents.drain(..) {
        let is_model = item.get("role").and_then(Value::as_str) == Some("model");
        if is_model
            && coalesced
                .last()
                .and_then(|last| last.get("role"))
                .and_then(Value::as_str)
                == Some("model")
        {
            if let (Some(target), Some(source)) = (
                coalesced
                    .last_mut()
                    .and_then(|last| last.get_mut("parts"))
                    .and_then(Value::as_array_mut),
                item.get_mut("parts").and_then(Value::as_array_mut),
            ) {
                target.append(source);
                continue;
            }
        }
        coalesced.push(item);
    }
    *contents = coalesced;
}

fn sanitize_model_function_signatures(contents: &mut [Value]) {
    for content in contents {
        if content.get("role").and_then(Value::as_str) != Some("model") {
            continue;
        }
        let Some(parts) = content.get_mut("parts").and_then(Value::as_array_mut) else {
            continue;
        };
        let mut first_call = true;
        for part in parts {
            if part.get("functionCall").is_none() {
                if part.get("thoughtSignature").and_then(Value::as_str)
                    == Some(GEMINI_THOUGHT_SIGNATURE_BYPASS)
                {
                    if let Some(object) = part.as_object_mut() {
                        object.remove("thoughtSignature");
                    }
                }
                continue;
            }
            if first_call {
                let compatible = part
                    .get("thoughtSignature")
                    .and_then(Value::as_str)
                    .and_then(compatible_gemini_signature);
                if let Some(object) = part.as_object_mut() {
                    object.insert(
                        "thoughtSignature".to_owned(),
                        Value::String(
                            compatible
                                .unwrap_or_else(|| GEMINI_THOUGHT_SIGNATURE_BYPASS.to_owned()),
                        ),
                    );
                }
                first_call = false;
            } else {
                let compatible = part
                    .get("thoughtSignature")
                    .and_then(Value::as_str)
                    // The validator bypass is synthetic and belongs only to
                    // the first parallel call. The generic compatibility
                    // validator accepts it, so exclude it before validating
                    // real provider signatures for unsigned siblings.
                    .filter(|signature| *signature != GEMINI_THOUGHT_SIGNATURE_BYPASS)
                    .and_then(compatible_gemini_signature);
                if let Some(object) = part.as_object_mut() {
                    object.remove("thought_signature");
                    if let Some(compatible) = compatible {
                        object.insert("thoughtSignature".to_owned(), Value::String(compatible));
                    } else {
                        object.remove("thoughtSignature");
                    }
                }
            }
        }
    }
}

fn strip_trailing_model_prefill(contents: &mut Vec<Value>) {
    let should_strip = contents.last().is_some_and(|last| {
        last.get("role").and_then(Value::as_str) == Some("model")
            && last
                .get("parts")
                .and_then(Value::as_array)
                .is_some_and(|parts| {
                    parts.iter().all(|part| {
                        !part
                            .get("thought")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                            && part.get("functionCall").is_none()
                            && part
                                .get("thoughtSignature")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .trim()
                                .is_empty()
                    })
                })
    });
    if should_strip {
        contents.pop();
    }
}

fn convert_tools(root: &Value) -> Option<Value> {
    let declarations = root
        .get("tools")?
        .as_array()?
        .iter()
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
        .map(|tool| {
            json!({
                "name": sanitize_function_name(&value_string_at(tool, "name")),
                "description": value_string_at(tool, "description"),
                "parametersJsonSchema": tool
                    .get("parameters")
                    .map(clean_gemini_tool_schema_with_root_placeholder_parity)
                    .unwrap_or_else(|| json!({}))
            })
        })
        .collect::<Vec<_>>();
    (!declarations.is_empty()).then(|| json!([{"functionDeclarations":declarations}]))
}

/// The pinned Go cleaner's path matcher intentionally/observably skips
/// placeholder properties at the schema root while still removing them in
/// nested object schemas. The shared Rust cleaner is stricter, so restore only
/// those root properties here to keep this wire adapter byte-semantically
/// aligned with upstream without weakening the cleaner for other callers.
fn clean_gemini_tool_schema_with_root_placeholder_parity(schema: &Value) -> Value {
    let mut cleaned = clean_json_schema_for_gemini(schema);
    let Some(original_properties) = schema.get("properties").and_then(Value::as_object) else {
        return cleaned;
    };
    let Some(cleaned_properties) = cleaned.get_mut("properties").and_then(Value::as_object_mut)
    else {
        return cleaned;
    };
    for name in ["_", "reason"] {
        if let Some(property) = original_properties.get(name) {
            cleaned_properties.insert(name.to_owned(), clean_json_schema_for_gemini(property));
        }
    }
    let restored = ["_", "reason"]
        .into_iter()
        .filter(|name| cleaned_properties.contains_key(*name))
        .collect::<std::collections::HashSet<_>>();
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        let mut retained = cleaned
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<std::collections::HashSet<_>>();
        retained.extend(restored);
        let required = required
            .iter()
            .filter(|name| name.as_str().is_some_and(|name| retained.contains(name)))
            .cloned()
            .collect::<Vec<_>>();
        if !required.is_empty() {
            cleaned["required"] = Value::Array(required);
        }
    }
    cleaned
}

fn generation_config(root: &Value) -> Option<Value> {
    let mut generation = Map::new();
    if let Some(value) = root.get("max_output_tokens") {
        generation.insert(
            "maxOutputTokens".to_owned(),
            Value::from(value.as_i64().unwrap_or_default()),
        );
    }
    if let Some(value) = root.get("temperature") {
        generation.insert(
            "temperature".to_owned(),
            Value::from(value.as_f64().unwrap_or_default()),
        );
    }
    if let Some(value) = root.get("top_p") {
        generation.insert(
            "topP".to_owned(),
            Value::from(value.as_f64().unwrap_or_default()),
        );
    }
    if let Some(sequences) = root.get("stop_sequences").and_then(Value::as_array) {
        generation.insert(
            "stopSequences".to_owned(),
            Value::Array(
                sequences
                    .iter()
                    .map(|value| Value::String(value_string(value)))
                    .collect(),
            ),
        );
    }
    if let Some(format) = root.pointer("/text/format") {
        match format
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "json_object" => {
                generation.insert(
                    "responseMimeType".to_owned(),
                    Value::String("application/json".to_owned()),
                );
            }
            "json_schema" => {
                generation.insert(
                    "responseMimeType".to_owned(),
                    Value::String("application/json".to_owned()),
                );
                if let Some(schema) = format
                    .get("schema")
                    .or_else(|| format.pointer("/json_schema/schema"))
                {
                    generation.insert("responseJsonSchema".to_owned(), schema.clone());
                }
            }
            _ => {}
        }
    }
    if let Some(effort) = root
        .pointer("/reasoning/effort")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|effort| !effort.is_empty())
    {
        let thinking = if effort.eq_ignore_ascii_case("auto") {
            json!({"thinkingBudget":-1})
        } else {
            json!({"thinkingLevel":effort.to_ascii_lowercase()})
        };
        generation.insert("thinkingConfig".to_owned(), thinking);
    }
    (!generation.is_empty()).then_some(Value::Object(generation))
}

fn sanitize_function_name(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let mut output: String = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || "_.:-".contains(character) {
                character
            } else {
                '_'
            }
        })
        .collect();
    if !output
        .as_bytes()
        .first()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
    {
        if output.len() >= 64 {
            output.truncate(63);
        }
        output.insert(0, '_');
    }
    output.truncate(output.len().min(64));
    output
}

fn data_url(raw: &str) -> Option<(&str, &str)> {
    let value = raw.strip_prefix("data:")?;
    value
        .split_once(";base64,")
        .or_else(|| value.split_once(','))
}

fn value_string_at(value: &Value, key: &str) -> String {
    value.get(key).map(value_string).unwrap_or_default()
}

fn value_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Null => String::new(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convert(input: &[u8]) -> Value {
        serde_json::from_slice(&convert_openai_responses_request_to_gemini(
            "gemini-3.6-flash-high",
            input,
            false,
        ))
        .unwrap()
    }

    #[test]
    fn converts_messages_media_controls_and_safety() {
        let output = convert(
            br#"{
            "instructions":"Be precise.",
            "input":[{"role":"user","content":[
                {"type":"input_text","text":"look"},
                {"type":"input_image","image_url":"data:image/png;base64,aW1n"},
                {"type":"input_audio","format":"mp3","data":"YXVkaW8="}
            ]}],
            "max_output_tokens":42,"temperature":0.5,"top_p":0.9,
            "stop_sequences":["END"],"reasoning":{"effort":"auto"}
        }"#,
        );
        assert_eq!(output["contents"][0]["parts"][0]["text"], "look");
        assert_eq!(
            output["contents"][0]["parts"][1]["inline_data"]["mime_type"],
            "image/png"
        );
        assert_eq!(
            output["contents"][0]["parts"][2]["inline_data"]["mime_type"],
            "audio/mpeg"
        );
        assert_eq!(
            output["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            -1
        );
        assert_eq!(output["safetySettings"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn orders_parallel_outputs_and_signs_only_first_call() {
        let output = convert(
            br#"{"input":[
          {"type":"function_call","call_id":"a","name":"1 bad name","arguments":"{\"x\":1}"},
          {"type":"function_call","call_id":"b","name":"second","arguments":"{}"},
          {"type":"function_call_output","call_id":"b","output":"plain"},
          {"type":"function_call_output","call_id":"a","output":"{\"ok\":true}"}
        ]}"#,
        );
        let model_parts = output["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(model_parts[0]["functionCall"]["name"], "_1_bad_name");
        assert_eq!(
            model_parts[0]["thoughtSignature"],
            GEMINI_THOUGHT_SIGNATURE_BYPASS
        );
        assert!(model_parts[1].get("thoughtSignature").is_none());
        let outputs = output["contents"][1]["parts"].as_array().unwrap();
        assert_eq!(outputs[0]["functionResponse"]["id"], "a");
        assert_eq!(outputs[1]["functionResponse"]["id"], "b");
    }

    #[test]
    fn removes_only_unsigned_trailing_model_prefill() {
        let output = convert(
            br#"{"input":[
          {"role":"assistant","content":"prefill"}
        ]}"#,
        );
        assert_eq!(output["contents"], json!([]));

        let output = convert(
            br#"{"input":[
          {"type":"function_call","call_id":"a","name":"run","arguments":"{}"}
        ]}"#,
        );
        assert_eq!(output["contents"].as_array().unwrap().len(), 1);
    }
}
