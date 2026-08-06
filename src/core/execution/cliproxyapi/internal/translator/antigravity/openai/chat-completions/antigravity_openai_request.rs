// ref: internal/translator/antigravity/openai/chat-completions/antigravity_openai_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::internal::translator::antigravity::gemini::sanitize_antigravity_claude_gemini_request_signatures;
use crate::internal::translator::gemini::openai::chat_completions::convert_openai_chat_request_to_gemini;

pub fn convert_openai_chat_request_to_antigravity(
    model_name: &str,
    input: &[u8],
    stream: bool,
) -> Vec<u8> {
    let root = serde_json::from_slice::<Value>(input).unwrap_or(Value::Null);
    let mut filtered = root.clone();
    remove_video_parts(&mut filtered);
    let mut request = serde_json::from_slice::<Value>(&convert_openai_chat_request_to_gemini(
        model_name,
        &serde_json::to_vec(&filtered).unwrap_or_default(),
        stream,
    ))
    .unwrap_or_else(|_| json!({"contents":[]}));
    request.as_object_mut().map(|value| value.remove("model"));

    adapt_generation_config(&root, &mut request);
    let names = disambiguated_names(&root);
    adapt_inline_data_and_thinking(&mut request);
    adapt_tools(&root, &mut request, &names);
    adapt_tool_choice(&root, &mut request, &names);
    adapt_history(&root, &mut request, &names);

    let output = normalize_antigravity_openai_thinking_config(
        serde_json::to_vec(&json!({"project":"","request":request,"model":model_name}))
            .unwrap_or_default(),
    );
    sanitize_antigravity_claude_gemini_request_signatures(model_name, output)
}

fn adapt_generation_config(root: &Value, request: &mut Value) {
    if root.get("generationConfig").is_none() {
        if let Some(legacy) = root.get("generation_config").and_then(Value::as_object) {
            let current = request
                .get("generationConfig")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let mut merged = legacy.clone();
            merged.extend(current);
            request["generationConfig"] = Value::Object(merged);
        }
    }
    if root.get("max_tokens").is_none() && root.get("max_completion_tokens").is_some() {
        if let Some(generation) = request
            .get_mut("generationConfig")
            .and_then(Value::as_object_mut)
        {
            generation.remove("maxOutputTokens");
        }
    }
    if let Some(generation) = request
        .get_mut("generationConfig")
        .and_then(Value::as_object_mut)
    {
        if let Some(schema) = generation.remove("responseJsonSchema") {
            generation.remove("responseSchema");
            generation.insert("responseSchema".into(), schema);
        }
        let response_format = root
            .get("response_format")
            .and_then(Value::as_object)
            .and_then(|format| format.get("type"))
            .and_then(Value::as_str)
            .map(str::trim)
            .map(str::to_ascii_lowercase);
        if response_format
            .as_deref()
            .is_some_and(|kind| matches!(kind, "json_object" | "json_schema"))
        {
            for alias in [
                "responseSchema",
                "responseJsonSchema",
                "response_schema",
                "response_json_schema",
            ] {
                generation.remove(alias);
            }
            generation.insert(
                "responseMimeType".into(),
                Value::String("application/json".into()),
            );
            if response_format.as_deref() == Some("json_schema") {
                if let Some(schema) = root.pointer("/response_format/json_schema/schema") {
                    generation.insert("responseSchema".into(), schema.clone());
                }
            }
        }
    }
    adapt_thinking_config(root, request);
}

fn adapt_thinking_config(root: &Value, request: &mut Value) {
    let generation = request
        .as_object_mut()
        .expect("translated Gemini request is an object")
        .entry("generationConfig")
        .or_insert_with(|| json!({}));
    let generation = generation
        .as_object_mut()
        .expect("generationConfig emitted by the Gemini translator is an object");
    let thinking = generation
        .entry("thinkingConfig")
        .or_insert_with(|| json!({}));
    let thinking = thinking
        .as_object_mut()
        .expect("thinkingConfig emitted by the Gemini translator is an object");

    let effort = root
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase);
    if let Some(effort) = effort.as_deref() {
        if effort == "auto" {
            thinking.insert("thinkingBudget".into(), Value::from(-1));
        } else {
            thinking.insert("thinkingLevel".into(), Value::String(effort.into()));
        }
        thinking.insert("includeThoughts".into(), Value::Bool(true));
    }

    let explicit_visibility = root
        .pointer("/generationConfig/thinkingConfig/includeThoughts")
        .or_else(|| root.pointer("/generationConfig/thinkingConfig/include_thoughts"))
        .or_else(|| root.pointer("/generationConfig/thinking_config/includeThoughts"))
        .or_else(|| root.pointer("/generationConfig/thinking_config/include_thoughts"))
        .or_else(|| root.pointer("/thinking/includeThoughts"))
        .or_else(|| root.pointer("/thinking/include_thoughts"))
        .and_then(Value::as_bool)
        .or_else(|| {
            root.pointer("/reasoning/exclude")
                .and_then(Value::as_bool)
                .map(|exclude| !exclude)
        })
        .or_else(|| {
            root.pointer("/extra_body/google/thinking_config/include_thoughts")
                .or_else(|| root.pointer("/extra_body/google/thinking_config/includeThoughts"))
                .and_then(Value::as_bool)
        });
    if let Some(include) = explicit_visibility {
        thinking.insert("includeThoughts".into(), Value::Bool(include));
    }
    if thinking.is_empty() {
        generation.remove("thinkingConfig");
    }
}

/// Canonicalizes the Antigravity thinking aliases while preserving the input
/// allocation when the payload is already canonical. This mirrors upstream's
/// sjson no-op behavior without introducing a mutable process-global cache.
pub(crate) fn normalize_antigravity_openai_thinking_config(input: Vec<u8>) -> Vec<u8> {
    let Ok(mut root) = serde_json::from_slice::<Value>(&input) else {
        return input;
    };
    let before = root.clone();
    let Some(generation) = root
        .pointer_mut("/request/generationConfig")
        .and_then(Value::as_object_mut)
    else {
        return input;
    };

    let legacy = generation
        .get("thinking_config")
        .and_then(Value::as_object)
        .cloned();
    let canonical = generation
        .get("thinkingConfig")
        .and_then(Value::as_object)
        .cloned();
    let mut normalized = canonical.clone().unwrap_or_default();
    for source in [legacy.as_ref(), canonical.as_ref()].into_iter().flatten() {
        for key in ["includeThoughts", "include_thoughts"] {
            if let Some(value) = source.get(key).and_then(Value::as_bool) {
                normalized.insert("includeThoughts".into(), Value::Bool(value));
            }
        }
        for (canonical_key, aliases) in [
            ("thinkingLevel", ["thinkingLevel", "thinking_level"]),
            ("thinkingBudget", ["thinkingBudget", "thinking_budget"]),
        ] {
            for alias in aliases {
                if let Some(value) = source.get(alias) {
                    normalized.insert(canonical_key.into(), value.clone());
                }
            }
        }
    }
    for key in ["includeThoughts", "include_thoughts"] {
        if let Some(value) = generation.get(key).and_then(Value::as_bool) {
            normalized.insert("includeThoughts".into(), Value::Bool(value));
        }
    }
    for alias in ["include_thoughts", "thinking_level", "thinking_budget"] {
        normalized.remove(alias);
    }
    if canonical
        .as_ref()
        .and_then(|value| value.get("includeThoughts"))
        .is_some_and(|value| !value.is_boolean())
    {
        normalized.remove("includeThoughts");
    }

    generation.remove("thinking_config");
    generation.remove("includeThoughts");
    generation.remove("include_thoughts");
    if !normalized.is_empty() || canonical.is_some() || legacy.is_some() {
        generation.insert("thinkingConfig".into(), Value::Object(normalized));
    }

    if root == before {
        input
    } else {
        serde_json::to_vec(&root).unwrap_or(input)
    }
}

fn adapt_inline_data_and_thinking(request: &mut Value) {
    if request
        .pointer("/generationConfig/thinkingConfig/includeThoughts")
        .is_none()
        && (request
            .pointer("/generationConfig/thinkingConfig/thinkingLevel")
            .is_some()
            || request
                .pointer("/generationConfig/thinkingConfig/thinkingBudget")
                .is_some())
    {
        request["generationConfig"]["thinkingConfig"]["includeThoughts"] = Value::Bool(true);
    }
    for part in request
        .get_mut("contents")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|content| content.get_mut("parts"))
        .filter_map(Value::as_array_mut)
        .flatten()
    {
        let Some(inline) = part.get_mut("inlineData").and_then(Value::as_object_mut) else {
            continue;
        };
        let mime = inline
            .get("mime_type")
            .or_else(|| inline.get("mimeType"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        if !mime.starts_with("audio/") {
            if let Some(value) = inline.remove("mime_type") {
                inline.insert("mimeType".into(), value);
            }
        }
    }
}

fn adapt_tools(root: &Value, request: &mut Value, names: &HashMap<String, String>) {
    let mut originals = Vec::new();
    let mut seen = HashSet::new();
    for tool in root
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            continue;
        }
        let name = tool
            .pointer("/function/name")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !name.is_empty() && seen.insert(name.to_owned()) {
            originals.push(name.to_owned());
        }
    }
    let declarations = request
        .get_mut("tools")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.get_mut("functionDeclarations"))
        .filter_map(Value::as_array_mut)
        .next();
    let Some(declarations) = declarations else {
        return;
    };
    declarations.truncate(originals.len());
    for (declaration, original) in declarations.iter_mut().zip(originals) {
        if let Some(mapped) = names.get(&original) {
            declaration["name"] = Value::String(mapped.clone());
        }
    }
}

fn adapt_tool_choice(root: &Value, request: &mut Value, names: &HashMap<String, String>) {
    let Some(choice) = root.get("tool_choice") else {
        return;
    };
    let (mode, allowed) = if let Some(choice) = choice.as_str() {
        match choice.trim().to_ascii_lowercase().as_str() {
            "none" => ("NONE", None),
            "auto" => ("AUTO", None),
            "required" | "any" => ("ANY", None),
            _ => return,
        }
    } else if choice.get("type").and_then(Value::as_str) == Some("function") {
        let original = choice
            .pointer("/function/name")
            .and_then(Value::as_str)
            .unwrap_or("");
        (
            "ANY",
            (!original.trim().is_empty()).then(|| {
                names
                    .get(original)
                    .cloned()
                    .unwrap_or_else(|| sanitize(original))
            }),
        )
    } else {
        return;
    };
    request["toolConfig"]["functionCallingConfig"]["mode"] = Value::String(mode.into());
    if let Some(allowed) = allowed {
        request["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"] = json!([allowed]);
    }
}

fn adapt_history(root: &Value, request: &mut Value, names: &HashMap<String, String>) {
    let mut calls = VecDeque::new();
    let mut outputs = HashMap::new();
    for message in root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if message.get("role").and_then(Value::as_str) == Some("assistant") {
            for call in message
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if call.get("type").and_then(Value::as_str) == Some("function") {
                    calls.push_back((
                        call.get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned(),
                        call.pointer("/function/name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned(),
                    ));
                }
            }
        } else if message.get("role").and_then(Value::as_str) == Some("tool") {
            let id = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            if !id.is_empty() {
                outputs.insert(
                    id.to_owned(),
                    message.get("content").cloned().unwrap_or(Value::Null),
                );
            }
        }
    }
    let mut response_ids = calls.clone();
    for content in request
        .get_mut("contents")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
    {
        for part in content
            .get_mut("parts")
            .and_then(Value::as_array_mut)
            .into_iter()
            .flatten()
        {
            if let Some(call) = part.get_mut("functionCall") {
                if let Some((id, original)) = calls.pop_front() {
                    call["id"] = Value::String(id);
                    call["name"] = Value::String(
                        names
                            .get(&original)
                            .cloned()
                            .unwrap_or_else(|| sanitize(&original)),
                    );
                }
            } else if let Some(response) = part.get_mut("functionResponse") {
                let id = response_ids
                    .pop_front()
                    .map(|value| value.0)
                    .unwrap_or_default();
                if !id.is_empty() {
                    response["id"] = Value::String(id.clone());
                    let original_name = root
                        .get("messages")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .flat_map(|message| {
                            message
                                .get("tool_calls")
                                .and_then(Value::as_array)
                                .into_iter()
                                .flatten()
                        })
                        .find(|call| call.get("id").and_then(Value::as_str) == Some(&id))
                        .and_then(|call| call.pointer("/function/name"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    response["name"] = Value::String(
                        names
                            .get(original_name)
                            .cloned()
                            .unwrap_or_else(|| sanitize(original_name)),
                    );
                    let value = outputs.get(&id).cloned().unwrap_or_else(|| json!({}));
                    response["response"] = if value.is_null() {
                        json!({})
                    } else if value.is_string() {
                        json!({"result":serde_json::to_string(&value).unwrap_or_default()})
                    } else {
                        json!({"result":value})
                    };
                }
            }
        }
    }
}

fn remove_video_parts(root: &mut Value) {
    for message in root
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
    {
        if let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) {
            parts.retain(|part| part.get("type").and_then(Value::as_str) != Some("video_url"));
        }
    }
}

pub(super) fn reverse_disambiguated_names(root: &[u8]) -> HashMap<String, String> {
    let root = serde_json::from_slice::<Value>(root).unwrap_or(Value::Null);
    disambiguated_names(&root)
        .into_iter()
        .filter(|(original, mapped)| original != mapped)
        .map(|(original, mapped)| (mapped, original))
        .collect()
}

fn disambiguated_names(root: &Value) -> HashMap<String, String> {
    let mut unique = HashSet::new();
    for tool in root
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        collect_names(tool, &mut unique);
    }
    let mut names = unique.into_iter().collect::<Vec<_>>();
    names.sort();
    let mut counts = HashMap::new();
    for name in &names {
        *counts.entry(sanitize(name)).or_insert(0_usize) += 1;
    }
    let mut used = HashSet::new();
    let mut output = HashMap::new();
    for name in names {
        let base = sanitize(&name);
        let mut mapped = base.clone();
        if counts.get(&base).copied().unwrap_or(0) > 1 || used.contains(&base) {
            for attempt in 0_u64.. {
                let mut hasher = Sha256::new();
                hasher.update(name.as_bytes());
                hasher.update([0]);
                hasher.update(attempt.to_string().as_bytes());
                let hash = hasher.finalize();
                let suffix = format!("_{}", hex12(&hash[..6]));
                let mut prefix = base.clone();
                prefix.truncate(prefix.len().min(64 - suffix.len()));
                let candidate = format!("{prefix}{suffix}");
                if !used.contains(&candidate) {
                    mapped = candidate;
                    break;
                }
            }
        }
        used.insert(mapped.clone());
        output.insert(name, mapped);
    }
    output
}

fn collect_names(tool: &Value, output: &mut HashSet<String>) {
    if let Some(tools) = tool.get("tools").and_then(Value::as_array) {
        for tool in tools {
            collect_names(tool, output);
        }
        return;
    }
    let declarations = tool
        .get("functionDeclarations")
        .or_else(|| tool.get("function_declarations"))
        .and_then(Value::as_array);
    if let Some(declarations) = declarations {
        for declaration in declarations {
            if let Some(name) = declaration.get("name").and_then(Value::as_str) {
                output.insert(name.to_owned());
            }
        }
        return;
    }
    if let Some(name) = tool
        .pointer("/function/name")
        .or_else(|| tool.get("name"))
        .and_then(Value::as_str)
    {
        output.insert(name.to_owned());
    }
}

fn sanitize(name: &str) -> String {
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
    if value.is_empty() {
        return "_".into();
    }
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

fn hex12(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::convert_openai_chat_request_to_antigravity;
    use serde_json::Value;

    #[test]
    fn wraps_request_and_disambiguates_colliding_function_names() {
        let output: Value = serde_json::from_slice(
            &convert_openai_chat_request_to_antigravity(
                "gemini-3",
                br#"{"messages":[{"role":"user","content":"run"}],"tools":[{"type":"function","function":{"name":"read file"}},{"type":"function","function":{"name":"read/file"}}],"tool_choice":{"type":"function","function":{"name":"read/file"}}}"#,
                false,
            ),
        )
        .unwrap();
        assert_eq!(output["model"], "gemini-3");
        assert_eq!(output["request"]["contents"][0]["parts"][0]["text"], "run");
        let declarations = output["request"]["tools"][0]["functionDeclarations"]
            .as_array()
            .unwrap();
        assert_eq!(declarations.len(), 2);
        assert_ne!(declarations[0]["name"], declarations[1]["name"]);
        assert_eq!(
            output["request"]["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0],
            declarations[1]["name"]
        );
    }
}
