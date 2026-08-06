// ref: internal/translator/antigravity/gemini/antigravity_gemini_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::{json, Value};

use crate::internal::signature::{
    compatible_antigravity_claude_thinking_signature, sanitize_gemini_request_thought_signatures,
};
use crate::internal::translator::gemini::common::attach_default_safety_settings;
use crate::internal::util::{map_sanitized_function_name, sanitized_function_name_map};

pub fn convert_gemini_request_to_antigravity(
    model_name: &str,
    input_raw_json: &[u8],
    _stream: bool,
) -> Vec<u8> {
    let Ok(mut request) = serde_json::from_slice::<Value>(input_raw_json) else {
        return Vec::new();
    };
    if request.get("contents").is_none() {
        return Vec::new();
    }
    let name_map = sanitized_function_name_map(input_raw_json);
    request.as_object_mut().map(|object| object.remove("model"));
    group_cli_tool_responses(&mut request);
    if let Some(object) = request.as_object_mut() {
        if let Some(system) = object.remove("system_instruction") {
            object.insert("systemInstruction".to_owned(), system);
        }
    }
    normalize_roles(&mut request);
    normalize_tools(&mut request, &name_map);
    rewrite_function_names(&mut request, &name_map);
    sanitize_signatures(model_name, &mut request);

    let wrapped = serde_json::to_vec(&json!({"project":"","request":request,"model":model_name}))
        .unwrap_or_default();
    attach_default_safety_settings(&wrapped, "request.safetySettings")
}

fn normalize_roles(request: &mut Value) {
    let Some(contents) = request.get_mut("contents").and_then(Value::as_array_mut) else {
        return;
    };
    let mut previous = "";
    for content in contents.iter_mut().filter_map(Value::as_object_mut) {
        let role = content.get("role").and_then(Value::as_str).unwrap_or("");
        let normalized = if matches!(role, "user" | "model") {
            role
        } else if previous.is_empty() || previous == "model" {
            content.insert("role".to_owned(), Value::String("user".to_owned()));
            "user"
        } else {
            content.insert("role".to_owned(), Value::String("model".to_owned()));
            "model"
        };
        previous = normalized;
    }
}

fn normalize_tools(request: &mut Value, name_map: &HashMap<String, String>) {
    let Some(tools) = request.get_mut("tools").and_then(Value::as_array_mut) else {
        return;
    };
    let mut seen = HashSet::new();
    for tool in tools.iter_mut().filter_map(Value::as_object_mut) {
        for key in ["functionDeclarations", "function_declarations"] {
            let Some(declarations) = tool.get_mut(key).and_then(Value::as_array_mut) else {
                continue;
            };
            declarations.retain_mut(|declaration| {
                let Some(declaration) = declaration.as_object_mut() else {
                    return true;
                };
                let original = declaration
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let mapped = map_sanitized_function_name(name_map, original);
                if !mapped.is_empty() && !seen.insert(mapped.clone()) {
                    return false;
                }
                declaration.insert("name".to_owned(), Value::String(mapped));
                if let Some(parameters) = declaration.remove("parameters") {
                    declaration.insert("parametersJsonSchema".to_owned(), parameters);
                }
                true
            });
        }
    }
    tools.retain_mut(|tool| {
        let Some(object) = tool.as_object_mut() else {
            return true;
        };
        for key in ["functionDeclarations", "function_declarations"] {
            if object
                .get(key)
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
            {
                object.remove(key);
            }
        }
        !object.is_empty()
    });
    if tools.is_empty() {
        request.as_object_mut().map(|object| object.remove("tools"));
    }
}

fn rewrite_function_names(request: &mut Value, name_map: &HashMap<String, String>) {
    for part in request
        .get_mut("contents")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|content| content.get_mut("parts"))
        .filter_map(Value::as_array_mut)
        .flatten()
    {
        for field in [
            "functionCall",
            "functionResponse",
            "function_call",
            "function_response",
        ] {
            if let Some(call) = part.get_mut(field).and_then(Value::as_object_mut) {
                if let Some(name) = call.get("name").and_then(Value::as_str) {
                    call.insert(
                        "name".to_owned(),
                        Value::String(map_sanitized_function_name(name_map, name)),
                    );
                }
            }
        }
    }
    for pointer in [
        "/toolConfig/functionCallingConfig/allowedFunctionNames",
        "/tool_config/function_calling_config/allowed_function_names",
    ] {
        if let Some(names) = request.pointer_mut(pointer).and_then(Value::as_array_mut) {
            for name in names {
                let current = name.as_str().unwrap_or("");
                *name = Value::String(map_sanitized_function_name(name_map, current));
            }
        }
    }
}

fn group_cli_tool_responses(request: &mut Value) {
    let Some(contents) = request.get_mut("contents").and_then(Value::as_array_mut) else {
        return;
    };
    if !contents.iter().any(|content| {
        content
            .get("parts")
            .and_then(Value::as_array)
            .is_some_and(|parts| {
                parts
                    .iter()
                    .any(|part| part.get("functionResponse").is_some())
            })
    }) {
        return;
    }
    let original = std::mem::take(contents);
    let mut output = Vec::new();
    let mut pending = VecDeque::<Vec<PendingCall>>::new();
    let mut responses = VecDeque::<Value>::new();
    for content in original {
        let response_parts = content
            .get("parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|part| part.get("functionResponse").is_some())
            .cloned()
            .collect::<Vec<_>>();
        if !response_parts.is_empty() {
            responses.extend(response_parts);
            flush_groups(&mut output, &mut pending, &mut responses);
            continue;
        }
        let calls = if content.get("role").and_then(Value::as_str) == Some("model") {
            content
                .get("parts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|part| {
                    let call = part.get("functionCall")?;
                    Some(PendingCall {
                        id: call
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned(),
                        name: call
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned(),
                    })
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        output.push(content);
        if !calls.is_empty() {
            pending.push_back(calls);
        }
    }
    flush_groups(&mut output, &mut pending, &mut responses);
    *contents = output;
}

struct PendingCall {
    id: String,
    name: String,
}

fn flush_groups(
    output: &mut Vec<Value>,
    pending: &mut VecDeque<Vec<PendingCall>>,
    responses: &mut VecDeque<Value>,
) {
    while pending
        .front()
        .is_some_and(|names| responses.len() >= names.len())
    {
        let names = pending.pop_front().unwrap();
        let mut parts = Vec::new();
        for call in names {
            let matching_index = responses.iter().position(|part| {
                let response = part.get("functionResponse");
                let response_id = response
                    .and_then(|value| value.get("id"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let response_name = response
                    .and_then(|value| value.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                (!call.id.is_empty() && response_id == call.id)
                    || (call.id.is_empty() && !call.name.is_empty() && response_name == call.name)
            });
            let mut part = matching_index
                .and_then(|index| responses.remove(index))
                .or_else(|| responses.pop_front())
                .unwrap();
            if let Some(response) = part
                .get_mut("functionResponse")
                .and_then(Value::as_object_mut)
            {
                if response
                    .get("name")
                    .and_then(Value::as_str)
                    .is_none_or(|value| value.trim().is_empty())
                {
                    response.insert("name".to_owned(), Value::String(call.name));
                }
            }
            parts.push(part);
        }
        output.push(json!({"parts":parts,"role":"function"}));
    }
}

fn sanitize_signatures(model_name: &str, request: &mut Value) {
    if !model_name.to_ascii_lowercase().contains("claude") {
        let bytes = serde_json::to_vec(request).unwrap_or_default();
        if let Ok(value) =
            serde_json::from_slice(&sanitize_gemini_request_thought_signatures(&bytes))
        {
            *request = value;
        }
        return;
    }
    let Some(contents) = request.get_mut("contents").and_then(Value::as_array_mut) else {
        return;
    };
    contents.retain_mut(|content| {
        let is_model = content.get("role").and_then(Value::as_str) == Some("model");
        let Some(parts) = content.get_mut("parts").and_then(Value::as_array_mut) else {
            return true;
        };
        parts.retain_mut(|part| sanitize_claude_part(part, is_model));
        !parts.is_empty()
    });
}

/// Sanitizes the already wrapped Antigravity request used by sibling protocol
/// translators. The conversion is request-local and preserves the original
/// bytes when no signature or empty-turn rewrite is needed.
pub(crate) fn sanitize_antigravity_claude_gemini_request_signatures(
    model_name: &str,
    input: Vec<u8>,
) -> Vec<u8> {
    if !model_name.to_ascii_lowercase().contains("claude") {
        return input;
    }
    let Ok(mut root) = serde_json::from_slice::<Value>(&input) else {
        return input;
    };
    let before = root.clone();
    let Some(request) = root.get_mut("request") else {
        return input;
    };
    sanitize_signatures(model_name, request);
    if root == before {
        input
    } else {
        serde_json::to_vec(&root).unwrap_or(input)
    }
}

fn sanitize_claude_part(part: &mut Value, is_model: bool) -> bool {
    let signature = signature_value(part).unwrap_or("").to_owned();
    let has_signature = !signature.is_empty();
    if part.get("functionResponse").is_some()
        || part.get("function_response").is_some()
        || !is_model
    {
        if has_signature {
            remove_signature_fields(part);
        }
        return true;
    }
    if part.get("thought").and_then(Value::as_bool) == Some(true) {
        let Some(normalized) = compatible_antigravity_claude_thinking_signature(&signature) else {
            return false;
        };
        if part
            .get("text")
            .and_then(Value::as_str)
            .is_none_or(|text| text.trim().is_empty())
        {
            return false;
        }
        remove_signature_fields(part);
        part.as_object_mut()
            .unwrap()
            .insert("thoughtSignature".to_owned(), Value::String(normalized));
        return true;
    }
    if has_signature {
        remove_signature_fields(part);
    }
    true
}

fn signature_value(part: &Value) -> Option<&str> {
    part.get("thoughtSignature")
        .or_else(|| part.get("thought_signature"))
        .or_else(|| part.pointer("/functionCall/thoughtSignature"))
        .or_else(|| part.pointer("/functionResponse/thoughtSignature"))
        .or_else(|| part.pointer("/extra_content/google/thought_signature"))
        .and_then(Value::as_str)
}

fn remove_signature_fields(part: &mut Value) {
    let Some(object) = part.as_object_mut() else {
        return;
    };
    object.remove("thoughtSignature");
    object.remove("thought_signature");
    for field in ["functionCall", "functionResponse"] {
        if let Some(value) = object.get_mut(field).and_then(Value::as_object_mut) {
            value.remove("thoughtSignature");
            value.remove("thought_signature");
        }
    }
    if let Some(google) = object
        .get_mut("extra_content")
        .and_then(Value::as_object_mut)
        .and_then(|v| v.get_mut("google"))
        .and_then(Value::as_object_mut)
    {
        google.remove("thought_signature");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_groups_sanitizes_and_normalizes_request() {
        let output = convert_gemini_request_to_antigravity("gemini-3", br#"{"model":"client","system_instruction":{"parts":[{"text":"rules"}]},"contents":[{"role":"model","parts":[{"functionCall":{"name":"read file"}}]},{"parts":[{"functionResponse":{"name":"","response":{"result":"ok"}}}]}],"tools":[{"functionDeclarations":[{"name":"read file","parameters":{"type":"object"}}]}]}"#, false);
        let output: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(output["model"], "gemini-3");
        assert!(output["request"].get("model").is_none());
        assert!(output["request"].get("system_instruction").is_none());
        assert_eq!(output["request"]["contents"][1]["role"], "user");
        assert_eq!(
            output["request"]["contents"][1]["parts"][0]["functionResponse"]["name"],
            "read_file"
        );
        assert_eq!(
            output["request"]["safetySettings"]
                .as_array()
                .unwrap()
                .len(),
            5
        );
    }
}
