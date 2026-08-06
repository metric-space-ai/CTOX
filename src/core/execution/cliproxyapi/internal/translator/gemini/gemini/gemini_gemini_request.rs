// ref: internal/translator/gemini/gemini/gemini_gemini_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::internal::signature::sanitize_gemini_request_thought_signatures;
use crate::internal::translator::gemini::common::attach_default_safety_settings;

pub fn convert_gemini_request_to_gemini(
    _model_name: &str,
    input_raw_json: &[u8],
    _stream: bool,
) -> Vec<u8> {
    let Ok(mut root) = serde_json::from_slice::<Value>(input_raw_json) else {
        return input_raw_json.to_vec();
    };
    let Some(object) = root.as_object_mut() else {
        return input_raw_json.to_vec();
    };
    if !object.contains_key("contents") {
        return attach_default_safety_settings(input_raw_json, "safetySettings");
    }
    let mut changed = normalize_tools(object.get_mut("tools"));

    if let Some(contents) = object.get_mut("contents").and_then(Value::as_array_mut) {
        let mut previous_role = "";
        for content in contents.iter_mut() {
            let Some(content) = content.as_object_mut() else {
                continue;
            };
            let role = content.get("role").and_then(Value::as_str).unwrap_or("");
            let role = if matches!(role, "user" | "model") {
                role
            } else if previous_role.is_empty() || previous_role == "model" {
                content.insert("role".to_owned(), Value::String("user".to_owned()));
                changed = true;
                "user"
            } else {
                content.insert("role".to_owned(), Value::String("model".to_owned()));
                changed = true;
                "model"
            };
            previous_role = role;
        }
    }

    let stage = if changed {
        serde_json::to_vec(&root).unwrap_or_else(|_| input_raw_json.to_vec())
    } else {
        input_raw_json.to_vec()
    };
    let sanitized = sanitize_gemini_request_thought_signatures(&stage);
    let Ok(mut root) = serde_json::from_slice::<Value>(&sanitized) else {
        return sanitized;
    };
    let mut changed_after = false;

    if let Some(generation) = root
        .get_mut("generationConfig")
        .and_then(Value::as_object_mut)
    {
        if let Some(schema) = generation.remove("responseSchema") {
            generation.insert("responseJsonSchema".to_owned(), schema);
            changed_after = true;
        }
    }
    changed_after |= backfill_empty_function_response_names(&mut root);

    let output = if changed_after {
        serde_json::to_vec(&root).unwrap_or(sanitized)
    } else {
        sanitized
    };
    attach_default_safety_settings(&output, "safetySettings")
}

fn normalize_tools(value: Option<&mut Value>) -> bool {
    let Some(tools) = value.and_then(Value::as_array_mut) else {
        return false;
    };
    let mut changed = false;
    for tool in tools.iter_mut().filter_map(Value::as_object_mut) {
        if let Some(declarations) = tool.remove("functionDeclarations") {
            tool.insert("function_declarations".to_owned(), declarations);
            changed = true;
        }
        if let Some(declarations) = tool
            .get_mut("function_declarations")
            .and_then(Value::as_array_mut)
        {
            for declaration in declarations.iter_mut().filter_map(Value::as_object_mut) {
                if let Some(parameters) = declaration.remove("parameters") {
                    declaration.insert("parametersJsonSchema".to_owned(), parameters);
                    changed = true;
                }
            }
        }
    }
    changed
}

fn backfill_empty_function_response_names(root: &mut Value) -> bool {
    let Some(contents) = root.get_mut("contents").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut pending_names = Vec::<String>::new();
    let mut changed = false;
    for content in contents {
        let is_model = content.get("role").and_then(Value::as_str) == Some("model");
        let Some(parts) = content.get_mut("parts").and_then(Value::as_array_mut) else {
            continue;
        };
        if is_model {
            pending_names = parts
                .iter()
                .filter_map(|part| part.pointer("/functionCall/name").and_then(Value::as_str))
                .map(str::to_owned)
                .collect();
            continue;
        }
        if pending_names.is_empty() {
            continue;
        }
        let mut response_index = 0;
        for part in parts {
            let Some(response) = part
                .get_mut("functionResponse")
                .and_then(Value::as_object_mut)
            else {
                continue;
            };
            let empty = response
                .get("name")
                .and_then(Value::as_str)
                .is_none_or(|name| name.trim().is_empty());
            if empty {
                if let Some(name) = pending_names.get(response_index) {
                    response.insert("name".to_owned(), Value::String(name.clone()));
                    changed = true;
                }
            }
            response_index += 1;
        }
        pending_names.clear();
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_roles_tools_schema_signatures_and_function_names() {
        let output = convert_gemini_request_to_gemini(
            "gemini-3",
            br#"{"tools":[{"functionDeclarations":[{"name":"lookup","parameters":{"type":"object"}}]}],"contents":[{"parts":[{"functionCall":{"name":"lookup"}}]},{"role":"bad","parts":[{"functionResponse":{"name":" ","response":{"ok":true}}}]}],"generationConfig":{"responseSchema":{"type":"string"}}}"#,
            false,
        );
        let output: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(output["contents"][0]["role"], "user");
        assert_eq!(output["contents"][1]["role"], "model");
        assert!(output["tools"][0].get("functionDeclarations").is_none());
        assert_eq!(
            output["tools"][0]["function_declarations"][0]["parametersJsonSchema"]["type"],
            "object"
        );
        assert!(output["generationConfig"].get("responseSchema").is_none());
        assert_eq!(output["safetySettings"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn backfills_immediately_following_user_responses() {
        let output = convert_gemini_request_to_gemini(
            "",
            br#"{"contents":[{"role":"model","parts":[{"functionCall":{"name":"a"}},{"functionCall":{"name":"b"}}]},{"role":"user","parts":[{"functionResponse":{"name":"","response":{}}},{"functionResponse":{"response":{}}}]}],"safetySettings":null}"#,
            false,
        );
        let output: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(
            output["contents"][1]["parts"][0]["functionResponse"]["name"],
            "a"
        );
        assert_eq!(
            output["contents"][1]["parts"][1]["functionResponse"]["name"],
            "b"
        );
        assert!(output["safetySettings"].is_null());
    }
}
