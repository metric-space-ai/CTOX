// ref: internal/translator/openai/openai/responses/openai_openai-responses_tools.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Helpers shared by the OpenAI Responses ↔ OpenAI Chat Completions
//! translator. The request side flattens `function`/`namespace`/`custom`
//! tool declarations; the response side restores the namespace
//! qualification that the wire-level provider elides.

use serde_json::{json, Value};
use std::collections::HashSet;

/// Returns the JSON shape used to declare a freeform ("custom") tool to
/// the provider when the request was originally expressed in the
/// Responses shape. The single freeform "input" string mirrors the
/// function-based shape Codex uses for apply_patch style custom tools.
pub fn convert_responses_tool_to_openai_chat_tools(tool: &Value) -> Vec<Vec<u8>> {
    match responses_tool_type(tool) {
        "" | "function" => convert_responses_function_tool_to_openai_chat(tool, "")
            .into_iter()
            .collect(),
        "namespace" => convert_responses_namespace_tool_to_openai_chat(tool),
        "custom" => convert_responses_custom_tool_to_openai_chat(tool, "")
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

/// Mirrors [`convert_responses_tool_to_openai_chat_tools`] but lets the
/// caller override the function name when emitting namespace children
/// or re-decorated Responses calls.
pub fn convert_responses_function_tool_to_openai_chat(
    tool: &Value,
    override_name: &str,
) -> Option<Vec<u8>> {
    let name = effective_tool_name(tool, override_name);
    if name.is_empty() {
        return None;
    }
    let mut chat_tool = json!({
        "type":"function",
        "function":{"name":name,"description":"","parameters":{}}
    });
    if let Some(description) = responses_tool_description(tool) {
        chat_tool["function"]["description"] = Value::String(description);
    }
    if let Some(parameters) = responses_tool_parameters(tool) {
        chat_tool["function"]["parameters"] = parameters;
    }
    Some(serde_json::to_vec(&chat_tool).expect("static tool JSON cannot fail"))
}

pub fn convert_responses_custom_tool_to_openai_chat(
    tool: &Value,
    override_name: &str,
) -> Option<Vec<u8>> {
    let name = effective_tool_name(tool, override_name);
    if name.is_empty() {
        return None;
    }
    let mut chat_tool = json!({
        "type":"function",
        "function":{
            "name":name,
            "description":"",
            "parameters":{
                "type":"object",
                "properties":{"input":{"type":"string"}},
                "required":["input"]
            }
        }
    });
    if let Some(description) = responses_tool_description(tool) {
        chat_tool["function"]["description"] = Value::String(description);
    }
    Some(serde_json::to_vec(&chat_tool).expect("static tool JSON cannot fail"))
}

pub fn convert_responses_namespace_tool_to_openai_chat(tool: &Value) -> Vec<Vec<u8>> {
    let namespace_name = tool
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("")
        .to_string();
    let Some(children) = tool.get("tools").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for child in children {
        let child_name = responses_tool_name(child);
        let qualified = qualify_responses_namespace_tool_name(&namespace_name, &child_name);
        match responses_tool_type(child) {
            "" | "function" => {
                if let Some(tool) =
                    convert_responses_function_tool_to_openai_chat(child, &qualified)
                {
                    out.push(tool);
                }
            }
            "custom" => {
                if let Some(tool) = convert_responses_custom_tool_to_openai_chat(child, &qualified)
                {
                    out.push(tool);
                }
            }
            _ => {}
        }
    }
    out
}

pub fn responses_tool_name(tool: &Value) -> String {
    for path in ["name", "function.name"] {
        if let Some(name) = tool
            .get(path)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return name.to_string();
        }
    }
    String::new()
}

pub fn responses_tool_description(tool: &Value) -> Option<String> {
    for path in ["description", "function.description"] {
        if let Some(description) = tool
            .get(path)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Some(description.to_string());
        }
    }
    None
}

pub fn responses_tool_parameters(tool: &Value) -> Option<Value> {
    for path in [
        "parameters",
        "parametersJsonSchema",
        "input_schema",
        "function.parameters",
        "function.parametersJsonSchema",
    ] {
        if let Some(parameters) = tool.get(path) {
            if !parameters.is_null() {
                return Some(parameters.clone());
            }
        }
    }
    None
}

/// Flattens a tool output value that may be a plain string or an array
/// of content parts ({"type":"input_text","text":...}) into a single
/// text payload for a Chat Completions tool message.
pub fn responses_tool_output_text(output: &Value) -> String {
    if let Some(text) = output.as_str() {
        return text.to_string();
    }
    if let Some(parts) = output.as_array() {
        let mut buffer = String::new();
        for part in parts {
            if let Some(text) = part.as_str() {
                buffer.push_str(text);
                continue;
            }
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                buffer.push_str(text);
            }
        }
        return buffer;
    }
    if !output.is_null() {
        return serde_json::to_string(output).unwrap_or_default();
    }
    String::new()
}

/// Collects the names of freeform ("custom") tools declared in the
/// original Responses request, both in the top-level "tools" field and
/// in Codex Desktop "additional_tools" input items. Namespace child
/// names use the qualified Chat Completions form.
pub fn responses_custom_tool_names(request_raw_json: &[u8]) -> HashSet<String> {
    let mut names = HashSet::new();
    let root: Value = match serde_json::from_slice(request_raw_json) {
        Ok(value) => value,
        Err(_) => return names,
    };
    collect_custom_tool_names(root.get("tools"), "", &mut names);
    if let Some(items) = root.get("input").and_then(Value::as_array) {
        for item in items {
            if item.get("type").and_then(Value::as_str) == Some("additional_tools") {
                collect_custom_tool_names(item.get("tools"), "", &mut names);
            }
        }
    }
    names
}

fn collect_custom_tool_names(tools: Option<&Value>, namespace: &str, names: &mut HashSet<String>) {
    let Some(tools) = tools.and_then(Value::as_array) else {
        return;
    };
    for tool in tools {
        match responses_tool_type(tool) {
            "custom" => {
                let base = responses_tool_name(tool);
                let qualified = qualify_responses_namespace_tool_name(namespace, &base);
                if !qualified.is_empty() {
                    names.insert(qualified);
                }
            }
            "namespace" => {
                let namespace_name = tool
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .unwrap_or("");
                collect_custom_tool_names(tool.get("tools"), namespace_name, names);
            }
            _ => {}
        }
    }
}

/// Returns the single custom tool name when exactly one is declared and
/// the request contains no other tools. The bool is true when the
/// custom tool is the only tool in the request.
pub fn responses_single_custom_tool_name(request_raw_json: &[u8]) -> Option<(String, bool)> {
    let custom_tool_names = responses_custom_tool_names(request_raw_json);
    if custom_tool_names.len() != 1 {
        return None;
    }
    let root: Value = match serde_json::from_slice(request_raw_json) {
        Ok(value) => value,
        Err(_) => return None,
    };
    let tool_count = collect_total_tool_count(root.get("tools"))
        + collect_additional_tool_count(root.get("input"));
    custom_tool_names
        .into_iter()
        .next()
        .map(|name| (name, tool_count == 1))
}

fn collect_total_tool_count(tools: Option<&Value>) -> i64 {
    let mut count = 0;
    collect_tool_count(tools, &mut count);
    count
}

fn collect_additional_tool_count(input: Option<&Value>) -> i64 {
    let mut count = 0;
    if let Some(items) = input.and_then(Value::as_array) {
        for item in items {
            if item.get("type").and_then(Value::as_str) == Some("additional_tools") {
                collect_tool_count(item.get("tools"), &mut count);
            }
        }
    }
    count
}

fn collect_tool_count(tools: Option<&Value>, count: &mut i64) {
    let Some(tools) = tools.and_then(Value::as_array) else {
        return;
    };
    for tool in tools {
        *count += i64::try_from(convert_responses_tool_to_openai_chat_tools(tool).len())
            .unwrap_or(i32::MAX as i64);
    }
}

/// Extracts the freeform input from the {"input": "..."} function-call
/// arguments produced for a converted custom tool; it falls back to the
/// raw arguments when the wrapper is absent.
pub fn unwrap_custom_tool_input(arguments: &str) -> String {
    if let Ok(value) = serde_json::from_str::<Value>(arguments) {
        if let Some(input) = value.get("input") {
            return match input {
                Value::String(text) => text.clone(),
                value => serde_json::to_string(value).unwrap_or_default(),
            };
        }
    }
    arguments.to_string()
}

/// Joins a namespace and child name with the conventional `__` separator
/// unless either side already implies the prefix. This mirrors the
/// Chat Completions wire shape used for Codex MCP / collaboration tools.
pub fn qualify_responses_namespace_tool_name(namespace_name: &str, child_name: &str) -> String {
    let child = child_name.trim();
    if child.is_empty() || namespace_name.is_empty() {
        return child.to_string();
    }
    if child.starts_with("mcp__") {
        return child.to_string();
    }
    if child.starts_with(namespace_name) {
        return child.to_string();
    }
    if namespace_name.ends_with("__") {
        format!("{namespace_name}{child}")
    } else {
        format!("{namespace_name}__{child}")
    }
}

/// Walks the original Responses request to find the namespace child
/// whose qualified name matches `qualified_name` and returns the
/// un-prefixed (name, namespace) pair. Returns the qualified name and
/// an empty namespace when no match is found.
pub fn split_responses_qualified_function_call_from_request(
    request_raw_json: &[u8],
    qualified_name: &str,
) -> (String, String) {
    let qualified = qualified_name.trim();
    if qualified.is_empty() {
        return (String::new(), String::new());
    }
    let root: Value = match serde_json::from_slice(request_raw_json) {
        Ok(value) => value,
        Err(_) => return (qualified.to_string(), String::new()),
    };
    let mut best: (String, String) = (qualified.to_string(), String::new());
    scan_tools_for_qualified(root.get("tools"), qualified, &mut best);
    if let Some(items) = root.get("input").and_then(Value::as_array) {
        for item in items {
            if item.get("type").and_then(Value::as_str) == Some("additional_tools") {
                scan_tools_for_qualified(item.get("tools"), qualified, &mut best);
            }
        }
    }
    best
}

fn scan_tools_for_qualified(
    tools: Option<&Value>,
    qualified_name: &str,
    best: &mut (String, String),
) {
    let Some(tools) = tools.and_then(Value::as_array) else {
        return;
    };
    for tool in tools {
        if responses_tool_type(tool) != "namespace" {
            continue;
        }
        let namespace = tool
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or("");
        if namespace.is_empty() {
            continue;
        }
        let Some(children) = tool.get("tools").and_then(Value::as_array) else {
            continue;
        };
        for child in children {
            let child_name = responses_tool_name(child);
            if child_name.is_empty() {
                continue;
            }
            if qualify_responses_namespace_tool_name(namespace, &child_name) == qualified_name {
                *best = (child_name, namespace.to_string());
            }
        }
    }
}

/// Mirrors the Go helper that picks the original (pre-translation)
/// request JSON when valid, falling back to the translated form.
pub fn pick_request_json<'a>(original: &'a [u8], request: &'a [u8]) -> Option<&'a [u8]> {
    if !original.is_empty() && is_valid_json(original) {
        Some(original)
    } else if !request.is_empty() && is_valid_json(request) {
        Some(request)
    } else {
        None
    }
}

fn is_valid_json(raw: &[u8]) -> bool {
    serde_json::from_slice::<Value>(raw).is_ok()
}

/// Restores the `name`/`namespace` fields of a function_call output
/// item built from a Chat Completions tool call, using the original
/// Responses request to map the qualified wire name back to its
/// namespace.
pub fn apply_responses_function_call_namespace_fields(
    mut item: Value,
    request_raw_json: &[u8],
    qualified_name: &str,
    item_path: &str,
) -> Value {
    let (name, namespace) =
        split_responses_qualified_function_call_from_request(request_raw_json, qualified_name);
    let (name_path, namespace_path) = if item_path.is_empty() {
        ("name".to_string(), "namespace".to_string())
    } else {
        (
            format!("{item_path}.name"),
            format!("{item_path}.namespace"),
        )
    };
    set_path(&mut item, &name_path, Value::String(name));
    if namespace.is_empty() {
        delete_path(&mut item, &namespace_path);
    } else {
        set_path(&mut item, &namespace_path, Value::String(namespace));
    }
    item
}

fn set_path(root: &mut Value, path: &str, value: Value) {
    let Some((head, tail)) = path.split_once('.') else {
        if let Some(object) = root.as_object_mut() {
            object.insert(path.to_string(), value);
        }
        return;
    };
    if let Some(object) = root.as_object_mut() {
        let entry = object
            .entry(head.to_string())
            .or_insert_with(|| Value::Object(Default::default()));
        set_path(entry, tail, value);
    }
}

fn delete_path(root: &mut Value, path: &str) {
    let Some((head, tail)) = path.split_once('.') else {
        if let Some(object) = root.as_object_mut() {
            object.remove(path);
        }
        return;
    };
    if let Some(object) = root.as_object_mut() {
        if let Some(child) = object.get_mut(head) {
            delete_path(child, tail);
        }
    }
}

fn responses_tool_type(tool: &Value) -> &str {
    tool.get("type")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("")
}

fn effective_tool_name(tool: &Value, override_name: &str) -> String {
    let trimmed = override_name.trim();
    if !trimmed.is_empty() {
        return trimmed.to_string();
    }
    responses_tool_name(tool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualify_uses_double_underscore_separator() {
        assert_eq!(
            qualify_responses_namespace_tool_name("mcp__test", "add"),
            "mcp__test__add"
        );
        assert_eq!(
            qualify_responses_namespace_tool_name("mcp__test__", "add"),
            "mcp__test__add"
        );
        assert_eq!(
            qualify_responses_namespace_tool_name("mcp__test", "mcp__test__add"),
            "mcp__test__add"
        );
        assert_eq!(qualify_responses_namespace_tool_name("", "add"), "add");
    }

    #[test]
    fn split_recovers_namespace_from_request() {
        let raw = br#"{
            "tools":[
                {"type":"namespace","name":"mcp__github","tools":[
                    {"type":"function","name":"get_me"}
                ]}
            ]
        }"#;
        let (name, namespace) =
            split_responses_qualified_function_call_from_request(raw, "mcp__github__get_me");
        assert_eq!(name, "get_me");
        assert_eq!(namespace, "mcp__github");
    }

    #[test]
    fn unwrap_input_returns_raw_string_when_not_json() {
        assert_eq!(unwrap_custom_tool_input("not json"), "not json");
        assert_eq!(unwrap_custom_tool_input(r#"{"input":"ls"}"#), "ls");
    }
}
