// ref: internal/runtime/executor/xai_executor_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use super::xai_executor_request::{ClientToolKey, NamespaceToolRef};

pub struct InternalXSearchResponseFilter {
    enabled: bool,
    declared: BTreeSet<ClientToolKey>,
    dropped_ids: BTreeSet<String>,
}

impl InternalXSearchResponseFilter {
    #[must_use]
    pub fn new(enabled: bool, declared: BTreeSet<ClientToolKey>) -> Self {
        Self {
            enabled,
            declared,
            dropped_ids: BTreeSet::new(),
        }
    }

    #[must_use]
    pub fn apply(&mut self, data: &[u8]) -> Vec<u8> {
        if !self.enabled {
            return data.to_vec();
        }
        let Ok(mut event) = serde_json::from_slice::<Value>(data) else {
            return data.to_vec();
        };
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if matches!(
            event_type,
            "response.output_item.added" | "response.output_item.done"
        ) {
            if let Some(item) = event.get("item") {
                if self.is_internal_call(item) {
                    if let Some(id) = call_id(item) {
                        self.dropped_ids.insert(id.to_owned());
                    }
                    return Vec::new();
                }
            }
        }
        if event_type == "response.completed" {
            if let Some(output) = event
                .pointer_mut("/response/output")
                .and_then(Value::as_array_mut)
            {
                output.retain(|item| !self.is_internal_call(item));
                for (index, item) in output.iter_mut().enumerate() {
                    if let Some(item) = item.as_object_mut() {
                        item.insert("output_index".into(), (index as u64).into());
                    }
                }
            }
        } else if event
            .get("item_id")
            .and_then(Value::as_str)
            .is_some_and(|id| self.dropped_ids.contains(id))
        {
            return Vec::new();
        }
        serde_json::to_vec(&event).unwrap_or_else(|_| data.to_vec())
    }

    fn is_internal_call(&self, item: &Value) -> bool {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
        let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
        let internal_name = name == "x_search"
            || name.starts_with("x_search__")
            || name.starts_with("tool_search__x_search");
        internal_name
            && !self.declared.contains(&ClientToolKey {
                tool_type: effective_declared_type(item_type).to_owned(),
                name: name.to_owned(),
            })
    }
}

fn effective_declared_type(item_type: &str) -> &str {
    match item_type {
        "function_call" | "custom_tool_call" => "function",
        other => other,
    }
}
fn call_id(item: &Value) -> Option<&str> {
    item.get("call_id")
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
}

#[must_use]
pub fn restore_namespace_tool_calls(
    data: &[u8],
    refs: &BTreeMap<String, NamespaceToolRef>,
) -> Vec<u8> {
    let Ok(mut value) = serde_json::from_slice::<Value>(data) else {
        return data.to_vec();
    };
    restore_namespace_value(&mut value, refs);
    serde_json::to_vec(&value).unwrap_or_else(|_| data.to_vec())
}

fn restore_namespace_value(value: &mut Value, refs: &BTreeMap<String, NamespaceToolRef>) {
    match value {
        Value::Array(values) => values
            .iter_mut()
            .for_each(|value| restore_namespace_value(value, refs)),
        Value::Object(object) => {
            let item_type = object
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            if matches!(item_type.as_str(), "function_call" | "custom_tool_call") {
                if let Some(flattened) = object
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                {
                    if let Some(reference) = refs.get(&flattened) {
                        object.insert("name".into(), Value::String(reference.name.clone()));
                        object.insert(
                            "namespace".into(),
                            Value::String(reference.namespace.clone()),
                        );
                    }
                }
            }
            for (key, child) in object {
                if key != "arguments" && key != "input" {
                    restore_namespace_value(child, refs);
                }
            }
        }
        _ => {}
    }
}

#[must_use]
pub fn sanitize_input_encrypted_content(body: &[u8]) -> Vec<u8> {
    let Ok(mut root) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    if let Some(input) = root.get_mut("input").and_then(Value::as_array_mut) {
        input.retain_mut(|item| {
            if item.get("type").and_then(Value::as_str) != Some("reasoning") {
                return true;
            }
            let valid = item
                .get("encrypted_content")
                .and_then(Value::as_str)
                .is_some_and(valid_encrypted_content);
            if !valid {
                item.as_object_mut()
                    .map(|object| object.remove("encrypted_content"));
            }
            item.get("summary")
                .and_then(Value::as_array)
                .is_some_and(|summary| !summary.is_empty())
                || valid
        });
    }
    serde_json::to_vec(&root).unwrap_or_else(|_| body.to_vec())
}

fn valid_encrypted_content(value: &str) -> bool {
    value.len() >= 16
        && base64::Engine::decode(&base64::engine::general_purpose::STANDARD, value).is_ok()
}

#[must_use]
pub fn normalize_reasoning_event_name(name: &str) -> &str {
    match name {
        "response.reasoning_text.delta" => "response.reasoning_summary_text.delta",
        "response.reasoning_text.done" => "response.reasoning_summary_text.done",
        other => other,
    }
}

#[must_use]
pub fn normalize_reasoning_event_data(data: &[u8]) -> Vec<u8> {
    let Ok(mut value) = serde_json::from_slice::<Value>(data) else {
        return data.to_vec();
    };
    if let Some(kind) = value.get("type").and_then(Value::as_str).map(str::to_owned) {
        value.as_object_mut().unwrap().insert(
            "type".into(),
            Value::String(normalize_reasoning_event_name(&kind).to_owned()),
        );
    }
    normalize_reasoning_items(&mut value);
    serde_json::to_vec(&value).unwrap_or_else(|_| data.to_vec())
}

fn normalize_reasoning_items(value: &mut Value) {
    match value {
        Value::Array(items) => items.iter_mut().for_each(normalize_reasoning_items),
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("reasoning") {
                if let Some(text) = object.remove("reasoning_text") {
                    object.entry("summary").or_insert_with(
                        || serde_json::json!([{"type":"summary_text","text":text}]),
                    );
                }
            }
            for child in object.values_mut() {
                normalize_reasoning_items(child);
            }
        }
        _ => {}
    }
}

#[must_use]
pub fn normalize_sse_stream(
    chunk: &[u8],
    filter: &mut InternalXSearchResponseFilter,
    refs: &BTreeMap<String, NamespaceToolRef>,
) -> Vec<Vec<u8>> {
    let mut output = Vec::new();
    let text = String::from_utf8_lossy(chunk);
    let mut event_name = String::new();
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("event:") {
            event_name = normalize_reasoning_event_name(name.trim()).to_owned();
        } else if let Some(data) = line.strip_prefix("data:") {
            let data = normalize_reasoning_event_data(data.trim().as_bytes());
            let data = restore_namespace_tool_calls(&data, refs);
            let data = filter.apply(&data);
            if !data.is_empty() {
                let actual = serde_json::from_slice::<Value>(&data)
                    .ok()
                    .and_then(|v| v.get("type").and_then(Value::as_str).map(str::to_owned))
                    .unwrap_or_else(|| event_name.clone());
                output.push(
                    format!(
                        "event: {actual}\ndata: {}\n\n",
                        String::from_utf8_lossy(&data)
                    )
                    .into_bytes(),
                );
            }
        }
    }
    output
}
