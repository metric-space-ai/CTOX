// ref: internal/runtime/executor/helps/payload_mutations.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde::Serialize;
use serde_json::Value;

use crate::internal::util::get_gjson_bytes_no_copy;

/// Updates `path` only when its value is not already the canonical JSON string.
/// Values with another JSON type are still normalized. The owned input is
/// returned directly on no-op and error paths, retaining its allocation and
/// every byte exactly as upstream retains the input slice.
#[must_use]
pub fn set_string_if_different(mut payload: Vec<u8>, path: &str, value: &str) -> Vec<u8> {
    let current = get_gjson_bytes_no_copy(&payload, path);
    if current.kind() == gjson::Kind::String && current.str() == value {
        return payload;
    }
    let replacement = Value::String(value.to_owned());
    set_json_path(&mut payload, path, replacement);
    payload
}

/// Updates `path` only when its value is not already the canonical JSON bool.
#[must_use]
pub fn set_bool_if_different(mut payload: Vec<u8>, path: &str, value: bool) -> Vec<u8> {
    let current = get_gjson_bytes_no_copy(&payload, path);
    if (value && current.kind() == gjson::Kind::True)
        || (!value && current.kind() == gjson::Kind::False)
    {
        return payload;
    }
    set_json_path(&mut payload, path, Value::Bool(value));
    payload
}

/// Updates `path` only when the existing raw JSON is byte-identical.
#[must_use]
pub fn set_raw_if_different(mut payload: Vec<u8>, path: &str, value: &[u8]) -> Vec<u8> {
    let current = get_gjson_bytes_no_copy(&payload, path);
    if current.exists() && current.json().as_bytes() == value {
        return payload;
    }
    let Ok(replacement) = serde_json::from_slice::<Value>(value) else {
        return payload;
    };
    set_json_path(&mut payload, path, replacement);
    payload
}

/// Joins validated raw JSON array items without re-encoding them.
#[must_use]
pub fn join_raw_json_array(items: &[impl AsRef<[u8]>]) -> Vec<u8> {
    let size = items
        .iter()
        .fold(items.len() + 1, |size, item| size + item.as_ref().len());
    let mut output = Vec::with_capacity(size);
    output.push(b'[');
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            output.push(b',');
        }
        output.extend_from_slice(item.as_ref());
    }
    output.push(b']');
    output
}

/// Joins raw JSON array items held as strings.
#[must_use]
pub fn join_raw_json_strings(items: &[String]) -> Vec<u8> {
    join_raw_json_array(items)
}

/// Serialization-once equivalent of upstream's `setPayloadValueIfDifferent`.
/// This remains crate-visible because payload rule application owns its use.
#[must_use]
pub fn set_payload_value_if_different<T: Serialize>(
    payload: Vec<u8>,
    path: &str,
    value: &T,
) -> Vec<u8> {
    let Ok(expected) = serde_json::to_vec(value) else {
        return payload;
    };
    set_raw_if_different(payload, path, &expected)
}

/// Removes tools with exactly the requested type. As in upstream, the
/// comparison is case-sensitive and a missing/non-array/no-match path is a
/// byte- and allocation-preserving no-op.
#[must_use]
pub fn remove_tool_type_from_tools_array(
    mut payload: Vec<u8>,
    tools_path: &str,
    tool_type: &str,
) -> Vec<u8> {
    let Ok(mut document) = serde_json::from_slice::<Value>(&payload) else {
        return payload;
    };
    let Some(tools) = value_at_path_mut(&mut document, tools_path).and_then(Value::as_array_mut)
    else {
        return payload;
    };
    if !tools.iter().any(|tool| {
        tool.get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind == tool_type)
    }) {
        return payload;
    }
    tools.retain(|tool| tool.get("type").and_then(Value::as_str) != Some(tool_type));
    if let Ok(updated) = serde_json::to_vec(&document) {
        payload = updated;
    }
    payload
}

fn set_json_path(payload: &mut Vec<u8>, path: &str, replacement: Value) {
    if path.trim().is_empty() {
        return;
    }
    let Ok(mut document) = serde_json::from_slice::<Value>(payload.as_slice()) else {
        return;
    };
    if !set_value_at_path(&mut document, path, replacement) {
        return;
    }
    if let Ok(updated) = serde_json::to_vec(&document) {
        *payload = updated;
    }
}

fn set_value_at_path(document: &mut Value, path: &str, replacement: Value) -> bool {
    let segments = path.split('.').collect::<Vec<_>>();
    if segments.iter().any(|segment| segment.is_empty()) {
        return false;
    }
    set_value_at_segments(document, &segments, replacement)
}

fn set_value_at_segments(document: &mut Value, segments: &[&str], replacement: Value) -> bool {
    let Some((head, tail)) = segments.split_first() else {
        *document = replacement;
        return true;
    };
    if let Ok(index) = head.parse::<usize>() {
        if !document.is_array() {
            *document = Value::Array(Vec::new());
        }
        let array = document.as_array_mut().expect("array initialized");
        if index >= array.len() {
            array.resize(index + 1, Value::Null);
        }
        return set_value_at_segments(&mut array[index], tail, replacement);
    }
    if !document.is_object() {
        *document = Value::Object(serde_json::Map::new());
    }
    let object = document.as_object_mut().expect("object initialized");
    if tail.is_empty() {
        object.insert((*head).to_owned(), replacement);
        return true;
    }
    let next_is_index = tail[0].parse::<usize>().is_ok();
    let child = object.entry((*head).to_owned()).or_insert_with(|| {
        if next_is_index {
            Value::Array(Vec::new())
        } else {
            Value::Object(serde_json::Map::new())
        }
    });
    set_value_at_segments(child, tail, replacement)
}

fn value_at_path_mut<'a>(document: &'a mut Value, path: &str) -> Option<&'a mut Value> {
    let mut current = document;
    for segment in path.split('.') {
        current = if let Ok(index) = segment.parse::<usize>() {
            current.as_array_mut()?.get_mut(index)?
        } else {
            current.as_object_mut()?.get_mut(segment)?
        };
    }
    Some(current)
}
