// Origin: CTOX
// License: AGPL-3.0-only

//! Shared JSON mutation primitives for the file-based thinking port.

use serde_json::{Map, Value};

pub(super) fn get_path<'a>(document: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.')
        .try_fold(document, |value, segment| value.as_object()?.get(segment))
}

pub(super) fn set_path(document: &mut Value, path: &str, value: Value) {
    fn recurse(current: &mut Value, segments: &[&str], value: Value) -> bool {
        if current.is_array() {
            // tidwall/sjson refuses a non-numeric object path into an array and
            // returns the original JSON; every thinking path is non-numeric.
            return false;
        }
        let Some((head, tail)) = segments.split_first() else {
            return false;
        };
        if tail.is_empty() {
            ensure_object(current).insert((*head).to_owned(), value);
            return true;
        }
        let child = ensure_object(current)
            .entry((*head).to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        recurse(child, tail, value)
    }
    let _ = recurse(document, &path.split('.').collect::<Vec<_>>(), value);
}

pub(super) fn remove_path(document: &mut Value, path: &str) {
    fn recurse(current: &mut Value, segments: &[&str]) {
        let Some((head, tail)) = segments.split_first() else {
            return;
        };
        let Some(object) = current.as_object_mut() else {
            return;
        };
        if tail.is_empty() {
            object.remove(*head);
        } else if let Some(child) = object.get_mut(*head) {
            recurse(child, tail);
        }
    }
    recurse(document, &path.split('.').collect::<Vec<_>>());
}

pub(super) fn remove_empty_object(document: &mut Value, path: &str) {
    if get_path(document, path)
        .is_some_and(|value| value.as_object().is_some_and(serde_json::Map::is_empty))
    {
        remove_path(document, path);
    }
}

pub(super) fn serialize_if_changed(
    original_bytes: &[u8],
    original: &Value,
    document: &Value,
) -> Vec<u8> {
    if document == original {
        original_bytes.to_vec()
    } else {
        serde_json::to_vec(document).unwrap_or_else(|_| original_bytes.to_vec())
    }
}

fn ensure_object(value: &mut Value) -> &mut Map<String, Value> {
    if !value.is_object() {
        *value = Value::Object(Map::new());
    }
    value.as_object_mut().expect("object inserted above")
}
