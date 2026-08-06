// ref: internal/translator/gemini/common/safety.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Map, Value};

pub fn default_safety_settings() -> Value {
    json!([
        {"category":"HARM_CATEGORY_HARASSMENT","threshold":"OFF"},
        {"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"OFF"},
        {"category":"HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold":"OFF"},
        {"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"OFF"},
        {"category":"HARM_CATEGORY_CIVIC_INTEGRITY","threshold":"BLOCK_NONE"}
    ])
}

pub fn attach_default_safety_settings(raw: &[u8], path: &str) -> Vec<u8> {
    let Ok(mut root) = serde_json::from_slice::<Value>(raw) else {
        return raw.to_vec();
    };
    let segments = path
        .split('.')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    if segments.is_empty() || path_exists(&root, &segments) {
        return raw.to_vec();
    }
    let mut cursor = &mut root;
    for segment in &segments[..segments.len() - 1] {
        if !cursor.is_object() {
            *cursor = Value::Object(Map::new());
        }
        cursor = cursor
            .as_object_mut()
            .expect("object was initialized")
            .entry((*segment).to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
    }
    if !cursor.is_object() {
        *cursor = Value::Object(Map::new());
    }
    cursor
        .as_object_mut()
        .expect("object was initialized")
        .insert(
            segments.last().expect("non-empty path").to_string(),
            default_safety_settings(),
        );
    serde_json::to_vec(&root).unwrap_or_else(|_| raw.to_vec())
}

fn path_exists(root: &Value, segments: &[&str]) -> bool {
    let mut cursor = root;
    for segment in segments {
        let Some(next) = cursor.get(*segment) else {
            return false;
        };
        cursor = next;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::attach_default_safety_settings;
    use serde_json::Value;

    #[test]
    fn attaches_at_nested_path_and_preserves_existing_or_invalid_input() {
        let output: Value = serde_json::from_slice(&attach_default_safety_settings(
            br#"{"request":{}}"#,
            "request.safetySettings",
        ))
        .unwrap();
        assert_eq!(
            output["request"]["safetySettings"]
                .as_array()
                .unwrap()
                .len(),
            5
        );
        let existing = br#"{"safetySettings":null}"#;
        assert_eq!(
            attach_default_safety_settings(existing, "safetySettings"),
            existing
        );
        assert_eq!(
            attach_default_safety_settings(b"not json", "safetySettings"),
            b"not json"
        );
    }
}
