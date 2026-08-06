// ref: internal/translator/antigravity/gemini/antigravity_gemini_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::internal::translator::common::gemini_token_count_json;
use crate::internal::util::{disambiguated_tool_name_map, restore_sanitized_tool_name};

pub fn convert_antigravity_response_to_gemini(
    original_request: &[u8],
    raw_json: &[u8],
    alt: Option<&str>,
) -> Vec<Vec<u8>> {
    let raw = raw_json
        .strip_prefix(b"data:")
        .map(trim_like_go_bytes)
        .unwrap_or(raw_json);
    let Some(alt) = alt else {
        return Vec::new();
    };
    if !alt.is_empty() {
        return vec![b"[]".to_vec()];
    }
    let Ok(root) = serde_json::from_slice::<Value>(raw) else {
        return vec![Vec::new()];
    };
    let Some(response) = root.get("response") else {
        return vec![Vec::new()];
    };
    let mut response = response.clone();
    restore_usage_metadata(&mut response);
    restore_function_names(&mut response, original_request);
    vec![serde_json::to_vec(&response).unwrap_or_default()]
}

pub fn convert_antigravity_response_to_gemini_non_stream(
    original_request: &[u8],
    raw_json: &[u8],
) -> Vec<u8> {
    let Ok(root) = serde_json::from_slice::<Value>(raw_json) else {
        return raw_json.to_vec();
    };
    let mut response = root.get("response").cloned().unwrap_or(root);
    restore_usage_metadata(&mut response);
    restore_function_names(&mut response, original_request);
    serde_json::to_vec(&response).unwrap_or_else(|_| raw_json.to_vec())
}

pub fn gemini_token_count(count: i64) -> Vec<u8> {
    gemini_token_count_json(count)
}

fn restore_usage_metadata(response: &mut Value) {
    let Some(object) = response.as_object_mut() else {
        return;
    };
    if let Some(usage) = object.remove("cpaUsageMetadata") {
        object.insert("usageMetadata".to_owned(), usage);
    }
}

fn restore_function_names(response: &mut Value, original_request: &[u8]) {
    let names = disambiguated_tool_name_map(original_request);
    if names.is_empty() {
        return;
    }
    for part in response
        .get_mut("candidates")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|candidate| candidate.pointer_mut("/content/parts"))
        .filter_map(Value::as_array_mut)
        .flatten()
    {
        for field in [
            "functionCall",
            "functionResponse",
            "function_call",
            "function_response",
        ] {
            if let Some(value) = part.get_mut(field).and_then(Value::as_object_mut) {
                if let Some(name) = value.get("name").and_then(Value::as_str) {
                    value.insert(
                        "name".to_owned(),
                        Value::String(restore_sanitized_tool_name(&names, name)),
                    );
                }
            }
        }
    }
}

fn trim_like_go_bytes(bytes: &[u8]) -> &[u8] {
    match std::str::from_utf8(bytes) {
        Ok(text) => text.trim().as_bytes(),
        Err(_) => bytes.trim_ascii(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwraps_restores_usage_and_colliding_function_name() {
        let request =
            br#"{"tools":[{"functionDeclarations":[{"name":"read file"},{"name":"read/file"}]}]}"#;
        let map = crate::internal::util::sanitized_function_name_map(request);
        let mapped = &map["read/file"];
        let raw = serde_json::to_vec(&serde_json::json!({"response":{"cpaUsageMetadata":{"promptTokenCount":2},"candidates":[{"content":{"parts":[{"functionCall":{"name":mapped}}]}}]}})).unwrap();
        let output: Value = serde_json::from_slice(
            &convert_antigravity_response_to_gemini_non_stream(request, &raw),
        )
        .unwrap();
        assert_eq!(output["usageMetadata"]["promptTokenCount"], 2);
        assert_eq!(
            output["candidates"][0]["content"]["parts"][0]["functionCall"]["name"],
            "read/file"
        );
    }
}
