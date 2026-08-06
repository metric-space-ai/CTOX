// ref: internal/translator/antigravity/openai/chat-completions/antigravity_openai_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};

use crate::internal::translator::gemini::openai::chat_completions::{
    convert_gemini_response_to_openai_chat_non_stream,
    convert_gemini_response_to_openai_chat_stream, GeminiToChatStreamState,
};

use super::antigravity_openai_request::reverse_disambiguated_names;

#[derive(Default)]
pub struct AntigravityToChatStreamState {
    gemini: GeminiToChatStreamState,
}

pub fn convert_antigravity_response_to_openai_chat_stream(
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
    state: &mut AntigravityToChatStreamState,
) -> Vec<Vec<u8>> {
    if raw == b"[DONE]" {
        return Vec::new();
    }
    let Ok(root) = serde_json::from_slice::<Value>(raw) else {
        return Vec::new();
    };
    let Some(mut response) = root.get("response").cloned() else {
        return Vec::new();
    };
    restore_names(&mut response, original_request);
    if let Some(object) = response.as_object_mut() {
        match object.get_mut("candidates").and_then(Value::as_array_mut) {
            Some(candidates) if !candidates.is_empty() => candidates.truncate(1),
            _ => {
                object.insert(
                    "candidates".into(),
                    json!([{"index":0,"content":{"parts":[]}}]),
                );
            }
        }
    }
    convert_gemini_response_to_openai_chat_stream(
        model_name,
        original_request,
        request,
        &serde_json::to_vec(&response).unwrap_or_default(),
        &mut state.gemini,
    )
}

pub fn convert_antigravity_response_to_openai_chat_non_stream(
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
) -> Vec<u8> {
    let Ok(root) = serde_json::from_slice::<Value>(raw) else {
        return Vec::new();
    };
    let Some(mut response) = root.get("response").cloned() else {
        return Vec::new();
    };
    restore_names(&mut response, original_request);
    convert_gemini_response_to_openai_chat_non_stream(
        original_request,
        request,
        &serde_json::to_vec(&response).unwrap_or_default(),
    )
}

fn restore_names(response: &mut Value, original_request: &[u8]) {
    let names = reverse_disambiguated_names(original_request);
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
        for field in ["functionCall", "functionResponse"] {
            let Some(function) = part.get_mut(field) else {
                continue;
            };
            let native = function.get("name").and_then(Value::as_str).unwrap_or("");
            if let Some(original) = names.get(native) {
                function["name"] = Value::String(original.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        convert_antigravity_response_to_openai_chat_non_stream,
        convert_antigravity_response_to_openai_chat_stream, AntigravityToChatStreamState,
    };
    use serde_json::Value;

    #[test]
    fn non_stream_unwraps_response_and_restores_colliding_name() {
        let request = br#"{"tools":[{"type":"function","function":{"name":"read file"}},{"type":"function","function":{"name":"read/file"}}]}"#;
        let raw = br#"{"response":{"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"read_file_1230182ae9f9","args":{}}}]}}]}}"#;
        let output: Value = serde_json::from_slice(
            &convert_antigravity_response_to_openai_chat_non_stream(request, b"", raw),
        )
        .unwrap();
        assert_eq!(
            output["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read/file"
        );
    }

    #[test]
    fn stream_uses_only_first_candidate_and_emits_an_empty_shell() {
        let mut state = AntigravityToChatStreamState::default();
        let output = convert_antigravity_response_to_openai_chat_stream(
            "fallback",
            b"{}",
            b"",
            br#"{"response":{"candidates":[{"index":0,"content":{"parts":[{"text":"one"}]}},{"index":1,"content":{"parts":[{"text":"two"}]}}]}}"#,
            &mut state,
        );
        let output: Value = serde_json::from_slice(&output[0]).unwrap();
        assert_eq!(output["choices"][0]["delta"]["content"], "one");

        let output = convert_antigravity_response_to_openai_chat_stream(
            "fallback",
            b"{}",
            b"",
            br#"{"response":{}}"#,
            &mut state,
        );
        assert_eq!(output.len(), 1);
    }
}
