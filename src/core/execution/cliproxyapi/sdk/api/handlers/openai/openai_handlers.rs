// ref: sdk/api/handlers/openai/openai_handlers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};

#[must_use]
pub fn should_treat_as_responses_format(raw_json: &[u8]) -> bool {
    let Ok(Value::Object(document)) = serde_json::from_slice(raw_json) else {
        return false;
    };
    document.contains_key("input")
        || document.contains_key("instructions")
        || document.contains_key("previous_response_id")
}

#[must_use]
pub fn convert_completions_request_to_chat_completions(raw_json: &[u8]) -> Vec<u8> {
    let Ok(Value::Object(mut document)) = serde_json::from_slice(raw_json) else {
        return raw_json.to_vec();
    };
    let prompt = document
        .remove("prompt")
        .unwrap_or(Value::String(String::new()));
    document.insert(
        "messages".to_owned(),
        json!([{"role":"user","content": prompt}]),
    );
    serde_json::to_vec(&document).unwrap_or_else(|_| raw_json.to_vec())
}

#[must_use]
pub fn convert_chat_completions_response_to_completions(raw_json: &[u8]) -> Vec<u8> {
    let Ok(Value::Object(mut document)) = serde_json::from_slice(raw_json) else {
        return raw_json.to_vec();
    };
    let choices = document
        .get_mut("choices")
        .and_then(Value::as_array_mut)
        .map(|choices| {
            for choice in choices {
                if let Value::Object(choice) = choice {
                    let text = choice
                        .remove("message")
                        .and_then(|message| message.get("content").cloned())
                        .unwrap_or(Value::Null);
                    choice.insert("text".to_owned(), text);
                }
            }
        });
    let _ = choices;
    document.insert(
        "object".to_owned(),
        Value::String("text_completion".to_owned()),
    );
    serde_json::to_vec(&document).unwrap_or_else(|_| raw_json.to_vec())
}
