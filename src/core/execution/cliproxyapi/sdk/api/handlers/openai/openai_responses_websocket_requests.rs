// ref: sdk/api/handlers/openai/openai_responses_websocket_requests.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeSet;

use serde_json::Value;

pub fn normalize_responses_websocket_passthrough_request(
    raw_json: &[u8],
    model: &str,
) -> Result<Vec<u8>, String> {
    let Value::Object(mut request) =
        serde_json::from_slice(raw_json).map_err(|_| "invalid JSON body")?
    else {
        return Err("request body must be an object".to_owned());
    };
    if model.trim().is_empty() {
        return Err("model is required".to_owned());
    }
    request.insert("model".to_owned(), Value::String(model.trim().to_owned()));
    request.insert("stream".to_owned(), Value::Bool(true));
    serde_json::to_vec(&request).map_err(|error| error.to_string())
}

#[must_use]
pub fn dedupe_responses_websocket_input_items_by_id(payload: &[u8]) -> Vec<u8> {
    let Ok(Value::Object(mut document)) = serde_json::from_slice(payload) else {
        return payload.to_vec();
    };
    let Some(Value::Array(input)) = document.get_mut("input") else {
        return payload.to_vec();
    };
    let mut seen = BTreeSet::new();
    input.retain(|item| {
        item.get("id")
            .and_then(Value::as_str)
            .is_none_or(|id| seen.insert(id.to_owned()))
    });
    serde_json::to_vec(&document).unwrap_or_else(|_| payload.to_vec())
}
