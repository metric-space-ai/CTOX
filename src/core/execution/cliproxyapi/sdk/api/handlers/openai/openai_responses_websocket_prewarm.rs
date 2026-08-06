// ref: sdk/api/handlers/openai/openai_responses_websocket_prewarm.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};

pub fn merge_json_array_raw(existing: &[u8], append: &[u8]) -> Result<Vec<u8>, serde_json::Error> {
    let mut existing: Vec<Value> = serde_json::from_slice(existing)?;
    existing.extend(serde_json::from_slice::<Vec<Value>>(append)?);
    serde_json::to_vec(&existing)
}

#[must_use]
pub fn normalize_json_array_raw(raw: &[u8]) -> Vec<u8> {
    serde_json::from_slice::<Vec<Value>>(raw)
        .ok()
        .and_then(|value| serde_json::to_vec(&value).ok())
        .unwrap_or_else(|| b"[]".to_vec())
}

pub fn synthetic_responses_websocket_prewarm_payloads(
    request_json: &[u8],
) -> Result<Vec<Vec<u8>>, serde_json::Error> {
    let request: Value = serde_json::from_slice(request_json)?;
    let id = request
        .get("request_id")
        .and_then(Value::as_str)
        .unwrap_or("prewarm");
    Ok(vec![
        serde_json::to_vec(
            &json!({"type":"response.created","response":{"id":id,"status":"in_progress"}}),
        )?,
        serde_json::to_vec(
            &json!({"type":"response.completed","response":{"id":id,"status":"completed","output":[]}}),
        )?,
    ])
}
