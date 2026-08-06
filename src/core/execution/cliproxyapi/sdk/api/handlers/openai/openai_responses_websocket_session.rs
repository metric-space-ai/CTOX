// ref: sdk/api/handlers/openai/openai_responses_websocket_session.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use serde_json::Value;

#[must_use]
pub fn websocket_upstream_supports_incremental_input(
    attributes: &BTreeMap<String, String>,
    metadata: &BTreeMap<String, Value>,
) -> bool {
    attributes
        .get("responses_websocket_incremental_input")
        .is_some_and(|value| parse_bool(value))
        || metadata
            .get("responses_websocket_incremental_input")
            .and_then(Value::as_bool)
            .unwrap_or(false)
}

#[must_use]
pub fn responses_websocket_resolved_model_name(model: &str) -> String {
    super::super::route_model_base_name(model)
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}
