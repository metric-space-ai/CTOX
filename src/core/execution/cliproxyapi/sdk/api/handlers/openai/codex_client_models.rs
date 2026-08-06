// ref: sdk/api/handlers/openai/codex_client_models.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{Map, Value};

#[must_use]
pub fn codex_client_models_response(models: &[Map<String, Value>]) -> Value {
    codex_client_models_response_with_multi_agent_v2(models, false)
}

#[must_use]
pub fn codex_client_models_response_with_multi_agent_v2(
    models: &[Map<String, Value>],
    enabled: bool,
) -> Value {
    let models = models
        .iter()
        .cloned()
        .map(|mut model| {
            if enabled {
                model.insert(
                    "multi_agent_version".to_owned(),
                    Value::String("v2".to_owned()),
                );
            } else {
                model
                    .entry("multi_agent_version".to_owned())
                    .or_insert(Value::Null);
            }
            model
        })
        .collect::<Vec<_>>();
    serde_json::json!({"models": models})
}

#[cfg(test)]
#[path = "codex_client_models_test.rs"]
mod codex_client_models_test;
