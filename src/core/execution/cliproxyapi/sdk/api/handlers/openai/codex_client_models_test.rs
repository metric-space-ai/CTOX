// ref: sdk/api/handlers/openai/codex_client_models_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;

#[test]
fn multi_agent_v2_follows_typed_runtime_config() {
    let models = vec![serde_json::Map::from_iter([(
        "slug".to_owned(),
        serde_json::json!("gpt-5-codex"),
    )])];
    let disabled = codex_client_models_response_with_multi_agent_v2(&models, false);
    assert!(disabled["models"][0]["multi_agent_version"].is_null());
    let enabled = codex_client_models_response_with_multi_agent_v2(&models, true);
    assert_eq!(enabled["models"][0]["multi_agent_version"], "v2");
    assert!(models[0].get("multi_agent_version").is_none());
}
