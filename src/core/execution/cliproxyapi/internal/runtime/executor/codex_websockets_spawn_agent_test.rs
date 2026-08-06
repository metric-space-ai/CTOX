// ref: internal/runtime/executor/codex_websockets_spawn_agent_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: replaced_by_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::client::codex::optimize_multi_agent_v2::{
    optimize_request, MultiAgentV2Context, SpawnAgentModelMetadata,
};

#[test]
fn websocket_path_uses_same_optimizer_without_transport_global() {
    let context = MultiAgentV2Context {
        enabled: true,
        user_agent: "codex-tui/1".to_owned(),
        available_models: Vec::new(),
        catalog_json: Vec::new(),
    };
    let payload = br#"{"tools":[{"type":"namespace","name":"collaboration","tools":[{"type":"function","name":"spawn_agent","description":"Spawns an agent","parameters":{"properties":{"message":{"encrypted":true}}}}]}]}"#;
    let metadata = |_: &str| -> Option<SpawnAgentModelMetadata> { None };
    let result = optimize_request(&context, payload, &metadata);
    let value: serde_json::Value = serde_json::from_slice(&result.payload).unwrap();
    assert!(result.namespace_optimized);
    assert!(value
        .pointer("/tools/0/tools/0/parameters/properties/message/encrypted")
        .is_none());
}
