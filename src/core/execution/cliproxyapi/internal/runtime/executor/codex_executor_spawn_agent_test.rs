// ref: internal/runtime/executor/codex_executor_spawn_agent_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: replaced_by_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::client::codex::optimize_multi_agent_v2::{
    optimize_request, MultiAgentV2Context, SpawnAgentModelMetadata,
    CODEX_OPTIMIZED_COLLABORATION_NAMESPACE,
};

#[test]
fn executor_uses_shared_multi_agent_v2_optimizer() {
    let context = MultiAgentV2Context {
        enabled: true,
        user_agent: "Codex Desktop/1".to_owned(),
        available_models: Vec::new(),
        catalog_json: Vec::new(),
    };
    let payload = br#"{"tools":[{"type":"namespace","name":"collaboration","tools":[{"type":"function","name":"spawn_agent","description":"Spawns an agent","parameters":{"properties":{"message":{"encrypted":true}}}}]}]}"#;
    let metadata = |_: &str| -> Option<SpawnAgentModelMetadata> { None };
    let result = optimize_request(&context, payload, &metadata);
    assert!(result.namespace_optimized);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&result.payload).unwrap()["tools"][0]["name"],
        CODEX_OPTIMIZED_COLLABORATION_NAMESPACE
    );
}
