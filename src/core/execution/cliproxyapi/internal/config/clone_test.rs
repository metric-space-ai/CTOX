// ref: internal/config/clone_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::CliproxyRuntimeConfig;

#[test]
fn clone_for_runtime_deep_copies_reference_fields() {
    let mut original: CliproxyRuntimeConfig = serde_json::from_value(serde_json::json!({
        "request_timeout_ms": 30000,
        "claude_accounts": [{
            "id": "claude-a",
            "models": ["claude-sonnet"],
            "access_token_secret": {"scope": "provider", "name": "access"},
            "refresh_token_secret": {"scope": "provider", "name": "refresh"}
        }]
    }))
    .unwrap();
    let snapshot = original.clone_for_runtime();

    original.claude_accounts[0].id = "mutated".into();
    original.claude_accounts[0].models[0] = "mutated-model".into();
    original.claude_accounts[0].access_token_secret.name = "mutated-secret-ref".into();

    assert_eq!(snapshot.claude_accounts[0].id, "claude-a");
    assert_eq!(snapshot.claude_accounts[0].models, ["claude-sonnet"]);
    assert_eq!(
        snapshot.claude_accounts[0].access_token_secret.name,
        "access"
    );
}
