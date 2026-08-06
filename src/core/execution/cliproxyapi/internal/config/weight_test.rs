// ref: internal/config/weight_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};

use super::config_types::{CliproxyRuntimeConfig, RuntimeConfigError};
use super::weight::{validate_credential_weight, MAX_CREDENTIAL_WEIGHT};

fn runtime_with_claude_weight(weight: Value) -> Value {
    json!({
        "request_timeout_ms": 30_000,
        "routing_strategy": "weighted-round-robin",
        "claude_accounts": [{
            "id": "claude-a",
            "weight": weight,
            "access_token_secret": {"scope": "subscriptions", "name": "claude-a-access"},
            "refresh_token_secret": {"scope": "subscriptions", "name": "claude-a-refresh"}
        }]
    })
}

#[test]
fn typed_runtime_rejects_non_integer_overflow_and_above_maximum_weights() {
    assert!(
        serde_json::from_value::<CliproxyRuntimeConfig>(runtime_with_claude_weight(json!(1.5)))
            .is_err()
    );
    assert!(serde_json::from_str::<CliproxyRuntimeConfig>(
        r#"{"request_timeout_ms":30000,"routing_strategy":"weighted-round-robin","claude_accounts":[{"id":"claude-a","weight":9223372036854775808,"access_token_secret":{"scope":"subscriptions","name":"a"},"refresh_token_secret":{"scope":"subscriptions","name":"b"}}]}"#
    )
    .is_err());

    let config: CliproxyRuntimeConfig =
        serde_json::from_value(runtime_with_claude_weight(json!(MAX_CREDENTIAL_WEIGHT + 1)))
            .unwrap();
    assert_eq!(
        config.validate(),
        Err(RuntimeConfigError::InvalidCredentialWeight)
    );
}

#[test]
fn typed_runtime_accepts_exclusion_maximum_and_preserves_explicit_zero() {
    for weight in [-1, MAX_CREDENTIAL_WEIGHT] {
        let config: CliproxyRuntimeConfig =
            serde_json::from_value(runtime_with_claude_weight(json!(weight))).unwrap();
        assert!(config.validate().is_ok());
    }

    let config: CliproxyRuntimeConfig =
        serde_json::from_value(runtime_with_claude_weight(json!(0))).unwrap();
    let encoded = serde_json::to_value(&config).unwrap();
    assert_eq!(encoded["claude_accounts"][0]["weight"], json!(0));
    let validated = config.validate().unwrap();
    assert_eq!(validated.claude_candidates()[0].weight, 0);
}

#[test]
fn nil_and_all_active_account_families_use_the_same_bound() {
    assert_eq!(validate_credential_weight(None), Ok(()));

    let mut config = runtime_with_claude_weight(json!(1));
    config["claude_accounts"] = json!([]);
    config["codex_accounts"] = json!([{
        "id": "codex-a",
        "weight": MAX_CREDENTIAL_WEIGHT + 1,
        "id_token_secret": {"scope": "subscriptions", "name": "codex-id"},
        "access_token_secret": {"scope": "subscriptions", "name": "codex-access"},
        "refresh_token_secret": {"scope": "subscriptions", "name": "codex-refresh"}
    }]);
    let config: CliproxyRuntimeConfig = serde_json::from_value(config).unwrap();
    assert_eq!(
        config.validate(),
        Err(RuntimeConfigError::InvalidCredentialWeight)
    );

    let config: CliproxyRuntimeConfig = serde_json::from_value(json!({
        "request_timeout_ms": 30_000,
        "routing_strategy": "weighted-round-robin",
        "antigravity_accounts": [{
            "id": "antigravity-a",
            "weight": MAX_CREDENTIAL_WEIGHT + 1,
            "access_token_secret": {"scope": "subscriptions", "name": "ag-access"},
            "refresh_token_secret": {"scope": "subscriptions", "name": "ag-refresh"},
            "state_secret": {"scope": "subscriptions", "name": "ag-state"}
        }]
    }))
    .unwrap();
    assert_eq!(
        config.validate(),
        Err(RuntimeConfigError::InvalidCredentialWeight)
    );
}
