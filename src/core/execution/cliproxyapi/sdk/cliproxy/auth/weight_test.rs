// ref: sdk/cliproxy/auth/weight_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};

use super::validate_auth_weight;
use crate::internal::config::CliproxyRuntimeConfig;
use crate::internal::credentialweight::{CredentialWeightError, MAX_CREDENTIAL_WEIGHT};

fn claude_account_with(extra: Value) -> Value {
    let mut account = json!({
        "id": "claude-a",
        "access_token_secret": {"scope": "subscriptions", "name": "claude-a-access"},
        "refresh_token_secret": {"scope": "subscriptions", "name": "claude-a-refresh"}
    });
    let object = account
        .as_object_mut()
        .expect("account fixture is an object");
    for (key, value) in extra.as_object().expect("extra fixture is an object") {
        object.insert(key.clone(), value.clone());
    }
    json!({
        "request_timeout_ms": 30_000,
        "routing_strategy": "weighted-round-robin",
        "claude_accounts": [account]
    })
}

#[test]
fn typed_weight_reaches_candidate_and_uses_the_shared_bound() {
    let config: CliproxyRuntimeConfig =
        serde_json::from_value(claude_account_with(json!({"weight": 7}))).unwrap();
    let validated = config.validate().unwrap();
    let candidate = &validated.claude_candidates()[0];

    assert_eq!(candidate.weight, 7);
    assert_eq!(validate_auth_weight(candidate.weight), Ok(()));
    assert_eq!(validate_auth_weight(-2), Ok(()));
    assert_eq!(
        validate_auth_weight(MAX_CREDENTIAL_WEIGHT + 1),
        Err(CredentialWeightError::AboveMaximum)
    );
}

#[test]
fn dynamic_attribute_and_metadata_weight_overrides_are_rejected() {
    for forbidden in [
        json!({"attributes": {"weight": "7"}}),
        json!({"metadata": {"weight": 7}}),
    ] {
        let error = serde_json::from_value::<CliproxyRuntimeConfig>(claude_account_with(forbidden))
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("unknown field"));
        assert!(message.contains("attributes") || message.contains("metadata"));
    }
}
