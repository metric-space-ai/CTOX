// ref: sdk/cliproxy/auth/home_concurrency_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: tuple validation, identity and registry installation behavior
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::Duration;

use crate::sdk::cliproxy::executionregistry::{Registry, ScopeSpec};

use super::{
    canonical_home_concurrency_model_key, decode_home_concurrency, decode_home_dispatch_error,
    install_home_concurrency_scope, verify_home_concurrency_identity, HomeConcurrencyBusyError,
    HomeConcurrencyError,
};

#[test]
fn canonical_model_only_removes_recognized_reasoning_suffixes() {
    assert_eq!(
        canonical_home_concurrency_model_key(" GPT-5 (HIGH) "),
        "gpt-5"
    );
    assert_eq!(
        canonical_home_concurrency_model_key("gpt-5(custom)"),
        "gpt-5(custom)"
    );
    assert_eq!(canonical_home_concurrency_model_key("(high)"), "(high)");
}

#[test]
fn accounted_tuple_installs_authoritative_identity() {
    let registry = Registry::new();
    let pending = registry.begin_dispatch().unwrap();
    let raw = br#"{"concurrency":{"accounted":true,"credential_id":"auth-1","model":"gpt-5"}}"#;
    let tuple = decode_home_concurrency(raw).unwrap().unwrap();
    verify_home_concurrency_identity(Some(&tuple), "auth-1", "auth-1").unwrap();
    let scope = install_home_concurrency_scope(
        &registry,
        &pending,
        Some(&tuple),
        ScopeSpec {
            credential_id: "untrusted".into(),
            model: "alias".into(),
            ..ScopeSpec::default()
        },
    )
    .unwrap();
    assert_eq!(scope.spec().credential_id, "auth-1");
    assert_eq!(scope.spec().model, "gpt-5");
    assert!(scope.spec().accounted);
    scope.end("done");
}

#[test]
fn malformed_or_mismatched_concurrency_fails_closed() {
    let malformed =
        br#"{"concurrency":{"accounted":false,"credential_id":"auth-1","model":"gpt-5"}}"#;
    assert_eq!(
        decode_home_concurrency(malformed),
        Err(HomeConcurrencyError::MalformedTuple)
    );
    let tuple = decode_home_concurrency(
        br#"{"concurrency":{"accounted":true,"credential_id":"auth-1","model":"gpt-5"}}"#,
    )
    .unwrap();
    assert_eq!(
        verify_home_concurrency_identity(tuple.as_ref(), "auth-2", "auth-2"),
        Err(HomeConcurrencyError::IdentityMismatch)
    );
}

#[test]
fn busy_response_exposes_only_rounded_retry_after() {
    let error = HomeConcurrencyBusyError::new(" busy ", Duration::from_millis(1_001));
    assert_eq!(error.status_code(), 429);
    assert_eq!(
        error.safe_response_headers(),
        vec![("Retry-After".into(), "2".into())]
    );
}

#[test]
fn no_candidate_errors_map_to_service_unavailable() {
    for code in ["auth_not_found", "auth_unavailable"] {
        let error = decode_home_dispatch_error(
            format!(r#"{{"error":{{"type":"{code}","message":"no auth available"}}}}"#).as_bytes(),
        )
        .unwrap();
        assert_eq!(error.code, code);
        assert_eq!(error.status_code, 503);
    }
}
