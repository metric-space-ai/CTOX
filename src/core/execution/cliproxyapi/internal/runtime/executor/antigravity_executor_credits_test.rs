// ref: internal/runtime/executor/antigravity_executor_credits_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::antigravity_executor_credits::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

#[test]
fn classifies_structured_429_decisions_and_human_retry_delay() {
    let body = br#"{"error":{"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"12s"},{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"RATE_LIMIT_EXCEEDED"}]}}"#;
    let decision = decide_antigravity_429(body);
    assert_eq!(
        decision.kind,
        Antigravity429DecisionKind::ShortCooldownSwitchAuth
    );
    assert_eq!(decision.retry_after, Some(Duration::from_secs(12)));
    assert_eq!(
        classify_antigravity_429(body),
        Antigravity429Category::RateLimited
    );
    assert_eq!(
        classify_antigravity_429(
            br#"{"error":{"status":"RESOURCE_EXHAUSTED","message":"quota exhausted"}}"#
        ),
        Antigravity429Category::QuotaExhausted
    );
}

#[test]
fn credit_types_are_injected_only_into_valid_objects() {
    let out: Value =
        serde_json::from_slice(&inject_enabled_credit_types(br#"{"request":{}}"#).unwrap())
            .unwrap();
    assert_eq!(out["enabledCreditTypes"], json!(["GOOGLE_ONE_AI"]));
    assert!(inject_enabled_credit_types(b"bad").is_none());
    assert!(inject_enabled_credit_types(b"[]").is_none());
}

#[test]
fn injected_store_owns_balance_cooldown_and_cross_process_refresh_lease() {
    let store = Arc::new(MemoryAntigravityCreditsStore::default());
    let controller = AntigravityCreditsController::new(store, Duration::from_secs(30));
    assert!(controller.has_credits("a").unwrap());
    controller
        .store_balance(
            "a",
            &AntigravityCreditsBalance {
                credit_amount: 1.0,
                min_credit_amount: 2.0,
                paid_tier_id: "tier".into(),
                known: true,
            },
        )
        .unwrap();
    assert!(!controller.has_credits("a").unwrap());
    controller
        .mark_short_cooldown("a", "m", 1_000, Duration::from_secs(5))
        .unwrap();
    assert_eq!(
        controller.cooldown_remaining("a", "m", 2_000).unwrap(),
        Some(Duration::from_secs(4))
    );
    assert!(controller.acquire_refresh_lease("a", 1_000).unwrap());
    assert!(!controller.acquire_refresh_lease("a", 2_000).unwrap());
}

#[test]
fn retry_classifiers_and_metadata_numbers_match_upstream() {
    assert!(antigravity_should_retry_no_capacity(
        503,
        b"No capacity available"
    ));
    assert!(antigravity_should_retry_transient_429(
        429,
        br#"{"error":{"status":"RESOURCE_EXHAUSTED"}}"#
    ));
    let metadata = BTreeMap::from([
        ("string".into(), Value::String(" 1.5 ".into())),
        ("number".into(), json!(2)),
    ]);
    assert_eq!(parse_meta_float(&metadata, "string"), Some(1.5));
    assert_eq!(parse_meta_float(&metadata, "number"), Some(2.0));
}
