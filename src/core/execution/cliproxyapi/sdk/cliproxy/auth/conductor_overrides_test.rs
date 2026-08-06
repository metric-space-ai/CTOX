// ref: sdk/cliproxy/auth/conductor_overrides_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;
use serde_json::json;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Store(Mutex<Vec<CooldownStateRecord>>);
impl CooldownStateStore for Store {
    fn load(&self) -> Result<Vec<CooldownStateRecord>, CooldownStoreError> {
        Ok(self.0.lock().unwrap().clone())
    }
    fn save(&self, records: &[CooldownStateRecord]) -> Result<(), CooldownStoreError> {
        *self.0.lock().unwrap() = records.to_vec();
        Ok(())
    }
}

#[test]
fn typed_auth_overrides_are_bounded_and_do_not_use_environment() {
    let mut auth = Auth::default();
    auth.metadata.insert("request_retry".into(), json!(-5));
    auth.metadata.insert("disable_cooling".into(), json!(true));
    assert_eq!(auth.request_retry_override(), Some(0));
    assert_eq!(auth.disable_cooling_override(), Some(true));
    auth.metadata.insert("request_retry".into(), json!(3));
    assert_eq!(auth.request_retry_override(), Some(3));
}

#[test]
fn request_scoped_bad_request_preserves_existing_cooldown() {
    let store = Arc::new(Store::default());
    let conductor = CooldownConductor::new(store.clone());
    let result = |status, at| AccountExecutionResult {
        provider: "claude".into(),
        auth_id: "a".into(),
        model: Some("m".into()),
        status,
        retry_delay_ms: Some(1_000),
        observed_at_ms: at,
    };
    assert!(conductor.record(result(429, 10)).unwrap());
    assert!(!conductor.record(result(400, 20)).unwrap());
    assert_eq!(store.0.lock().unwrap()[0].next_retry_after_ms, Some(1_010));
}

#[test]
fn transient_and_auth_failures_remain_distinct() {
    let store = Arc::new(Store::default());
    let conductor = CooldownConductor::new(store.clone());
    let result = |status| AccountExecutionResult {
        provider: "claude".into(),
        auth_id: "a".into(),
        model: None,
        status,
        retry_delay_ms: None,
        observed_at_ms: 1_000,
    };
    conductor.record(result(503)).unwrap();
    assert_eq!(
        store.0.lock().unwrap()[0].reason,
        "transient_upstream_error"
    );
    conductor.record(result(401)).unwrap();
    assert_eq!(store.0.lock().unwrap()[0].reason, "unauthorized");
}
