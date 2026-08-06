// ref: sdk/cliproxy/auth/cooldown_state_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: portable persisted state, redacted diagnostics and serialized store transitions
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use super::cooldown_backoff_test::{result, MemoryCooldownStore};
use super::{CooldownConductor, CooldownErrorState, CooldownStateRecord, CooldownStateStore};

#[test]
fn state_round_trip_omits_paths_and_redacts_provider_message() {
    let mut record = CooldownStateRecord {
        provider: "codex".into(),
        auth_id: "auth".into(),
        model: Some("gpt".into()),
        status: "cooling".into(),
        next_retry_after_ms: Some(2_000),
        reason: "quota".into(),
        quota: Default::default(),
        last_error: None,
        updated_at_ms: 1_000,
    };
    record.last_error = Some(CooldownErrorState {
        code: "rate_limit".into(),
        message: "secret provider body".into(),
        retryable: true,
        http_status: Some(429),
    });
    let json = serde_json::to_string(&record).unwrap();
    assert!(!json.contains("auth_file"));
    assert!(!format!("{record:?}").contains("secret provider body"));
    assert_eq!(
        serde_json::from_str::<CooldownStateRecord>(&json).unwrap(),
        record
    );
}

#[test]
fn concurrent_transitions_leave_one_valid_latest_scope_record() {
    let store = Arc::new(MemoryCooldownStore::default());
    let conductor = Arc::new(CooldownConductor::new(store.clone()));
    let handles = (0..16)
        .map(|index| {
            let conductor = conductor.clone();
            std::thread::spawn(move || conductor.record(result(503, 1_000 + index)).unwrap())
        })
        .collect::<Vec<_>>();
    for handle in handles {
        handle.join().unwrap();
    }
    let records = CooldownStateStore::load(&*store).unwrap();
    assert_eq!(records.len(), 1);
    records[0].validate().unwrap();
}

#[test]
fn account_reset_removes_account_and_model_scopes() {
    let store = Arc::new(MemoryCooldownStore::default());
    let conductor = CooldownConductor::new(store.clone());
    conductor.record(result(429, 1_000)).unwrap();
    let models = conductor.reset_account("auth").unwrap();
    assert_eq!(models, ["gpt"]);
    assert!(store.0.lock().unwrap().is_empty());
}
