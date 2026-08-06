// ref: sdk/cliproxy/auth/cooldown_backoff_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: deterministic quota windows and finite recoverable cooldowns through injected store
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use super::{
    AccountExecutionResult, CooldownConductor, CooldownStateRecord, CooldownStateStore,
    CooldownStoreError,
};

#[derive(Default)]
pub(super) struct MemoryCooldownStore(pub(super) Mutex<Vec<CooldownStateRecord>>);
impl CooldownStateStore for MemoryCooldownStore {
    fn load(&self) -> Result<Vec<CooldownStateRecord>, CooldownStoreError> {
        Ok(self.0.lock().unwrap().clone())
    }
    fn save(&self, records: &[CooldownStateRecord]) -> Result<(), CooldownStoreError> {
        *self.0.lock().unwrap() = records.to_vec();
        Ok(())
    }
}

pub(super) fn result(status: u16, observed_at_ms: i64) -> AccountExecutionResult {
    AccountExecutionResult {
        provider: "codex".into(),
        auth_id: "auth".into(),
        model: Some("gpt".into()),
        status,
        retry_delay_ms: None,
        observed_at_ms,
    }
}

#[test]
fn quota_backoff_escalates_only_after_active_window_expires() {
    let store = Arc::new(MemoryCooldownStore::default());
    let conductor = CooldownConductor::new(store.clone());
    conductor.record(result(429, 1_000)).unwrap();
    let first = store.0.lock().unwrap()[0].clone();
    assert_eq!(first.blocking_until_ms(), Some(2_000));
    assert_eq!(first.quota.backoff_level, 1);
    conductor.record(result(429, 1_500)).unwrap();
    let same_window = store.0.lock().unwrap()[0].clone();
    assert_eq!(same_window.blocking_until_ms(), Some(2_000));
    assert_eq!(same_window.quota.backoff_level, 1);
    conductor.record(result(429, 2_000)).unwrap();
    let escalated = store.0.lock().unwrap()[0].clone();
    assert_eq!(escalated.blocking_until_ms(), Some(4_000));
    assert_eq!(escalated.quota.backoff_level, 2);
}

#[test]
fn recoverable_failures_have_finite_deadlines_and_success_clears() {
    let store = Arc::new(MemoryCooldownStore::default());
    let conductor = CooldownConductor::new(store.clone());
    conductor.record(result(503, 10_000)).unwrap();
    assert_eq!(store.0.lock().unwrap()[0].blocking_until_ms(), Some(70_000));
    conductor.record(result(200, 11_000)).unwrap();
    assert!(store.0.lock().unwrap().is_empty());
}

#[test]
fn provider_retry_hint_is_absolute_and_overflow_safe() {
    let store = Arc::new(MemoryCooldownStore::default());
    let conductor = CooldownConductor::new(store.clone());
    let mut hinted = result(429, i64::MAX - 5);
    hinted.retry_delay_ms = Some(10);
    conductor.record(hinted).unwrap();
    assert_eq!(
        store.0.lock().unwrap()[0].blocking_until_ms(),
        Some(i64::MAX)
    );
}
