// ref: sdk/cliproxy/auth/conductor_update_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::json;

use super::{
    Auth, AuthLifecycle, AuthLifecycleError, AuthMutationOptions, AuthStatus, AuthStore,
    AuthStoreError, ModelState, PersistenceIntent, QuotaState, RefreshSchedule,
};

fn at(value: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(value)
        .expect("test timestamp")
        .with_timezone(&Utc)
}

#[derive(Default)]
struct LifecycleStore {
    records: Mutex<BTreeMap<String, Auth>>,
    saves: AtomicUsize,
    fail_save: AtomicBool,
}

impl LifecycleStore {
    fn insert(&self, auth: Auth) {
        self.records
            .lock()
            .expect("test store")
            .insert(auth.id.clone(), auth);
    }

    fn remove_without_lifecycle(&self, id: &str) {
        self.records.lock().expect("test store").remove(id);
    }
}

impl AuthStore for LifecycleStore {
    fn list(&self) -> Result<Vec<Auth>, AuthStoreError> {
        Ok(self
            .records
            .lock()
            .map_err(|_| AuthStoreError::Read)?
            .values()
            .cloned()
            .collect())
    }

    fn save(&self, auth: &Auth) -> Result<String, AuthStoreError> {
        self.saves.fetch_add(1, Ordering::SeqCst);
        if self.fail_save.load(Ordering::SeqCst) {
            return Err(AuthStoreError::Write);
        }
        self.records
            .lock()
            .map_err(|_| AuthStoreError::Write)?
            .insert(auth.id.clone(), auth.clone());
        Ok(format!("ctox-auth://{}", auth.id))
    }

    fn delete(&self, id: &str) -> Result<(), AuthStoreError> {
        self.records
            .lock()
            .map_err(|_| AuthStoreError::Delete)?
            .remove(id);
        Ok(())
    }
}

fn oauth(id: &str) -> Auth {
    let mut auth = Auth::default();
    auth.id = id.into();
    auth.provider = "claude".into();
    auth.status = AuthStatus::Active;
    auth.metadata.insert("access_token".into(), json!("secret"));
    auth
}

fn lifecycle(store: Arc<LifecycleStore>) -> AuthLifecycle {
    AuthLifecycle::new(
        store,
        Arc::new(RefreshSchedule::default()),
        Duration::from_secs(1),
    )
}

#[test]
fn load_replaces_cache_from_authoritative_store_and_assigns_indexes() {
    let store = Arc::new(LifecycleStore::default());
    store.insert(oauth("stored"));
    let lifecycle = lifecycle(store);
    assert_eq!(lifecycle.load(at("2026-08-03T12:00:00Z")), Ok(1));
    let loaded = lifecycle.get_cached("stored").expect("loaded auth");
    assert_eq!(loaded.id, "stored");
    assert_eq!(loaded.index.len(), 16);
}

#[test]
fn register_persists_before_cache_publication_and_generates_stable_id() {
    let store = Arc::new(LifecycleStore::default());
    let lifecycle = lifecycle(store.clone());
    store.fail_save.store(true, Ordering::SeqCst);
    assert!(matches!(
        lifecycle.register(
            oauth("must-not-publish"),
            AuthMutationOptions::default(),
            at("2026-08-03T12:00:00Z"),
        ),
        Err(AuthLifecycleError::Store(AuthStoreError::Write))
    ));
    assert!(lifecycle.is_empty());

    store.fail_save.store(false, Ordering::SeqCst);
    let mut generated = oauth("");
    generated.id.clear();
    let generated = lifecycle
        .register(
            generated,
            AuthMutationOptions::default(),
            at("2026-08-03T12:00:00Z"),
        )
        .expect("registered auth");
    assert!(!generated.id.is_empty());
    assert_eq!(lifecycle.len(), 1);
}

#[test]
fn active_update_preserves_model_states_counters_and_recent_requests() {
    let store = Arc::new(LifecycleStore::default());
    let lifecycle = lifecycle(store);
    let mut original = oauth("active");
    original.success = 7;
    original.failed = 3;
    original.record_recent_request(1_700_000_000, true);
    original.model_states.insert(
        "claude-opus".into(),
        ModelState {
            quota: QuotaState {
                backoff_level: 7,
                ..QuotaState::default()
            },
            ..ModelState::default()
        },
    );
    lifecycle
        .register(
            original,
            AuthMutationOptions::default(),
            at("2026-08-03T12:00:00Z"),
        )
        .expect("register");

    let mut incoming = oauth("active");
    incoming
        .metadata
        .insert("access_token".into(), json!("new"));
    let updated = lifecycle
        .update(
            incoming,
            AuthMutationOptions::default(),
            at("2026-08-03T12:01:00Z"),
        )
        .expect("update")
        .expect("existing auth");
    assert_eq!(updated.model_states["claude-opus"].quota.backoff_level, 7);
    assert_eq!(updated.success, 7);
    assert_eq!(updated.failed, 3);
    assert_eq!(
        updated
            .recent_requests_snapshot(1_700_000_000)
            .last()
            .expect("current bucket")
            .success,
        1
    );
}

#[test]
fn disabled_boundary_never_inherits_stale_model_state() {
    for (existing_disabled, incoming_disabled) in [(true, true), (false, true), (true, false)] {
        let store = Arc::new(LifecycleStore::default());
        let lifecycle = lifecycle(store);
        let mut existing = oauth("transition");
        existing.disabled = existing_disabled;
        existing.status = if existing_disabled {
            AuthStatus::Disabled
        } else {
            AuthStatus::Active
        };
        existing
            .model_states
            .insert("stale".into(), ModelState::default());
        lifecycle
            .register(
                existing,
                AuthMutationOptions::default(),
                at("2026-08-03T12:00:00Z"),
            )
            .expect("register");
        let mut incoming = oauth("transition");
        incoming.disabled = incoming_disabled;
        incoming.status = if incoming_disabled {
            AuthStatus::Disabled
        } else {
            AuthStatus::Active
        };
        let updated = lifecycle
            .update(
                incoming,
                AuthMutationOptions::default(),
                at("2026-08-03T12:01:00Z"),
            )
            .expect("update")
            .expect("existing");
        assert!(updated.model_states.is_empty());
    }
}

#[test]
fn missing_durable_record_cannot_be_resurrected_by_late_update() {
    let store = Arc::new(LifecycleStore::default());
    let lifecycle = lifecycle(store.clone());
    lifecycle
        .register(
            oauth("removed-at-source"),
            AuthMutationOptions::default(),
            at("2026-08-03T12:00:00Z"),
        )
        .expect("register");
    store.remove_without_lifecycle("removed-at-source");
    assert!(matches!(
        lifecycle.update(
            oauth("removed-at-source"),
            AuthMutationOptions::default(),
            at("2026-08-03T12:01:00Z"),
        ),
        Err(AuthLifecycleError::DurableRecordMissing)
    ));
}

#[test]
fn update_store_failure_keeps_previous_cache_visible() {
    let store = Arc::new(LifecycleStore::default());
    let lifecycle = lifecycle(store.clone());
    lifecycle
        .register(
            oauth("rollback"),
            AuthMutationOptions::default(),
            at("2026-08-03T12:00:00Z"),
        )
        .expect("register");
    store.fail_save.store(true, Ordering::SeqCst);
    let mut changed = oauth("rollback");
    changed.label = "must-not-publish".into();
    assert!(matches!(
        lifecycle.update(
            changed,
            AuthMutationOptions::default(),
            at("2026-08-03T12:01:00Z"),
        ),
        Err(AuthLifecycleError::Store(AuthStoreError::Write))
    ));
    assert!(lifecycle
        .get_cached("rollback")
        .expect("previous cache")
        .label
        .is_empty());
}

#[test]
fn persistence_ownership_cannot_change_during_update() {
    let store = Arc::new(LifecycleStore::default());
    let lifecycle = lifecycle(store);
    lifecycle
        .register(
            oauth("ownership"),
            AuthMutationOptions::default(),
            at("2026-08-03T12:00:00Z"),
        )
        .expect("register");
    let mut runtime_only = oauth("ownership");
    runtime_only
        .attributes
        .insert("runtime_only".into(), "true".into());
    assert!(matches!(
        lifecycle.update(
            runtime_only,
            AuthMutationOptions::default(),
            at("2026-08-03T12:01:00Z"),
        ),
        Err(AuthLifecycleError::PersistenceClassChange)
    ));
}

#[test]
fn source_already_persisted_is_typed_and_skips_writeback() {
    let store = Arc::new(LifecycleStore::default());
    store.insert(oauth("watcher"));
    let lifecycle = lifecycle(store.clone());
    lifecycle
        .register(
            oauth("watcher"),
            AuthMutationOptions {
                persistence: PersistenceIntent::SourceAlreadyPersisted,
            },
            at("2026-08-03T12:00:00Z"),
        )
        .expect("runtime projection");
    assert_eq!(store.saves.load(Ordering::SeqCst), 0);
}
