// ref: sdk/cliproxy/auth/conductor_remove_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::json;

use super::{
    register_refresh_lead_provider, Auth, AuthLifecycle, AuthManager, AuthManagerError,
    AuthMutationOptions, AuthRefresher, AuthSchedulerView, AuthStore, AuthStoreError,
    ExecutionSessionCloser, ProviderExecutorRegistration, ProviderExecutorRegistry,
    RefreshExecutorError, RefreshSchedule, SchedulerCapabilities, SchedulerCapabilitySource,
    SchedulerViewError, CLOSE_ALL_EXECUTION_SESSIONS_ID,
};

fn at(value: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(value)
        .expect("test timestamp")
        .with_timezone(&Utc)
}

#[derive(Default)]
struct RemoveStore {
    records: Mutex<BTreeMap<String, Auth>>,
    deletes: AtomicUsize,
}

impl AuthStore for RemoveStore {
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
        self.records
            .lock()
            .map_err(|_| AuthStoreError::Write)?
            .insert(auth.id.clone(), auth.clone());
        Ok(auth.id.clone())
    }

    fn delete(&self, id: &str) -> Result<(), AuthStoreError> {
        self.deletes.fetch_add(1, Ordering::SeqCst);
        self.records
            .lock()
            .map_err(|_| AuthStoreError::Delete)?
            .remove(id);
        Ok(())
    }
}

#[derive(Default)]
struct ManagerCapabilities(Mutex<BTreeMap<String, SchedulerCapabilities>>);

impl ManagerCapabilities {
    fn set(&self, auth_id: &str, weight: i64) {
        self.0.lock().expect("capabilities").insert(
            auth_id.to_owned(),
            SchedulerCapabilities {
                weight,
                supported_models: vec!["model".into()],
                ..SchedulerCapabilities::default()
            },
        );
    }
}

impl SchedulerCapabilitySource for ManagerCapabilities {
    fn capabilities_for(&self, auth_id: &str, _: &str) -> Option<SchedulerCapabilities> {
        self.0.lock().expect("capabilities").get(auth_id).cloned()
    }
}

#[derive(Default)]
struct ManagerExecutor {
    manager: Mutex<Option<Weak<AuthManager>>>,
    closed: Mutex<Vec<String>>,
}

impl AuthRefresher for ManagerExecutor {
    fn refresh(&self, _: &mut Auth) -> Result<Option<Auth>, RefreshExecutorError> {
        Ok(None)
    }
}

impl ExecutionSessionCloser for ManagerExecutor {
    fn close_execution_session(&self, session_id: &str) {
        self.closed
            .lock()
            .expect("closed sessions")
            .push(session_id.to_owned());
        if let Some(manager) = self
            .manager
            .lock()
            .expect("manager weak")
            .as_ref()
            .and_then(Weak::upgrade)
        {
            manager
                .refresh_scheduler_all()
                .expect("session callback can re-enter manager");
        }
    }
}

fn assembled_manager(
    store: Arc<RemoveStore>,
    capabilities: Arc<ManagerCapabilities>,
) -> (Arc<AuthManager>, Arc<ManagerExecutor>) {
    let lifecycle = Arc::new(AuthLifecycle::new(
        store,
        Arc::new(RefreshSchedule::default()),
        Duration::from_secs(1),
    ));
    let view = Arc::new(AuthSchedulerView::new(lifecycle.clone(), capabilities));
    let executors = Arc::new(ProviderExecutorRegistry::default());
    let manager = Arc::new(AuthManager::new(lifecycle, executors.clone(), view));
    let executor = Arc::new(ManagerExecutor::default());
    *executor.manager.lock().expect("manager weak") = Some(Arc::downgrade(&manager));
    let refresher: Arc<dyn AuthRefresher> = executor.clone();
    let closer: Arc<dyn ExecutionSessionCloser> = executor.clone();
    manager.register_executor(Arc::new(
        ProviderExecutorRegistration::new("claude", refresher)
            .expect("provider")
            .with_session_closer(closer),
    ));
    (manager, executor)
}

fn oauth(id: &str, provider: &str, expires_at: &str) -> Auth {
    let mut auth = Auth::default();
    auth.id = id.into();
    auth.provider = provider.into();
    auth.metadata.insert("access_token".into(), json!("secret"));
    auth.metadata.insert("expires_at".into(), json!(expires_at));
    auth
}

#[test]
fn remove_runtime_unschedules_without_deleting_owning_store() {
    register_refresh_lead_provider("worker-18bk-remove", || Some(Duration::from_secs(10 * 60)));
    let store = Arc::new(RemoveStore::default());
    let schedule = Arc::new(RefreshSchedule::default());
    let lifecycle = AuthLifecycle::new(store.clone(), schedule.clone(), Duration::from_secs(1));
    lifecycle
        .register(
            oauth(
                "remove-runtime",
                "worker-18bk-remove",
                "2026-08-03T13:00:00Z",
            ),
            AuthMutationOptions::default(),
            at("2026-08-03T12:00:00Z"),
        )
        .expect("register");
    assert_eq!(schedule.len(), 1);

    assert!(lifecycle.remove_runtime("remove-runtime"));
    assert!(lifecycle.get_cached("remove-runtime").is_none());
    assert!(schedule.is_empty());
    assert_eq!(store.deletes.load(Ordering::SeqCst), 0);
    assert_eq!(store.list().expect("store").len(), 1);
}

#[test]
fn late_update_after_runtime_remove_is_a_noop() {
    let store = Arc::new(RemoveStore::default());
    let lifecycle = AuthLifecycle::new(
        store,
        Arc::new(RefreshSchedule::default()),
        Duration::from_secs(1),
    );
    lifecycle
        .register(
            oauth("late", "claude", "2026-08-03T13:00:00Z"),
            AuthMutationOptions::default(),
            at("2026-08-03T12:00:00Z"),
        )
        .expect("register");
    assert!(lifecycle.remove_runtime("late"));
    assert!(lifecycle
        .update(
            oauth("late", "claude", "2026-08-03T14:00:00Z"),
            AuthMutationOptions::default(),
            at("2026-08-03T12:01:00Z"),
        )
        .expect("no-op update")
        .is_none());
}

#[test]
fn explicit_delete_removes_store_cache_and_schedule() {
    let store = Arc::new(RemoveStore::default());
    let schedule = Arc::new(RefreshSchedule::default());
    let lifecycle = AuthLifecycle::new(store.clone(), schedule, Duration::from_secs(1));
    lifecycle
        .register(
            oauth("delete", "claude", "2026-08-03T13:00:00Z"),
            AuthMutationOptions::default(),
            at("2026-08-03T12:00:00Z"),
        )
        .expect("register");
    assert_eq!(lifecycle.delete("delete"), Ok(true));
    assert!(lifecycle.is_empty());
    assert!(store.list().expect("store").is_empty());
    assert_eq!(store.deletes.load(Ordering::SeqCst), 1);
}

#[test]
fn manager_runtime_remove_prunes_view_then_closes_sessions_outside_lock() {
    let store = Arc::new(RemoveStore::default());
    let capabilities = Arc::new(ManagerCapabilities::default());
    capabilities.set("managed-remove", 1);
    let (manager, executor) = assembled_manager(store.clone(), capabilities);
    manager
        .register(
            oauth("managed-remove", "claude", "2026-08-03T13:00:00Z"),
            AuthMutationOptions::default(),
            at("2026-08-03T12:00:00Z"),
        )
        .expect("register");
    assert_eq!(manager.candidates().len(), 1);

    assert!(manager.remove_runtime("managed-remove"));
    assert!(manager.candidates().is_empty());
    assert!(manager.lifecycle().get_cached("managed-remove").is_none());
    assert_eq!(store.list().expect("store").len(), 1);
    assert_eq!(store.deletes.load(Ordering::SeqCst), 0);
    assert_eq!(
        *executor.closed.lock().expect("closed sessions"),
        vec![CLOSE_ALL_EXECUTION_SESSIONS_ID.to_owned()]
    );
}

#[test]
fn manager_invalid_updated_capability_removes_stale_routing_entry() {
    let store = Arc::new(RemoveStore::default());
    let capabilities = Arc::new(ManagerCapabilities::default());
    capabilities.set("invalid-update", 1);
    let (manager, _) = assembled_manager(store, capabilities.clone());
    manager
        .register(
            oauth("invalid-update", "claude", "2026-08-03T13:00:00Z"),
            AuthMutationOptions::default(),
            at("2026-08-03T12:00:00Z"),
        )
        .expect("register");
    assert_eq!(manager.candidates().len(), 1);

    capabilities.set("invalid-update", 1_000_001);
    let mut updated = oauth("invalid-update", "claude", "2026-08-03T14:00:00Z");
    updated.metadata.insert("revision".into(), json!(2));
    assert!(matches!(
        manager.update(
            updated,
            AuthMutationOptions::default(),
            at("2026-08-03T12:01:00Z"),
        ),
        Err(AuthManagerError::Scheduler(
            SchedulerViewError::InvalidWeight
        ))
    ));
    assert!(manager.candidates().is_empty());
    assert_eq!(
        manager
            .lifecycle()
            .get_cached("invalid-update")
            .unwrap()
            .metadata["revision"],
        json!(2)
    );
}

#[test]
fn manager_load_clears_old_view_when_new_snapshot_cannot_publish() {
    let store = Arc::new(RemoveStore::default());
    let capabilities = Arc::new(ManagerCapabilities::default());
    capabilities.set("invalid-load", 1);
    let (manager, _) = assembled_manager(store, capabilities.clone());
    manager
        .register(
            oauth("invalid-load", "claude", "2026-08-03T13:00:00Z"),
            AuthMutationOptions::default(),
            at("2026-08-03T12:00:00Z"),
        )
        .expect("register");
    assert_eq!(manager.candidates().len(), 1);

    capabilities.set("invalid-load", 1_000_001);
    assert_eq!(
        manager.load(at("2026-08-03T12:01:00Z")),
        Err(AuthManagerError::Scheduler(
            SchedulerViewError::InvalidWeight
        ))
    );
    assert!(manager.candidates().is_empty());
    assert_eq!(manager.lifecycle().len(), 1);
}
