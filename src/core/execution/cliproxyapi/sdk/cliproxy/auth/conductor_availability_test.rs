// ref: sdk/cliproxy/auth/conductor_availability_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Default)]
struct Store(Mutex<BTreeMap<String, Auth>>);
impl AuthStore for Store {
    fn list(&self) -> Result<Vec<Auth>, AuthStoreError> {
        Ok(self
            .0
            .lock()
            .map_err(|_| AuthStoreError::Read)?
            .values()
            .cloned()
            .collect())
    }
    fn save(&self, auth: &Auth) -> Result<String, AuthStoreError> {
        self.0
            .lock()
            .map_err(|_| AuthStoreError::Write)?
            .insert(auth.id.clone(), auth.clone());
        Ok(auth.id.clone())
    }
    fn delete(&self, id: &str) -> Result<(), AuthStoreError> {
        self.0
            .lock()
            .map_err(|_| AuthStoreError::Delete)?
            .remove(id);
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct Caps(Mutex<BTreeMap<String, SchedulerCapabilities>>);
impl Caps {
    pub(super) fn set(&self, id: &str, weight: i64) {
        self.0.lock().unwrap().insert(
            id.into(),
            SchedulerCapabilities {
                weight,
                supported_models: vec!["model".into()],
                ..SchedulerCapabilities::default()
            },
        );
    }
}
impl SchedulerCapabilitySource for Caps {
    fn capabilities_for(&self, id: &str, _: &str) -> Option<SchedulerCapabilities> {
        self.0.lock().unwrap().get(id).cloned()
    }
}

pub(super) fn manager() -> (AuthManager, Arc<Caps>) {
    let lifecycle = Arc::new(AuthLifecycle::new(
        Arc::new(Store::default()),
        Arc::new(RefreshSchedule::default()),
        Duration::from_secs(60),
    ));
    let caps = Arc::new(Caps::default());
    let view = Arc::new(AuthSchedulerView::new(lifecycle.clone(), caps.clone()));
    (
        AuthManager::new(
            lifecycle,
            Arc::new(ProviderExecutorRegistry::default()),
            view,
        ),
        caps,
    )
}

pub(super) fn auth(id: &str, disabled: bool) -> Auth {
    let mut auth = Auth::default();
    auth.id = id.into();
    auth.provider = "Claude".into();
    auth.status = AuthStatus::Active;
    auth.disabled = disabled;
    auth
}

pub(super) fn register(manager: &AuthManager, auth: Auth) -> Result<Auth, AuthManagerError> {
    let now = DateTime::parse_from_rfc3339("2026-08-04T12:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    manager.register(auth, AuthMutationOptions::default(), now)
}

#[test]
fn unavailable_without_deadline_does_not_create_endless_block() {
    let record = CooldownStateRecord {
        provider: "claude".into(),
        auth_id: "a".into(),
        model: None,
        status: "error".into(),
        next_retry_after_ms: None,
        reason: "error".into(),
        quota: CooldownQuotaState::default(),
        last_error: None,
        updated_at_ms: 1,
    };
    assert!(record.is_available_at(2));
    let future = CooldownStateRecord {
        next_retry_after_ms: Some(100),
        ..record
    };
    assert!(!future.is_available_at(99));
}

#[test]
fn available_provider_view_excludes_disabled_and_is_case_insensitive() {
    let (manager, caps) = manager();
    caps.set("active", 1);
    caps.set("disabled", 1);
    register(&manager, auth("active", false)).unwrap();
    register(&manager, auth("disabled", true)).unwrap();
    assert_eq!(manager.available_providers(), ["claude"]);
    assert!(manager.has_provider_auth("CLAUDE"));
    assert!(!manager.has_provider_auth("codex"));
}
