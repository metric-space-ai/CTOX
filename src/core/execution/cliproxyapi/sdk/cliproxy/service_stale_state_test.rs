// ref: sdk/cliproxy/service_stale_state_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use chrono::{DateTime, Utc};

use super::auth::types::is_go_zero_time;
use super::auth::{ModelState, QuotaState};
use super::service_test_support::{auth, runtime_fixture};

fn at(value: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(value)
        .expect("test timestamp")
        .with_timezone(&Utc)
}

#[test]
fn delete_then_readd_does_not_inherit_stale_runtime_state() {
    let fixture = runtime_fixture(None);
    let mut initial = auth("stale-auth", "claude");
    initial.last_refreshed_at = at("2026-03-01T08:00:00Z");
    initial.next_refresh_after = at("2026-03-01T08:30:00Z");
    initial.model_states.insert(
        "stale-model".into(),
        ModelState {
            quota: QuotaState {
                backoff_level: 7,
                ..QuotaState::default()
            },
            ..ModelState::default()
        },
    );
    fixture
        .runtime
        .apply_core_auth_add_or_update(initial)
        .unwrap();
    assert!(fixture.runtime.apply_core_auth_removal("stale-auth"));
    assert!(fixture
        .runtime
        .auth_manager()
        .lifecycle()
        .get_cached("stale-auth")
        .is_none());

    fixture
        .runtime
        .apply_core_auth_add_or_update(auth("stale-auth", "claude"))
        .unwrap();
    let readded = fixture
        .runtime
        .auth_manager()
        .lifecycle()
        .get_cached("stale-auth")
        .expect("re-added auth");
    assert!(is_go_zero_time(&readded.last_refreshed_at));
    assert!(is_go_zero_time(&readded.next_refresh_after));
    assert!(readded.model_states.is_empty());
    assert!(!fixture.registry.models_for("stale-auth").is_empty());
    assert!(fixture
        .registry
        .unregisters()
        .contains(&"stale-auth".to_owned()));
}

#[test]
fn active_modify_preserves_refresh_and_model_state() {
    let fixture = runtime_fixture(None);
    let mut initial = auth("active-auth", "claude");
    initial.last_refreshed_at = at("2026-03-01T08:00:00Z");
    initial.next_refresh_after = at("2026-03-01T08:30:00Z");
    initial
        .model_states
        .insert("model".into(), ModelState::default());
    fixture
        .runtime
        .apply_core_auth_add_or_update(initial)
        .unwrap();
    let mut update = auth("active-auth", "claude");
    update.label = "updated".into();
    fixture
        .runtime
        .apply_core_auth_add_or_update(update)
        .unwrap();
    let current = fixture
        .runtime
        .auth_manager()
        .lifecycle()
        .get_cached("active-auth")
        .unwrap();
    assert_eq!(current.last_refreshed_at, at("2026-03-01T08:00:00Z"));
    assert_eq!(current.next_refresh_after, at("2026-03-01T08:30:00Z"));
    assert!(current.model_states.contains_key("model"));
}
