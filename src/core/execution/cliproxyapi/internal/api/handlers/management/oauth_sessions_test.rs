// ref: internal/api/handlers/management/oauth_sessions_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use super::{
    ManagementOAuthClock, ManagementOAuthSessionError, ManagementOAuthSessionSource,
    ManagementOAuthSessions,
};

#[derive(Default)]
struct Clock(AtomicI64);

impl ManagementOAuthClock for Clock {
    fn now_ms(&self) -> i64 {
        self.0.load(Ordering::SeqCst)
    }
}

#[test]
fn completion_keeps_short_lived_tombstone_and_does_not_extend_it() {
    let clock = Arc::new(Clock::default());
    let sessions = ManagementOAuthSessions::with_ttl(clock.clone(), 60_000);
    sessions
        .register_builtin("completed-state", "codex")
        .unwrap();
    assert!(sessions.complete("completed-state").unwrap());
    let before = sessions.details("completed-state").unwrap().unwrap();
    assert!(before.completed);
    assert!(sessions
        .visible_status("completed-state")
        .unwrap()
        .is_none());
    clock.0.store(10_000, Ordering::SeqCst);
    assert!(!sessions.complete("completed-state").unwrap());
    assert_eq!(
        sessions
            .details("completed-state")
            .unwrap()
            .unwrap()
            .expires_at_ms,
        before.expires_at_ms
    );
}

#[test]
fn provider_completion_skips_existing_tombstones_and_other_sources() {
    let sessions = ManagementOAuthSessions::new(Arc::new(Clock::default()));
    sessions.register_builtin("done", "codex").unwrap();
    sessions.register_builtin("pending", "openai").unwrap();
    sessions
        .register_plugin("plugin", "codex", BTreeMap::new())
        .unwrap();
    sessions.complete("done").unwrap();
    assert_eq!(
        sessions
            .complete_provider("codex", ManagementOAuthSessionSource::Builtin)
            .unwrap(),
        1
    );
    assert!(!sessions.details("plugin").unwrap().unwrap().completed);
}

#[test]
fn cancel_only_removes_pending_sessions() {
    let sessions = ManagementOAuthSessions::new(Arc::new(Clock::default()));
    sessions.register_builtin("pending", "xai").unwrap();
    assert!(sessions.cancel("pending").unwrap());
    assert!(sessions.details("pending").unwrap().is_none());
    assert!(!sessions.cancel("pending").unwrap());

    sessions.register_builtin("complete", "codex").unwrap();
    sessions.complete("complete").unwrap();
    assert!(!sessions.cancel("complete").unwrap());
    sessions.register_builtin("error", "anthropic").unwrap();
    sessions.set_error("error", "failed").unwrap();
    assert!(!sessions.cancel("error").unwrap());
}

#[test]
fn save_guard_rejects_cancelled_completed_errored_and_expired_sessions() {
    let clock = Arc::new(Clock::default());
    let sessions = ManagementOAuthSessions::with_ttl(clock.clone(), 100);
    for state in ["cancelled", "completed", "errored", "expired"] {
        sessions.register_builtin(state, "claude").unwrap();
    }
    sessions.cancel("cancelled").unwrap();
    sessions.complete("completed").unwrap();
    sessions.set_error("errored", "failed").unwrap();
    clock.0.store(101, Ordering::SeqCst);
    for state in ["cancelled", "completed", "errored", "expired"] {
        assert_eq!(
            sessions.guard_pending_for_save(state, "anthropic"),
            Err(ManagementOAuthSessionError::SessionNotPending)
        );
    }
}

#[test]
fn states_and_plugin_providers_reject_path_or_identifier_injection() {
    let sessions = ManagementOAuthSessions::new(Arc::new(Clock::default()));
    for state in ["", "../state", "nested/state", "bad state"] {
        assert_eq!(
            sessions.register_builtin(state, "codex"),
            Err(ManagementOAuthSessionError::InvalidState)
        );
    }
    for provider in ["", "plugin/name", "Plugin.Name"] {
        assert_eq!(
            sessions.register_plugin("safe-state", provider, BTreeMap::new()),
            Err(ManagementOAuthSessionError::UnsupportedProvider)
        );
    }
}

#[test]
fn metadata_is_redacted_and_cleared_on_completion() {
    let sessions = ManagementOAuthSessions::new(Arc::new(Clock::default()));
    sessions
        .register_plugin(
            "plugin-state",
            "custom-plugin",
            BTreeMap::from([(
                "access_token".to_owned(),
                serde_json::Value::String("never-render".to_owned()),
            )]),
        )
        .unwrap();
    let before = sessions.details("plugin-state").unwrap().unwrap();
    assert!(!format!("{before:?}").contains("never-render"));
    sessions.complete("plugin-state").unwrap();
    assert!(sessions
        .details("plugin-state")
        .unwrap()
        .unwrap()
        .metadata
        .is_empty());
}
