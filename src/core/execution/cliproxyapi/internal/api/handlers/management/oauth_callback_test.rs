// ref: internal/api/handlers/management/oauth_callback_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use super::{
    ManagementOAuthCallback, ManagementOAuthCallbackHandler, ManagementOAuthCallbackRequest,
    ManagementOAuthCallbackRequestError, ManagementOAuthCallbackSink,
    ManagementOAuthCallbackSinkError, ManagementOAuthCallbacks, ManagementOAuthClock,
    ManagementOAuthSessions,
};

pub(super) struct Clock;

impl ManagementOAuthClock for Clock {
    fn now_ms(&self) -> i64 {
        1_000
    }
}

#[derive(Default)]
pub(super) struct Sink(pub(super) Mutex<Vec<ManagementOAuthCallback>>);

impl ManagementOAuthCallbackSink for Sink {
    fn exchange(
        &self,
        callback: &ManagementOAuthCallback,
    ) -> Result<(), ManagementOAuthCallbackSinkError> {
        self.0.lock().unwrap().push(callback.clone());
        Ok(())
    }
}

pub(super) fn setup() -> (
    Arc<ManagementOAuthSessions>,
    Arc<Sink>,
    ManagementOAuthCallbackHandler,
) {
    let sessions = Arc::new(ManagementOAuthSessions::new(Arc::new(Clock)));
    let sink = Arc::new(Sink::default());
    let callbacks = Arc::new(ManagementOAuthCallbacks::new(
        sessions.clone(),
        sink.clone(),
    ));
    let handler = ManagementOAuthCallbackHandler::new(sessions.clone(), callbacks);
    (sessions, sink, handler)
}

#[test]
fn post_callback_uses_session_provider_and_redacts_code() {
    let (sessions, sink, handler) = setup();
    sessions.register_builtin("state-a", "openai").unwrap();
    let request = ManagementOAuthCallbackRequest {
        state: "state-a".to_owned(),
        code: "private-code".to_owned(),
        ..Default::default()
    };
    assert!(!format!("{request:?}").contains("private-code"));
    handler.submit(request).unwrap();
    assert_eq!(sink.0.lock().unwrap()[0].provider, "codex");
    assert!(sessions.details("state-a").unwrap().unwrap().completed);
}

#[test]
fn redirect_url_supplies_plugin_provider_callback_fields() {
    let (sessions, sink, handler) = setup();
    sessions
        .register_plugin("plugin-state", "custom-provider", BTreeMap::new())
        .unwrap();
    handler
        .submit(ManagementOAuthCallbackRequest {
            provider: "custom-provider".to_owned(),
            redirect_url: "https://callback.invalid/?state=plugin-state&code=plugin-secret"
                .to_owned(),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(sink.0.lock().unwrap()[0].provider, "custom-provider");
}

#[test]
fn plugin_provider_is_not_aliased_to_builtin_provider() {
    let (sessions, _, handler) = setup();
    sessions
        .register_plugin("plugin-state", "openai", BTreeMap::new())
        .unwrap();
    assert_eq!(
        handler.submit(ManagementOAuthCallbackRequest {
            provider: "codex".to_owned(),
            state: "plugin-state".to_owned(),
            code: "code".to_owned(),
            ..Default::default()
        }),
        Err(ManagementOAuthCallbackRequestError::ProviderMismatch)
    );
}

#[test]
fn unknown_completed_and_path_like_states_fail_closed() {
    let (sessions, _, handler) = setup();
    sessions.register_builtin("completed", "codex").unwrap();
    sessions.complete("completed").unwrap();
    for (state, expected) in [
        (
            "missing",
            ManagementOAuthCallbackRequestError::UnknownSession,
        ),
        (
            "completed",
            ManagementOAuthCallbackRequestError::CompletedSession,
        ),
        (
            "../state",
            ManagementOAuthCallbackRequestError::InvalidState,
        ),
    ] {
        assert_eq!(
            handler.submit(ManagementOAuthCallbackRequest {
                state: state.to_owned(),
                code: "code".to_owned(),
                ..Default::default()
            }),
            Err(expected)
        );
    }
}

#[test]
fn oauth_error_marks_only_the_selected_session_failed() {
    let (sessions, sink, handler) = setup();
    sessions.register_builtin("failed", "codex").unwrap();
    sessions.register_builtin("pending", "codex").unwrap();
    handler
        .submit(ManagementOAuthCallbackRequest {
            state: "failed".to_owned(),
            error: "access_denied".to_owned(),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        sessions.details("failed").unwrap().unwrap().status,
        "access_denied"
    );
    assert!(sessions.is_pending("pending", "codex").unwrap());
    assert!(sink.0.lock().unwrap().is_empty());
}
