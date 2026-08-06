// ref: sdk/cliproxy/auth/error_events_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use chrono::{TimeZone, Utc};
use serde_json::Value;

use super::{
    Auth, AuthError, AuthExecutionResult, AuthStatus, ErrorEventClock, ErrorEventPublisher,
    ErrorEventSink, ErrorEventSinkError, ModelState, QuotaState,
};

#[derive(Default)]
struct RecordingSink(Mutex<Vec<Vec<u8>>>);

impl ErrorEventSink for RecordingSink {
    fn publish(&self, payload: &[u8]) -> Result<(), ErrorEventSinkError> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(payload.to_vec());
        Ok(())
    }
}

struct FixedClock;

impl ErrorEventClock for FixedClock {
    fn now(&self) -> chrono::DateTime<Utc> {
        Utc.timestamp_opt(1_700_000_000, 0).single().unwrap()
    }
}

fn failed_result() -> AuthExecutionResult {
    AuthExecutionResult {
        auth_id: "auth-error-event".to_owned(),
        provider: "codex".to_owned(),
        model: "gpt-5".to_owned(),
        success: false,
        error: Some(AuthError {
            code: "rate_limit".to_owned(),
            message: r#"{"error":"quota"}"#.to_owned(),
            retryable: true,
            http_status: 429,
        }),
    }
}

fn error_auth() -> Auth {
    let quota = QuotaState {
        exceeded: true,
        reason: "quota".to_owned(),
        ..QuotaState::default()
    };
    let mut auth = Auth::default();
    auth.id = "auth-error-event".to_owned();
    auth.index = "stable-index".to_owned();
    auth.provider = "codex".to_owned();
    auth.status = AuthStatus::Error;
    auth.unavailable = true;
    auth.quota = quota.clone();
    auth.model_states = [(
        "gpt-5".to_owned(),
        ModelState {
            status: AuthStatus::Error,
            unavailable: true,
            quota,
            ..ModelState::default()
        },
    )]
    .into();
    auth
}

#[test]
fn failed_result_publishes_state_after_transition() {
    let sink = Arc::new(RecordingSink::default());
    let publisher = ErrorEventPublisher::new(sink.clone(), Arc::new(FixedClock));
    assert_eq!(
        publisher.publish(&failed_result(), &error_auth(), false),
        Ok(true)
    );

    let payloads = sink
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let event: Value = serde_json::from_slice(&payloads[0]).unwrap();
    assert_eq!(event["provider"], "codex");
    assert_eq!(event["model"], "gpt-5");
    assert_eq!(event["auth_id"], "auth-error-event");
    assert_eq!(event["auth_index"], "stable-index");
    assert_eq!(event["status_code"], 429);
    assert_eq!(event["body"], r#"{"error":"quota"}"#);
    assert_eq!(event["code"], "rate_limit");
    assert_eq!(event["retryable"], true);
    assert_eq!(event["auth_status"]["status"], "error");
    assert_eq!(event["auth_status"]["unavailable"], true);
    assert_eq!(event["auth_status"]["quota"]["reason"], "quota");
    assert_eq!(event["auth_status"]["model"]["name"], "gpt-5");
    assert_eq!(event["auth_status"]["model"]["quota"]["exceeded"], true);
}

#[test]
fn home_mode_and_success_skip_publication() {
    let sink = Arc::new(RecordingSink::default());
    let publisher = ErrorEventPublisher::new(sink.clone(), Arc::new(FixedClock));
    assert_eq!(
        publisher.publish(&failed_result(), &error_auth(), true),
        Ok(false)
    );

    let mut success = failed_result();
    success.success = true;
    assert_eq!(publisher.publish(&success, &error_auth(), false), Ok(false));
    assert!(sink
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_empty());
}
