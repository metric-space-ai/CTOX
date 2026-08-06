// ref: sdk/cliproxy/auth/error_events.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::Serialize;

use super::types::go_zero_time;
use super::{Auth, AuthError, AuthStatus, ModelState, QuotaState};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AuthExecutionResult {
    pub auth_id: String,
    pub provider: String,
    pub model: String,
    pub success: bool,
    pub error: Option<AuthError>,
}

pub trait ErrorEventClock: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
}

pub trait ErrorEventSink: Send + Sync {
    fn publish(&self, payload: &[u8]) -> Result<(), ErrorEventSinkError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ErrorEventSinkError;

impl fmt::Display for ErrorEventSinkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("error event sink rejected payload")
    }
}

impl std::error::Error for ErrorEventSinkError {}

/// Explicitly assembled publisher. CTOX owns persistence and delivery; this
/// SDK layer only builds the upstream-compatible event and invokes its sink.
pub struct ErrorEventPublisher {
    sink: Arc<dyn ErrorEventSink>,
    clock: Arc<dyn ErrorEventClock>,
}

impl ErrorEventPublisher {
    #[must_use]
    pub fn new(sink: Arc<dyn ErrorEventSink>, clock: Arc<dyn ErrorEventClock>) -> Self {
        Self { sink, clock }
    }

    pub fn publish(
        &self,
        result: &AuthExecutionResult,
        auth_snapshot: &Auth,
        home_enabled: bool,
    ) -> Result<bool, ErrorEventSinkError> {
        if result.success || home_enabled {
            return Ok(false);
        }
        let Some(payload) = build_error_event_payload(result, auth_snapshot, self.clock.now())
        else {
            return Ok(false);
        };
        self.sink.publish(&payload)?;
        Ok(true)
    }
}

impl fmt::Debug for ErrorEventPublisher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ErrorEventPublisher")
            .finish_non_exhaustive()
    }
}

#[derive(Serialize)]
struct ErrorEvent<'a> {
    timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "str::is_empty")]
    provider: &'a str,
    #[serde(skip_serializing_if = "str::is_empty")]
    model: &'a str,
    #[serde(skip_serializing_if = "str::is_empty")]
    auth_id: &'a str,
    auth_index: String,
    status_code: u16,
    body: String,
    #[serde(skip_serializing_if = "str::is_empty")]
    code: &'a str,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    retryable: bool,
    auth_status: ErrorEventAuthStatus<'a>,
}

#[derive(Serialize)]
struct ErrorEventAuthStatus<'a> {
    status: &'a AuthStatus,
    #[serde(skip_serializing_if = "str::is_empty")]
    status_message: &'a str,
    disabled: bool,
    unavailable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_retry_after: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quota: Option<ErrorEventQuotaStatus<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<ErrorEventModelStatus<'a>>,
}

#[derive(Serialize)]
struct ErrorEventQuotaStatus<'a> {
    exceeded: bool,
    #[serde(skip_serializing_if = "str::is_empty")]
    reason: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_recover_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "is_zero_i64")]
    backoff_level: i64,
}

#[derive(Serialize)]
struct ErrorEventModelStatus<'a> {
    name: &'a str,
    status: &'a AuthStatus,
    #[serde(skip_serializing_if = "str::is_empty")]
    status_message: &'a str,
    unavailable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_retry_after: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quota: Option<ErrorEventQuotaStatus<'a>>,
}

#[must_use]
pub fn build_error_event_payload(
    result: &AuthExecutionResult,
    auth_snapshot: &Auth,
    timestamp: DateTime<Utc>,
) -> Option<Vec<u8>> {
    if result.success {
        return None;
    }
    let mut indexed_auth = auth_snapshot.clone();
    let auth_index = indexed_auth.ensure_index();
    let error = result.error.as_ref();
    let event = ErrorEvent {
        timestamp,
        provider: result.provider.trim(),
        model: result.model.trim(),
        auth_id: result.auth_id.trim(),
        auth_index,
        status_code: error_event_status_code(error),
        body: error_event_body(error),
        code: error.map_or("", |error| error.code.trim()),
        retryable: error.is_some_and(|error| error.retryable),
        auth_status: build_error_event_auth_status(result.model.trim(), auth_snapshot),
    };
    serde_json::to_vec(&event).ok()
}

fn build_error_event_auth_status<'a>(
    model: &'a str,
    auth_snapshot: &'a Auth,
) -> ErrorEventAuthStatus<'a> {
    ErrorEventAuthStatus {
        status: &auth_snapshot.status,
        status_message: auth_snapshot.status_message.trim(),
        disabled: auth_snapshot.disabled,
        unavailable: auth_snapshot.unavailable,
        next_retry_after: time_if_set(auth_snapshot.next_retry_after),
        quota: error_event_quota_status_from(&auth_snapshot.quota),
        model: error_event_model_status_from(model, auth_snapshot),
    }
}

fn error_event_model_status_from<'a>(
    model: &'a str,
    auth_snapshot: &'a Auth,
) -> Option<ErrorEventModelStatus<'a>> {
    let model = model.trim();
    if model.is_empty() {
        return None;
    }
    let state = auth_snapshot.model_states.get(model)?;
    Some(model_status(model, state))
}

fn model_status<'a>(model: &'a str, state: &'a ModelState) -> ErrorEventModelStatus<'a> {
    ErrorEventModelStatus {
        name: model,
        status: &state.status,
        status_message: state.status_message.trim(),
        unavailable: state.unavailable,
        next_retry_after: time_if_set(state.next_retry_after),
        quota: error_event_quota_status_from(&state.quota),
    }
}

fn error_event_quota_status_from(quota: &QuotaState) -> Option<ErrorEventQuotaStatus<'_>> {
    if !quota.exceeded
        && quota.reason.trim().is_empty()
        && quota.next_recover_at == go_zero_time()
        && quota.backoff_level == 0
    {
        return None;
    }
    Some(ErrorEventQuotaStatus {
        exceeded: quota.exceeded,
        reason: quota.reason.trim(),
        next_recover_at: time_if_set(quota.next_recover_at),
        backoff_level: quota.backoff_level,
    })
}

fn error_event_status_code(error: Option<&AuthError>) -> u16 {
    error
        .map(AuthError::status_code)
        .filter(|code| *code > 0)
        .unwrap_or(500)
}

fn error_event_body(error: Option<&AuthError>) -> String {
    error
        .map(|error| error.message.trim())
        .filter(|message| !message.is_empty())
        .unwrap_or("request failed")
        .to_owned()
}

fn time_if_set(value: DateTime<Utc>) -> Option<DateTime<Utc>> {
    (value != go_zero_time()).then_some(value)
}

const fn is_zero_i64(value: &i64) -> bool {
    *value == 0
}
