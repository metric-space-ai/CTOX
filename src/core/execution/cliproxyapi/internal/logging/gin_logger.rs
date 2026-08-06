// ref: internal/logging/gin_logger.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::global_logger::{LogEntry, LogLevel};
use super::requestid::generate_request_id;
use crate::internal::api::middleware::request_logging::mask_sensitive_query;
use std::time::{Duration, SystemTime};

const AI_API_PREFIXES: &[&str] = &["/v1", "/v1beta", "/openai/v1", "/backend-api/codex"];

pub fn is_ai_api_path(path: &str) -> bool {
    AI_API_PREFIXES.iter().any(|prefix| {
        path == *prefix
            || path
                .strip_prefix(prefix)
                .is_some_and(|tail| tail.starts_with('/'))
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestLogStart {
    started_at: SystemTime,
    path: String,
    masked_query: String,
    method: String,
    client_ip: String,
    request_id: Option<String>,
}

impl RequestLogStart {
    pub fn begin(
        started_at: SystemTime,
        path: impl Into<String>,
        query: &str,
        method: impl Into<String>,
        client_ip: impl Into<String>,
    ) -> Self {
        let path = path.into();
        let request_id = is_ai_api_path(&path).then(generate_request_id);
        Self {
            started_at,
            path,
            masked_query: mask_sensitive_query(query),
            method: method.into(),
            client_ip: client_ip.into(),
            request_id,
        }
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn finish(&self, outcome: &RequestLogOutcome, finished_at: SystemTime) -> Option<LogEntry> {
        if outcome.skip {
            return None;
        }
        let latency = finished_at
            .duration_since(self.started_at)
            .unwrap_or(Duration::ZERO);
        let latency = if latency > Duration::from_secs(60) {
            Duration::from_secs(latency.as_secs())
        } else {
            Duration::from_millis(latency.as_millis() as u64)
        };
        let mut path = self.path.clone();
        if !self.masked_query.is_empty() {
            path.push('?');
            path.push_str(&self.masked_query);
        }
        let mut message = format!(
            "{:3} | {:>13?} | {:>15} | {:<7} \"{}\"",
            outcome.status, latency, self.client_ip, self.method, path
        );
        if outcome.credits_used {
            message.push_str(" [credits]");
        }
        if let Some(error) = outcome
            .private_error
            .as_deref()
            .filter(|value| !value.is_empty())
        {
            message.push_str(" | ");
            message.push_str(error);
        }
        let level = match outcome.status {
            500.. => LogLevel::Error,
            400.. => LogLevel::Warn,
            _ => LogLevel::Info,
        };
        let mut entry = LogEntry::new(level, message, finished_at);
        entry.fields.insert(
            "request_id".to_owned(),
            self.request_id
                .clone()
                .unwrap_or_else(|| "--------".to_owned()),
        );
        Some(entry)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestLogOutcome {
    pub status: u16,
    pub private_error: Option<String>,
    pub credits_used: bool,
    pub skip: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryDecision {
    RePanicAbort,
    InternalServerError(LogEntry),
}

pub fn recovery_decision(
    panic_message: &str,
    is_abort_handler: bool,
    path: &str,
    timestamp: SystemTime,
) -> RecoveryDecision {
    if is_abort_handler {
        return RecoveryDecision::RePanicAbort;
    }
    let mut entry = LogEntry::new(LogLevel::Error, "recovered from panic", timestamp);
    entry
        .fields
        .insert("error".to_owned(), panic_message.to_owned());
    entry.fields.insert("reason".to_owned(), path.to_owned());
    RecoveryDecision::InternalServerError(entry)
}
