// ref: internal/runtime/executor/claude_executor_fast_error.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::error::Error;
use std::fmt;

use crate::sdk::cliproxy::executor::{
    Headers, RequestScopedError, RequestTerminatedError, StatusError,
};

#[derive(Debug)]
pub struct ClaudeFastRequestError {
    status: u16,
    cause: Box<dyn Error + Send + Sync>,
}

impl fmt::Display for ClaudeFastRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl Error for ClaudeFastRequestError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.cause.as_ref())
    }
}
impl StatusError for ClaudeFastRequestError {
    fn status_code(&self) -> u16 {
        if (200..300).contains(&self.status) {
            0
        } else {
            self.status
        }
    }
}
impl RequestScopedError for ClaudeFastRequestError {
    fn is_request_scoped(&self) -> bool {
        true
    }
}

pub fn wrap_claude_fast_request_error(
    fast: bool,
    status: u16,
    error: Box<dyn Error + Send + Sync>,
) -> Box<dyn Error + Send + Sync> {
    if fast {
        Box::new(ClaudeFastRequestError {
            status,
            cause: error,
        })
    } else {
        error
    }
}

/// Preserves an already-decoded Fast response across the auth conductor. The
/// representation headers are removed because they describe the compressed
/// upstream bytes, not `body`.
pub fn claude_fast_direct_response_error(
    status: u16,
    mut headers: Headers,
    body: &[u8],
) -> RequestTerminatedError {
    remove_header_case_insensitive(&mut headers, "content-encoding");
    remove_header_case_insensitive(&mut headers, "content-length");
    RequestTerminatedError {
        http_status: status,
        headers,
        body: body.to_vec(),
    }
}

fn remove_header_case_insensitive(headers: &mut Headers, name: &str) {
    let keys = headers
        .keys()
        .filter(|key| key.eq_ignore_ascii_case(name))
        .cloned()
        .collect::<Vec<_>>();
    for key in keys {
        headers.remove(&key);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaudeEntitlementError {
    pub status: u16,
    pub body: Vec<u8>,
}
impl fmt::Display for ClaudeEntitlementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Claude upstream returned {}", self.status)
    }
}
impl Error for ClaudeEntitlementError {}
impl StatusError for ClaudeEntitlementError {
    fn status_code(&self) -> u16 {
        self.status
    }
}
impl RequestScopedError for ClaudeEntitlementError {
    fn is_request_scoped(&self) -> bool {
        true
    }
}

pub fn claude_body_indicates_fast_mode_credits(body: &[u8]) -> bool {
    let parsed = serde_json::from_slice::<serde_json::Value>(body).ok();
    let message = parsed
        .as_ref()
        .and_then(|root| root.pointer("/error/message"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| std::str::from_utf8(body).unwrap_or_default())
        .to_ascii_lowercase();
    message.contains("fast mode")
        && (message.contains("usage credits") || message.contains("credits are required"))
}

pub fn classify_claude_upstream_error(status: u16, body: &[u8]) -> Option<ClaudeEntitlementError> {
    (status == 429 && claude_body_indicates_fast_mode_credits(body)).then(|| {
        ClaudeEntitlementError {
            status,
            body: body.to_vec(),
        }
    })
}
