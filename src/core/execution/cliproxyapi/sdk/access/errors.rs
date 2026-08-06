// ref: sdk/access/errors.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::borrow::Cow;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

pub const HTTP_STATUS_UNAUTHORIZED: i32 = 401;
pub const HTTP_STATUS_INTERNAL_SERVER_ERROR: i32 = 500;

/// Classifies authentication failures.
///
/// The string newtype preserves Go's zero value and future error codes without
/// restricting equality to the constants known by this pinned revision.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AuthErrorCode(Cow<'static, str>);

impl AuthErrorCode {
    const fn borrowed(code: &'static str) -> Self {
        Self(Cow::Borrowed(code))
    }

    #[must_use]
    pub fn new(code: impl Into<String>) -> Self {
        Self(Cow::Owned(code.into()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub const AUTH_ERROR_CODE_NO_CREDENTIALS: AuthErrorCode = AuthErrorCode::borrowed("no_credentials");
pub const AUTH_ERROR_CODE_INVALID_CREDENTIAL: AuthErrorCode =
    AuthErrorCode::borrowed("invalid_credential");
pub const AUTH_ERROR_CODE_NOT_HANDLED: AuthErrorCode = AuthErrorCode::borrowed("not_handled");
pub const AUTH_ERROR_CODE_INTERNAL: AuthErrorCode = AuthErrorCode::borrowed("internal_error");

impl Default for AuthErrorCode {
    fn default() -> Self {
        Self::borrowed("")
    }
}

impl From<&str> for AuthErrorCode {
    fn from(code: &str) -> Self {
        Self::new(code)
    }
}

impl From<String> for AuthErrorCode {
    fn from(code: String) -> Self {
        Self(Cow::Owned(code))
    }
}

impl fmt::Display for AuthErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub type AuthErrorCause = Arc<dyn Error + Send + Sync + 'static>;

/// Carries authentication failure details and an HTTP status.
#[derive(Clone, Default)]
pub struct AuthError {
    pub code: AuthErrorCode,
    pub message: String,
    pub status_code: i32,
    pub cause: Option<AuthErrorCause>,
}

impl fmt::Debug for AuthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthError")
            .field("code", &self.code)
            .field("message", &self.message)
            .field("status_code", &self.status_code)
            .field("cause", &self.cause.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

impl AuthError {
    #[must_use]
    pub fn http_status_code(&self) -> i32 {
        auth_error_http_status_code(Some(self))
    }
}

impl fmt::Display for AuthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = self.message.trim();
        let message = if message.is_empty() {
            "authentication error"
        } else {
            message
        };
        if let Some(cause) = &self.cause {
            write!(formatter, "{message}: {cause}")
        } else {
            formatter.write_str(message)
        }
    }
}

impl Error for AuthError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.cause
            .as_deref()
            .map(|cause| cause as &(dyn Error + 'static))
    }
}

fn new_auth_error(
    code: AuthErrorCode,
    message: impl Into<String>,
    status_code: i32,
    cause: Option<AuthErrorCause>,
) -> AuthError {
    AuthError {
        code,
        message: message.into(),
        status_code,
        cause,
    }
}

#[must_use]
pub fn new_no_credentials_error() -> AuthError {
    new_auth_error(
        AUTH_ERROR_CODE_NO_CREDENTIALS,
        "Missing API key",
        HTTP_STATUS_UNAUTHORIZED,
        None,
    )
}

#[must_use]
pub fn new_invalid_credential_error() -> AuthError {
    new_auth_error(
        AUTH_ERROR_CODE_INVALID_CREDENTIAL,
        "Invalid API key",
        HTTP_STATUS_UNAUTHORIZED,
        None,
    )
}

#[must_use]
pub fn new_not_handled_error() -> AuthError {
    new_auth_error(
        AUTH_ERROR_CODE_NOT_HANDLED,
        "authentication provider did not handle request",
        0,
        None,
    )
}

#[must_use]
pub fn new_internal_auth_error(
    message: impl Into<String>,
    cause: Option<AuthErrorCause>,
) -> AuthError {
    let message = message.into();
    let normalized_message = message.trim();
    let normalized_message = if normalized_message.is_empty() {
        "Authentication service error"
    } else {
        normalized_message
    };
    new_auth_error(
        AUTH_ERROR_CODE_INTERNAL,
        normalized_message,
        HTTP_STATUS_INTERNAL_SERVER_ERROR,
        cause,
    )
}

#[must_use]
pub fn is_auth_error_code(auth_error: Option<&AuthError>, code: &AuthErrorCode) -> bool {
    auth_error.is_some_and(|auth_error| auth_error.code == *code)
}

/// Rust equivalent of calling Go's `Error` method on a possibly nil pointer.
#[must_use]
pub fn auth_error_message(auth_error: Option<&AuthError>) -> String {
    auth_error.map(ToString::to_string).unwrap_or_default()
}

/// Rust equivalent of calling Go's `Unwrap` method on a possibly nil pointer.
#[must_use]
pub fn auth_error_cause(auth_error: Option<&AuthError>) -> Option<&(dyn Error + 'static)> {
    auth_error.and_then(Error::source)
}

/// Rust equivalent of calling Go's `HTTPStatusCode` on a possibly nil pointer.
#[must_use]
pub fn auth_error_http_status_code(auth_error: Option<&AuthError>) -> i32 {
    auth_error
        .map(|auth_error| auth_error.status_code)
        .filter(|status_code| *status_code > 0)
        .unwrap_or(HTTP_STATUS_INTERNAL_SERVER_ERROR)
}
