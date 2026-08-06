// ref: internal/api/handlers/management/auth_files_oauth_callback.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: typed callback dispatch through injected authorities; no loopback listener or raw-token HTTP response
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;

use super::{ManagementOAuthSessionError, ManagementOAuthSessions};

const CALLBACK_PREFIX: &str = "/management/oauth/";
const CALLBACK_SUFFIX: &str = "/callback";

#[derive(Clone, PartialEq, Eq)]
pub struct ManagementOAuthCallback {
    pub provider: String,
    pub state: String,
    pub code: String,
}

impl fmt::Debug for ManagementOAuthCallback {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementOAuthCallback")
            .field("provider", &self.provider)
            .field("state", &self.state)
            .field("code", &"[REDACTED]")
            .finish()
    }
}

pub trait ManagementOAuthCallbackSink: Send + Sync {
    fn exchange(
        &self,
        callback: &ManagementOAuthCallback,
    ) -> Result<(), ManagementOAuthCallbackSinkError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManagementOAuthCallbackSinkError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagementOAuthCallbackError {
    InvalidPath,
    InvalidCallback,
    Session(ManagementOAuthSessionError),
    ExchangeFailed,
}

impl fmt::Display for ManagementOAuthCallbackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidPath => "OAuth callback path is invalid",
            Self::InvalidCallback => "OAuth callback payload is invalid",
            Self::Session(_) => "OAuth callback session is invalid",
            Self::ExchangeFailed => "OAuth callback exchange failed",
        })
    }
}

impl std::error::Error for ManagementOAuthCallbackError {}

impl From<ManagementOAuthSessionError> for ManagementOAuthCallbackError {
    fn from(error: ManagementOAuthSessionError) -> Self {
        Self::Session(error)
    }
}

pub struct ManagementOAuthCallbacks {
    sessions: Arc<ManagementOAuthSessions>,
    sink: Arc<dyn ManagementOAuthCallbackSink>,
}

impl ManagementOAuthCallbacks {
    #[must_use]
    pub fn new(
        sessions: Arc<ManagementOAuthSessions>,
        sink: Arc<dyn ManagementOAuthCallbackSink>,
    ) -> Self {
        Self { sessions, sink }
    }

    pub fn submit(
        &self,
        path: &str,
        state: &str,
        code: Option<&str>,
        error: Option<&str>,
    ) -> Result<(), ManagementOAuthCallbackError> {
        let provider =
            oauth_callback_provider(path).ok_or(ManagementOAuthCallbackError::InvalidPath)?;
        self.sessions.guard_pending_for_save(state, &provider)?;
        if let Some(message) = error.map(str::trim).filter(|message| !message.is_empty()) {
            self.sessions.set_error(state, message)?;
            return Ok(());
        }
        let code = code
            .map(str::trim)
            .filter(|code| !code.is_empty())
            .ok_or(ManagementOAuthCallbackError::InvalidCallback)?;
        let callback = ManagementOAuthCallback {
            provider,
            state: state.trim().to_owned(),
            code: code.to_owned(),
        };
        if self.sink.exchange(&callback).is_err() {
            let _ = self
                .sessions
                .set_error(state, "Authentication exchange failed");
            return Err(ManagementOAuthCallbackError::ExchangeFailed);
        }
        if self.sessions.complete(state)? {
            Ok(())
        } else {
            Err(ManagementOAuthCallbackError::InvalidCallback)
        }
    }
}

impl fmt::Debug for ManagementOAuthCallbacks {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementOAuthCallbacks")
            .finish_non_exhaustive()
    }
}

#[must_use]
pub fn management_oauth_callback_path(provider: &str) -> Option<String> {
    let provider = provider.trim().to_ascii_lowercase();
    valid_path_provider(&provider).then(|| format!("{CALLBACK_PREFIX}{provider}{CALLBACK_SUFFIX}"))
}

#[must_use]
pub fn oauth_callback_provider(path: &str) -> Option<String> {
    let provider = path
        .strip_prefix(CALLBACK_PREFIX)?
        .strip_suffix(CALLBACK_SUFFIX)?;
    (!provider.contains('/') && valid_path_provider(provider)).then(|| provider.to_owned())
}

fn valid_path_provider(provider: &str) -> bool {
    !provider.is_empty()
        && provider
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}
