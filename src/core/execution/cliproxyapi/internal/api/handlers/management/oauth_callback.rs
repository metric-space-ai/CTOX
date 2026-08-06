// ref: internal/api/handlers/management/oauth_callback.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;

use serde::Deserialize;
use url::Url;

use super::{
    management_oauth_callback_path, normalize_oauth_provider, normalize_plugin_oauth_provider,
    validate_oauth_state, ManagementOAuthCallbackError, ManagementOAuthCallbacks,
    ManagementOAuthSessionError, ManagementOAuthSessionSource, ManagementOAuthSessions,
};

#[derive(Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagementOAuthCallbackRequest {
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub redirect_url: String,
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub state: String,
    #[serde(default)]
    pub error: String,
}

impl fmt::Debug for ManagementOAuthCallbackRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementOAuthCallbackRequest")
            .field("provider", &self.provider)
            .field("redirect_url", &"[REDACTED]")
            .field("code", &"[REDACTED]")
            .field("state", &self.state)
            .field("error", &self.error)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagementOAuthCallbackRequestError {
    InvalidBody,
    InvalidRedirectUrl,
    InvalidState,
    MissingResult,
    UnknownSession,
    CompletedSession,
    UnsupportedProvider,
    ProviderMismatch,
    SessionConflict,
    CallbackFailed,
}

pub struct ManagementOAuthCallbackHandler {
    sessions: Arc<ManagementOAuthSessions>,
    callbacks: Arc<ManagementOAuthCallbacks>,
}

impl ManagementOAuthCallbackHandler {
    #[must_use]
    pub fn new(
        sessions: Arc<ManagementOAuthSessions>,
        callbacks: Arc<ManagementOAuthCallbacks>,
    ) -> Self {
        Self {
            sessions,
            callbacks,
        }
    }

    pub fn submit_json(&self, body: &[u8]) -> Result<(), ManagementOAuthCallbackRequestError> {
        let request = serde_json::from_slice(body)
            .map_err(|_| ManagementOAuthCallbackRequestError::InvalidBody)?;
        self.submit(request)
    }

    pub fn submit(
        &self,
        mut request: ManagementOAuthCallbackRequest,
    ) -> Result<(), ManagementOAuthCallbackRequestError> {
        request.provider = request.provider.trim().to_owned();
        request.state = request.state.trim().to_owned();
        request.code = request.code.trim().to_owned();
        request.error = request.error.trim().to_owned();
        if !request.redirect_url.trim().is_empty() {
            merge_redirect(&mut request)?;
        }
        validate_oauth_state(&request.state)
            .map_err(|_| ManagementOAuthCallbackRequestError::InvalidState)?;
        if request.code.is_empty() && request.error.is_empty() {
            return Err(ManagementOAuthCallbackRequestError::MissingResult);
        }
        let session = self
            .sessions
            .details(&request.state)
            .map_err(map_session_error)?
            .ok_or(ManagementOAuthCallbackRequestError::UnknownSession)?;
        if session.completed {
            return Err(ManagementOAuthCallbackRequestError::CompletedSession);
        }
        if !session.status.is_empty() {
            return Err(ManagementOAuthCallbackRequestError::SessionConflict);
        }
        let provider = if request.provider.is_empty() {
            session.provider.clone()
        } else {
            request.provider
        };
        let provider = match session.source {
            ManagementOAuthSessionSource::Builtin => normalize_oauth_provider(&provider),
            ManagementOAuthSessionSource::Plugin => normalize_plugin_oauth_provider(&provider),
        }
        .map_err(|_| ManagementOAuthCallbackRequestError::UnsupportedProvider)?;
        if provider != session.provider {
            return Err(ManagementOAuthCallbackRequestError::ProviderMismatch);
        }
        let path = management_oauth_callback_path(&provider)
            .ok_or(ManagementOAuthCallbackRequestError::UnsupportedProvider)?;
        self.callbacks
            .submit(
                &path,
                &request.state,
                (!request.code.is_empty()).then_some(request.code.as_str()),
                (!request.error.is_empty()).then_some(request.error.as_str()),
            )
            .map_err(map_callback_error)
    }
}

impl fmt::Debug for ManagementOAuthCallbackHandler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementOAuthCallbackHandler")
            .finish_non_exhaustive()
    }
}

fn merge_redirect(
    request: &mut ManagementOAuthCallbackRequest,
) -> Result<(), ManagementOAuthCallbackRequestError> {
    let url = Url::parse(request.redirect_url.trim())
        .map_err(|_| ManagementOAuthCallbackRequestError::InvalidRedirectUrl)?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(ManagementOAuthCallbackRequestError::InvalidRedirectUrl);
    }
    for (key, value) in url.query_pairs() {
        let value = value.trim();
        match key.as_ref() {
            "state" if request.state.is_empty() => request.state = value.to_owned(),
            "code" if request.code.is_empty() => request.code = value.to_owned(),
            "error" | "error_description" if request.error.is_empty() => {
                request.error = value.to_owned()
            }
            _ => {}
        }
    }
    Ok(())
}

fn map_session_error(_: ManagementOAuthSessionError) -> ManagementOAuthCallbackRequestError {
    ManagementOAuthCallbackRequestError::SessionConflict
}

fn map_callback_error(error: ManagementOAuthCallbackError) -> ManagementOAuthCallbackRequestError {
    match error {
        ManagementOAuthCallbackError::Session(_) => {
            ManagementOAuthCallbackRequestError::SessionConflict
        }
        ManagementOAuthCallbackError::InvalidPath
        | ManagementOAuthCallbackError::InvalidCallback
        | ManagementOAuthCallbackError::ExchangeFailed => {
            ManagementOAuthCallbackRequestError::CallbackFailed
        }
    }
}
