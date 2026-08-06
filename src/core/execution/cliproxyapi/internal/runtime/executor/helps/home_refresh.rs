// ref: internal/runtime/executor/helps/home_refresh.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Deserialize;

use crate::sdk::cliproxy::auth::{access_token_sha256, Auth, AuthStatus};

pub const MAX_HOME_REFRESH_PAYLOAD_BYTES: usize = 1024 * 1024;

pub trait HomeRefreshClient: Send + Sync {
    fn heartbeat_ok(&self) -> bool;

    fn get_refresh_auth<'a>(
        &'a self,
        auth_index: &'a str,
        access_token_sha256: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, HomeRefreshClientError>> + Send + 'a>>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HomeRefreshClientErrorKind {
    Other,
    Cancelled,
    DeadlineExceeded,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HomeRefreshClientError {
    message: String,
    kind: HomeRefreshClientErrorKind,
}

impl HomeRefreshClientError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: HomeRefreshClientErrorKind::Other,
        }
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: HomeRefreshClientErrorKind::Cancelled,
        }
    }

    pub fn deadline_exceeded(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: HomeRefreshClientErrorKind::DeadlineExceeded,
        }
    }

    pub fn kind(&self) -> HomeRefreshClientErrorKind {
        self.kind
    }
}

impl fmt::Display for HomeRefreshClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for HomeRefreshClientError {}

/// Explicit host-owned replacement for upstream's mutable package-global
/// Home client. Disabled mode is represented without a client; enabled mode
/// requires an injected authority.
#[derive(Clone, Default)]
pub struct HomeRefreshAuthority {
    client: Option<Arc<dyn HomeRefreshClient>>,
}

impl HomeRefreshAuthority {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn enabled(client: Arc<dyn HomeRefreshClient>) -> Self {
        Self {
            client: Some(client),
        }
    }

    /// Replaces local refresh when the injected Home authority is enabled.
    ///
    /// `Disabled` corresponds to upstream `(nil, false, nil)`. Any error from
    /// enabled mode is handled by this authority and carries an HTTP status.
    pub async fn refresh_auth_via_home(
        &self,
        auth: Option<&Auth>,
    ) -> Result<HomeRefreshDisposition, HomeStatusError> {
        let Some(client) = &self.client else {
            return Ok(HomeRefreshDisposition::Disabled);
        };
        let Some(auth) = auth else {
            return Err(HomeStatusError::new(500, "home refresh: auth is nil"));
        };
        if !client.heartbeat_ok() {
            return Err(HomeStatusError::new(503, "home control center unavailable"));
        }

        let mut auth_index = auth.index.trim().to_owned();
        if auth_index.is_empty() {
            let mut indexed = auth.clone();
            auth_index = indexed.ensure_index().trim().to_owned();
        }
        if auth_index.is_empty() {
            return Err(HomeStatusError::new(
                502,
                "home refresh: auth_index is empty",
            ));
        }

        let raw = client
            .get_refresh_auth(&auth_index, &access_token_sha256(auth))
            .await
            .map_err(|error| match error.kind() {
                HomeRefreshClientErrorKind::Cancelled => {
                    HomeStatusError::cancelled(error.to_string())
                }
                HomeRefreshClientErrorKind::DeadlineExceeded => {
                    HomeStatusError::deadline_exceeded(error.to_string())
                }
                HomeRefreshClientErrorKind::Other => {
                    HomeStatusError::new(503, "home refresh temporarily unavailable")
                }
            })?;
        if raw.len() > MAX_HOME_REFRESH_PAYLOAD_BYTES {
            return Err(HomeStatusError::new(
                502,
                "home returned invalid auth payload",
            ));
        }
        if let Some(detail) = parse_home_error(&raw) {
            let code = if detail.error_type.trim().is_empty() {
                detail.code.trim()
            } else {
                detail.error_type.trim()
            };
            let status = status_from_home_error_code(code);
            let message = match status {
                401 => "credential unauthorized",
                404 => "credential refresh target not found",
                _ => "credential refresh temporarily unavailable",
            };
            return Err(HomeStatusError::new(status, message));
        }

        let (mut updated, returned_index) = parse_home_refresh_auth(&raw)
            .map_err(|_| HomeStatusError::new(502, "home returned invalid auth payload"))?;
        if updated.disabled || updated.status == AuthStatus::Disabled {
            return Err(HomeStatusError::new(401, "credential unauthorized"));
        }
        if !returned_index.is_empty() {
            auth_index = returned_index;
        }
        updated.index = auth_index;
        let _ = updated.ensure_index();
        updated.preserve_runtime_state_from(auth);
        Ok(HomeRefreshDisposition::Refreshed(Box::new(updated)))
    }
}

impl fmt::Debug for HomeRefreshAuthority {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HomeRefreshAuthority")
            .field("enabled", &self.client.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub enum HomeRefreshDisposition {
    Disabled,
    Refreshed(Box<Auth>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HomeStatusError {
    code: u16,
    message: String,
    client_kind: HomeRefreshClientErrorKind,
}

impl HomeStatusError {
    fn new(code: u16, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            client_kind: HomeRefreshClientErrorKind::Other,
        }
    }

    fn cancelled(message: impl Into<String>) -> Self {
        Self {
            code: 0,
            message: message.into(),
            client_kind: HomeRefreshClientErrorKind::Cancelled,
        }
    }

    fn deadline_exceeded(message: impl Into<String>) -> Self {
        Self {
            code: 0,
            message: message.into(),
            client_kind: HomeRefreshClientErrorKind::DeadlineExceeded,
        }
    }

    pub fn status_code(&self) -> u16 {
        self.code
    }

    pub fn client_error_kind(&self) -> HomeRefreshClientErrorKind {
        self.client_kind
    }
}

impl fmt::Display for HomeStatusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.message.is_empty() {
            write!(formatter, "status {}", self.code)
        } else {
            formatter.write_str(&self.message)
        }
    }
}

impl std::error::Error for HomeStatusError {}

#[derive(Deserialize)]
struct HomeRefreshAuthEnvelope {
    auth: Auth,
    #[serde(default)]
    auth_index: String,
}

#[derive(Deserialize)]
struct HomeErrorEnvelope {
    error: Option<HomeErrorDetail>,
}

#[derive(Deserialize)]
struct HomeErrorDetail {
    #[serde(rename = "type", default)]
    error_type: String,
    #[serde(rename = "message", default)]
    _message: String,
    #[serde(default)]
    code: String,
}

fn parse_home_error(raw: &[u8]) -> Option<HomeErrorDetail> {
    serde_json::from_slice::<HomeErrorEnvelope>(raw)
        .ok()
        .and_then(|envelope| envelope.error)
}

fn parse_home_refresh_auth(raw: &[u8]) -> Result<(Auth, String), serde_json::Error> {
    let raw_object = serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(raw)?;
    if raw_object.contains_key("auth") {
        let envelope = serde_json::from_value::<HomeRefreshAuthEnvelope>(
            serde_json::Value::Object(raw_object),
        )?;
        return Ok((envelope.auth, envelope.auth_index.trim().to_owned()));
    }
    serde_json::from_value::<Auth>(serde_json::Value::Object(raw_object))
        .map(|auth| (auth, String::new()))
}

pub fn status_from_home_error_code(code: &str) -> u16 {
    match code.trim().to_ascii_lowercase().as_str() {
        "authentication_error"
        | "unauthorized"
        | "invalid_grant"
        | "refresh_token_expired"
        | "refresh_token_revoked"
        | "refresh_token_reused" => 401,
        "model_not_found" => 404,
        "auth_not_found"
        | "auth_unavailable"
        | "refresh_temporarily_unavailable"
        | "refresh_unsupported"
        | "home_unavailable" => 503,
        _ => 503,
    }
}
