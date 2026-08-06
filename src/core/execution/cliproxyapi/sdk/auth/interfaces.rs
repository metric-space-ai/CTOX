// ref: sdk/auth/interfaces.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;

use crate::sdk::cliproxy::auth::Auth;

pub type LoginFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<Auth>, AuthenticatorError>> + Send + 'a>>;
pub type PromptCallback = Arc<dyn Fn(&str) -> Result<String, PromptError> + Send + Sync + 'static>;

/// Typed configuration required by SDK login flows. The host owns the actual
/// runtime configuration; the SDK does not read process environment or mutate
/// a token store's base directory during login.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct LoginConfig {
    pub auth_dir: PathBuf,
}

#[derive(Clone, Default)]
pub struct LoginOptions {
    pub no_browser: bool,
    pub project_id: String,
    pub callback_port: u16,
    pub metadata: BTreeMap<String, String>,
    pub prompt: Option<PromptCallback>,
}

impl fmt::Debug for LoginOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LoginOptions")
            .field("no_browser", &self.no_browser)
            .field("has_project_id", &!self.project_id.is_empty())
            .field("callback_port", &self.callback_port)
            .field("metadata_keys", &self.metadata.keys().collect::<Vec<_>>())
            .field("has_prompt", &self.prompt.is_some())
            .finish()
    }
}

/// Request-local replacement for Go context cancellation.
#[derive(Clone, Debug)]
pub struct LoginCancellation {
    sender: watch::Sender<bool>,
    receiver: watch::Receiver<bool>,
}

impl Default for LoginCancellation {
    fn default() -> Self {
        let (sender, receiver) = watch::channel(false);
        Self { sender, receiver }
    }
}

impl LoginCancellation {
    pub fn cancel(&self) {
        self.sender.send_replace(true);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        *self.receiver.borrow()
    }

    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let mut receiver = self.receiver.clone();
        while receiver.changed().await.is_ok() {
            if *receiver.borrow() {
                return;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthenticatorErrorKind {
    Cancelled,
    LoginFailed,
    InvalidRecord,
    RefreshNotSupported,
}

#[derive(Clone)]
pub struct AuthenticatorError {
    pub kind: AuthenticatorErrorKind,
    pub source: Option<Arc<dyn Error + Send + Sync + 'static>>,
}

impl AuthenticatorError {
    #[must_use]
    pub fn new(kind: AuthenticatorErrorKind) -> Self {
        Self { kind, source: None }
    }

    #[must_use]
    pub fn with_source(
        kind: AuthenticatorErrorKind,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind,
            source: Some(Arc::new(source)),
        }
    }
}

impl fmt::Debug for AuthenticatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthenticatorError")
            .field("kind", &self.kind)
            .field("has_source", &self.source.is_some())
            .finish()
    }
}

impl fmt::Display for AuthenticatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            AuthenticatorErrorKind::Cancelled => "authentication cancelled",
            AuthenticatorErrorKind::LoginFailed => "authentication login failed",
            AuthenticatorErrorKind::InvalidRecord => "authenticator returned an invalid record",
            AuthenticatorErrorKind::RefreshNotSupported => "cliproxy auth: refresh not supported",
        })
    }
}

impl Error for AuthenticatorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn Error + 'static))
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PromptError;

impl fmt::Display for PromptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("authentication prompt failed")
    }
}

impl Error for PromptError {}

pub trait Authenticator: Send + Sync {
    fn provider(&self) -> &str;

    fn login<'a>(
        &'a self,
        cancellation: &'a LoginCancellation,
        config: &'a LoginConfig,
        options: &'a LoginOptions,
    ) -> LoginFuture<'a>;

    fn refresh_lead(&self) -> Option<Duration> {
        None
    }
}
