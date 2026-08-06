// ref: sdk/auth/xai.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use chrono::{DateTime, SecondsFormat, Utc};
use serde_json::Value;

use crate::internal::auth::models::{
    shared_token_storage, TokenStorage as TokenStorageTrait, TokenStorageError,
};
use crate::internal::auth::xai::{
    credential_file_name, TokenStorage, XaiAuth, XaiAuthErrorKind, XaiClock, XaiCredentialHandles,
    XaiSecretStore, XaiSecretStoreError, REFRESH_LEAD,
};
use crate::sdk::cliproxy::auth::Auth;

use super::{
    Authenticator, AuthenticatorError, AuthenticatorErrorKind, LoginCancellation, LoginConfig,
    LoginFuture, LoginOptions, PromptError,
};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct XaiDevicePresentation {
    pub verification_url: String,
    pub user_code: String,
    pub expires_in_seconds: i64,
    pub automatic_browser_allowed: bool,
}

pub trait XaiLoginPresenter: Send + Sync {
    fn present(&self, challenge: &XaiDevicePresentation) -> Result<(), PromptError>;
}

pub trait XaiHandleFactory: Send + Sync {
    fn handles_for(&self, record_id: &str) -> Result<XaiCredentialHandles, XaiSecretStoreError>;
}

pub struct XaiAuthenticator {
    service: Arc<XaiAuth>,
    clock: Arc<dyn XaiClock>,
    secret_store: Arc<dyn XaiSecretStore>,
    handles: Arc<dyn XaiHandleFactory>,
    presenter: Arc<dyn XaiLoginPresenter>,
}

impl XaiAuthenticator {
    pub fn new(
        service: Arc<XaiAuth>,
        clock: Arc<dyn XaiClock>,
        secret_store: Arc<dyn XaiSecretStore>,
        handles: Arc<dyn XaiHandleFactory>,
        presenter: Arc<dyn XaiLoginPresenter>,
    ) -> Self {
        Self {
            service,
            clock,
            secret_store,
            handles,
            presenter,
        }
    }
}

impl fmt::Debug for XaiAuthenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XaiAuthenticator")
            .field("service", &"[INJECTED]")
            .field("clock", &"[INJECTED]")
            .field("secret_store", &"[INJECTED]")
            .field("handles", &"[INJECTED]")
            .field("presenter", &"[INJECTED]")
            .finish()
    }
}

impl Authenticator for XaiAuthenticator {
    fn provider(&self) -> &str {
        "xai"
    }

    fn login<'a>(
        &'a self,
        cancellation: &'a LoginCancellation,
        _config: &'a LoginConfig,
        options: &'a LoginOptions,
    ) -> LoginFuture<'a> {
        Box::pin(async move {
            let code = self
                .service
                .start_device_flow(cancellation)
                .await
                .map_err(auth_error)?;
            let verification_url = if code.verification_uri_complete.trim().is_empty() {
                code.verification_uri.trim()
            } else {
                code.verification_uri_complete.trim()
            };
            if verification_url.is_empty() {
                return Err(AuthenticatorError::new(
                    AuthenticatorErrorKind::InvalidRecord,
                ));
            }
            self.presenter
                .present(&XaiDevicePresentation {
                    verification_url: verification_url.to_owned(),
                    user_code: code.user_code.trim().to_owned(),
                    expires_in_seconds: code.expires_in,
                    automatic_browser_allowed: !options.no_browser,
                })
                .map_err(|error| {
                    AuthenticatorError::with_source(AuthenticatorErrorKind::LoginFailed, error)
                })?;
            let bundle = self
                .service
                .wait_for_authorization(cancellation, &code)
                .await
                .map_err(auth_error)?;
            let storage = self
                .service
                .create_token_storage(Some(&bundle))
                .ok_or_else(|| AuthenticatorError::new(AuthenticatorErrorKind::InvalidRecord))?;
            let now = self.clock.now();
            let millis = now
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let id = credential_file_name(storage.email(), storage.subject(), millis);
            let handles = self.handles.handles_for(&id).map_err(|error| {
                AuthenticatorError::with_source(AuthenticatorErrorKind::LoginFailed, error)
            })?;
            let adapter = XaiStorageAdapter {
                storage: storage.clone(),
                store: self.secret_store.clone(),
                handles,
            };
            let mut metadata = std::collections::BTreeMap::new();
            metadata.insert("type".to_owned(), Value::String("xai".to_owned()));
            metadata.insert("auth_kind".to_owned(), Value::String("oauth".to_owned()));
            metadata.insert(
                "token_type".to_owned(),
                Value::String(storage.token_type().to_owned()),
            );
            metadata.insert("expires_in".to_owned(), Value::from(storage.expires_in()));
            metadata.insert(
                "last_refresh".to_owned(),
                Value::String(format_time(storage.last_refresh())),
            );
            metadata.insert(
                "base_url".to_owned(),
                Value::String(storage.base_url().to_owned()),
            );
            metadata.insert(
                "token_endpoint".to_owned(),
                Value::String(storage.token_endpoint().to_owned()),
            );
            if let Some(expiry) = storage.expires_at() {
                metadata.insert("expired".to_owned(), Value::String(format_time(expiry)));
            }
            if !storage.email().is_empty() {
                metadata.insert(
                    "email".to_owned(),
                    Value::String(storage.email().to_owned()),
                );
            }
            if !storage.subject().is_empty() {
                metadata.insert(
                    "sub".to_owned(),
                    Value::String(storage.subject().to_owned()),
                );
            }
            let mut record = Auth::default();
            record.id.clone_from(&id);
            record.provider = "xai".to_owned();
            record.file_name = id;
            record.label = if storage.email().is_empty() {
                "xAI".to_owned()
            } else {
                storage.email().to_owned()
            };
            record.storage = Some(shared_token_storage(adapter));
            record.metadata = metadata;
            record
                .attributes
                .insert("auth_kind".to_owned(), "oauth".to_owned());
            record
                .attributes
                .insert("base_url".to_owned(), storage.base_url().to_owned());
            Ok(Some(record))
        })
    }

    fn refresh_lead(&self) -> Option<Duration> {
        Some(REFRESH_LEAD)
    }
}

#[derive(Clone)]
struct XaiStorageAdapter {
    storage: TokenStorage,
    store: Arc<dyn XaiSecretStore>,
    handles: XaiCredentialHandles,
}
impl TokenStorageTrait for XaiStorageAdapter {
    fn save_token_to_file(&mut self, _path: &Path) -> Result<(), TokenStorageError> {
        self.storage
            .persist_credentials(self.store.as_ref(), &self.handles)
            .map_err(|error| Box::new(error) as TokenStorageError)
    }
}

fn auth_error(error: crate::internal::auth::xai::XaiAuthError) -> AuthenticatorError {
    let kind = if error.kind == XaiAuthErrorKind::Cancelled {
        AuthenticatorErrorKind::Cancelled
    } else {
        AuthenticatorErrorKind::LoginFailed
    };
    AuthenticatorError::with_source(kind, error)
}

fn format_time(value: std::time::SystemTime) -> String {
    DateTime::<Utc>::from(value).to_rfc3339_opts(SecondsFormat::Secs, true)
}
