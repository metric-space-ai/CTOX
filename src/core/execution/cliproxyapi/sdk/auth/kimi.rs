// ref: sdk/auth/kimi.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, SecondsFormat, Utc};
use serde_json::Value;

use crate::internal::auth::kimi::{
    KimiAuth, KimiClock, KimiCredentialHandles, KimiSecretStore, KimiSecretStoreError,
    KimiTokenStorage, KIMI_REFRESH_THRESHOLD,
};
use crate::internal::auth::models::{shared_token_storage, TokenStorage, TokenStorageError};
use crate::sdk::cliproxy::auth::Auth;

use super::{
    Authenticator, AuthenticatorError, AuthenticatorErrorKind, LoginCancellation, LoginConfig,
    LoginFuture, LoginOptions, PromptError,
};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct KimiDevicePresentation {
    pub verification_url: String,
    pub user_code: String,
    pub expires_in_seconds: i64,
    pub automatic_browser_allowed: bool,
}

pub trait KimiLoginPresenter: Send + Sync {
    fn present(&self, challenge: &KimiDevicePresentation) -> Result<(), PromptError>;
}

pub trait KimiHandleFactory: Send + Sync {
    fn handles_for(&self, record_id: &str) -> Result<KimiCredentialHandles, KimiSecretStoreError>;
}

pub struct KimiAuthenticator {
    service: Arc<KimiAuth>,
    clock: Arc<dyn KimiClock>,
    secret_store: Arc<dyn KimiSecretStore>,
    handles: Arc<dyn KimiHandleFactory>,
    presenter: Arc<dyn KimiLoginPresenter>,
}

impl KimiAuthenticator {
    #[must_use]
    pub fn new(
        service: Arc<KimiAuth>,
        clock: Arc<dyn KimiClock>,
        secret_store: Arc<dyn KimiSecretStore>,
        handles: Arc<dyn KimiHandleFactory>,
        presenter: Arc<dyn KimiLoginPresenter>,
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

impl fmt::Debug for KimiAuthenticator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KimiAuthenticator")
            .field("service", &"[INJECTED]")
            .field("clock", &"[INJECTED]")
            .field("secret_store", &"[INJECTED]")
            .field("handles", &"[INJECTED]")
            .field("presenter", &"[INJECTED]")
            .finish()
    }
}

impl Authenticator for KimiAuthenticator {
    fn provider(&self) -> &str {
        "kimi"
    }

    fn login<'a>(
        &'a self,
        cancellation: &'a LoginCancellation,
        _config: &'a LoginConfig,
        options: &'a LoginOptions,
    ) -> LoginFuture<'a> {
        Box::pin(async move {
            let device_code = self
                .service
                .start_device_flow(cancellation)
                .await
                .map_err(auth_error)?;
            let verification_url = if device_code.verification_uri_complete.trim().is_empty() {
                device_code.verification_uri.trim()
            } else {
                device_code.verification_uri_complete.trim()
            };
            if verification_url.is_empty() {
                return Err(AuthenticatorError::new(
                    AuthenticatorErrorKind::InvalidRecord,
                ));
            }
            self.presenter
                .present(&KimiDevicePresentation {
                    verification_url: verification_url.to_owned(),
                    user_code: device_code.user_code.trim().to_owned(),
                    expires_in_seconds: device_code.expires_in,
                    automatic_browser_allowed: !options.no_browser,
                })
                .map_err(|error| {
                    AuthenticatorError::with_source(AuthenticatorErrorKind::LoginFailed, error)
                })?;
            let bundle = self
                .service
                .wait_for_authorization(cancellation, &device_code)
                .await
                .map_err(auth_error)?;
            let storage = self.service.create_token_storage(&bundle);
            let now = self.clock.now();
            let timestamp = now
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let id = format!("kimi-{timestamp}.json");
            let handles = self.handles.handles_for(&id).map_err(|error| {
                AuthenticatorError::with_source(AuthenticatorErrorKind::LoginFailed, error)
            })?;
            let adapter = KimiStorageAdapter {
                storage,
                store: self.secret_store.clone(),
                handles,
            };
            let token = bundle.token_data();
            let mut metadata = serde_json::Map::new();
            metadata.insert("type".to_owned(), Value::String("kimi".to_owned()));
            metadata.insert(
                "token_type".to_owned(),
                Value::String(token.token_type().to_owned()),
            );
            metadata.insert("scope".to_owned(), Value::String(token.scope().to_owned()));
            metadata.insert("timestamp".to_owned(), Value::from(timestamp as u64));
            if let Some(expires_at) = token.expires_at() {
                metadata.insert(
                    "expired".to_owned(),
                    Value::String(format_rfc3339(expires_at)),
                );
            }
            if !bundle.device_id().is_empty() {
                metadata.insert(
                    "device_id".to_owned(),
                    Value::String(bundle.device_id().to_owned()),
                );
            }

            let mut record = Auth::default();
            record.id.clone_from(&id);
            record.provider = "kimi".to_owned();
            record.file_name = id;
            record.label = "Kimi User".to_owned();
            record.storage = Some(shared_token_storage(adapter));
            record.metadata = metadata.into_iter().collect();
            Ok(Some(record))
        })
    }

    fn refresh_lead(&self) -> Option<Duration> {
        Some(KIMI_REFRESH_THRESHOLD)
    }
}

struct KimiStorageAdapter {
    storage: KimiTokenStorage,
    store: Arc<dyn KimiSecretStore>,
    handles: KimiCredentialHandles,
}

impl fmt::Debug for KimiStorageAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KimiStorageAdapter")
            .field("storage", &self.storage)
            .field("store", &"[INJECTED]")
            .field("handles", &self.handles)
            .finish()
    }
}

impl TokenStorage for KimiStorageAdapter {
    fn save_token_to_file(&mut self, _auth_file_path: &Path) -> Result<(), TokenStorageError> {
        self.storage
            .persist_credentials(self.store.as_ref(), &self.handles)
            .map_err(|error| Box::new(error) as TokenStorageError)
    }
}

fn auth_error(error: crate::internal::auth::kimi::KimiAuthError) -> AuthenticatorError {
    let kind = if error.kind == crate::internal::auth::kimi::KimiAuthErrorKind::Cancelled {
        AuthenticatorErrorKind::Cancelled
    } else {
        AuthenticatorErrorKind::LoginFailed
    };
    AuthenticatorError::with_source(kind, error)
}

fn format_rfc3339(value: SystemTime) -> String {
    DateTime::<Utc>::from(value).to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use crate::internal::auth::kimi::{
        DeviceFlowClient, KimiHttpFuture, KimiHttpRequest, KimiHttpResponse, KimiHttpTransport,
        KimiRefreshCoordinator, KimiSecretHandle, KimiSecretKind, KimiSleepFuture,
        KimiStoredCredentials, KimiTransportFailure,
    };

    use super::*;

    struct FixedClock(Mutex<SystemTime>);

    impl KimiClock for FixedClock {
        fn now(&self) -> SystemTime {
            *self.0.lock().unwrap()
        }

        fn sleep<'a>(
            &'a self,
            duration: Duration,
            cancellation: &'a LoginCancellation,
        ) -> KimiSleepFuture<'a> {
            Box::pin(async move {
                if cancellation.is_cancelled() {
                    return Err(KimiTransportFailure::Cancelled);
                }
                *self.0.lock().unwrap() += duration;
                Ok(())
            })
        }
    }

    struct SequenceTransport(Mutex<VecDeque<KimiHttpResponse>>);

    impl KimiHttpTransport for SequenceTransport {
        fn execute<'a>(
            &'a self,
            _request: &'a KimiHttpRequest,
            _timeout: Duration,
            _cancellation: &'a LoginCancellation,
        ) -> KimiHttpFuture<'a> {
            Box::pin(async move {
                self.0
                    .lock()
                    .unwrap()
                    .pop_front()
                    .ok_or(KimiTransportFailure::Protocol)
            })
        }
    }

    #[derive(Default)]
    struct Presenter(Mutex<Vec<KimiDevicePresentation>>);

    impl KimiLoginPresenter for Presenter {
        fn present(&self, challenge: &KimiDevicePresentation) -> Result<(), PromptError> {
            self.0.lock().unwrap().push(challenge.clone());
            Ok(())
        }
    }

    struct Handles;

    impl KimiHandleFactory for Handles {
        fn handles_for(
            &self,
            record_id: &str,
        ) -> Result<KimiCredentialHandles, KimiSecretStoreError> {
            Ok(KimiCredentialHandles {
                access: KimiSecretHandle::new(
                    format!("{record_id}/access"),
                    KimiSecretKind::Access,
                )
                .unwrap(),
                refresh: KimiSecretHandle::new(
                    format!("{record_id}/refresh"),
                    KimiSecretKind::Refresh,
                )
                .unwrap(),
            })
        }
    }

    #[derive(Default)]
    struct Store(Mutex<Option<KimiStoredCredentials>>);

    impl KimiSecretStore for Store {
        fn load_credentials(
            &self,
            _handles: &KimiCredentialHandles,
        ) -> Result<KimiStoredCredentials, KimiSecretStoreError> {
            self.0
                .lock()
                .unwrap()
                .clone()
                .ok_or(KimiSecretStoreError::Missing)
        }

        fn store_credentials(
            &self,
            _handles: &KimiCredentialHandles,
            credentials: &KimiStoredCredentials,
        ) -> Result<(), KimiSecretStoreError> {
            *self.0.lock().unwrap() = Some(credentials.clone());
            Ok(())
        }
    }

    #[tokio::test]
    async fn login_runs_full_device_flow_and_persists_only_through_secret_store() {
        let now = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let clock = Arc::new(FixedClock(Mutex::new(now)));
        let transport = Arc::new(SequenceTransport(Mutex::new(VecDeque::from([
            KimiHttpResponse::new(
                200,
                br#"{"device_code":"device-code","user_code":"USER","verification_uri_complete":"https://auth.kimi.test/verify","expires_in":60,"interval":1}"#.to_vec(),
            ),
            KimiHttpResponse::new(
                200,
                br#"{"access_token":"access-secret","refresh_token":"refresh-secret","token_type":"Bearer","scope":"openid","expires_in":3600}"#.to_vec(),
            ),
        ]))));
        let client = Arc::new(DeviceFlowClient::new(
            transport,
            clock.clone(),
            crate::internal::auth::kimi::KimiDeviceIdentity::new(
                "device", "host", "model", "version",
            )
            .unwrap(),
            Arc::new(KimiRefreshCoordinator::default()),
        ));
        let presenter = Arc::new(Presenter::default());
        let store = Arc::new(Store::default());
        let authenticator = KimiAuthenticator::new(
            Arc::new(KimiAuth::new(client)),
            clock,
            store.clone(),
            Arc::new(Handles),
            presenter.clone(),
        );

        let record = authenticator
            .login(
                &LoginCancellation::default(),
                &LoginConfig::default(),
                &LoginOptions {
                    no_browser: true,
                    ..LoginOptions::default()
                },
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(record.provider, "kimi");
        assert!(record.id.starts_with("kimi-"));
        assert!(!record.metadata.contains_key("access_token"));
        assert!(!record.metadata.contains_key("refresh_token"));
        assert_eq!(presenter.0.lock().unwrap()[0].user_code, "USER");
        assert!(!presenter.0.lock().unwrap()[0].automatic_browser_allowed);

        record
            .storage
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .save_token_to_file(Path::new("ignored-plaintext-path"))
            .unwrap();
        let stored = store.0.lock().unwrap().clone().unwrap();
        assert_eq!(stored.access_token().expose_secret(), "access-secret");
        assert_eq!(
            stored.refresh_token().unwrap().expose_secret(),
            "refresh-secret"
        );
        assert!(!format!("{authenticator:?}").contains("access-secret"));
    }
}
