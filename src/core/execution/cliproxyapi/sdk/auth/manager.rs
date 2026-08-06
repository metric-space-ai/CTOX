// ref: sdk/auth/manager.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::sdk::cliproxy::auth::{Auth, AuthStore, AuthStoreError};

use super::{
    Authenticator, AuthenticatorError, AuthenticatorErrorKind, LoginCancellation, LoginConfig,
    LoginOptions,
};

#[derive(Default)]
pub struct Manager {
    authenticators: BTreeMap<String, Arc<dyn Authenticator>>,
    store: Option<Arc<dyn AuthStore>>,
}

impl fmt::Debug for Manager {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Manager")
            .field("providers", &self.authenticators.keys().collect::<Vec<_>>())
            .field("has_store", &self.store.is_some())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagerErrorKind {
    AuthenticatorNotRegistered,
    Authentication,
    InvalidRecord,
    Store,
}

#[derive(Clone)]
pub struct ManagerError {
    pub kind: ManagerErrorKind,
    pub provider: String,
    pub authentication: Option<AuthenticatorError>,
    pub store: Option<AuthStoreError>,
}

impl fmt::Debug for ManagerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagerError")
            .field("kind", &self.kind)
            .field("provider", &self.provider)
            .field("has_authentication_error", &self.authentication.is_some())
            .field("store", &self.store)
            .finish()
    }
}

impl fmt::Display for ManagerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            ManagerErrorKind::AuthenticatorNotRegistered => "authenticator not registered",
            ManagerErrorKind::Authentication => "authentication failed",
            ManagerErrorKind::InvalidRecord => "authenticator returned an invalid record",
            ManagerErrorKind::Store => "auth record persistence failed",
        })
    }
}

impl std::error::Error for ManagerError {}

impl Manager {
    #[must_use]
    pub fn new(
        store: Option<Arc<dyn AuthStore>>,
        authenticators: impl IntoIterator<Item = Arc<dyn Authenticator>>,
    ) -> Self {
        let mut manager = Self {
            authenticators: BTreeMap::new(),
            store,
        };
        for authenticator in authenticators {
            manager.register(authenticator);
        }
        manager
    }

    pub fn register(&mut self, authenticator: Arc<dyn Authenticator>) {
        let provider = authenticator.provider().trim().to_owned();
        if !provider.is_empty() {
            self.authenticators.insert(provider, authenticator);
        }
    }

    pub fn set_store(&mut self, store: Option<Arc<dyn AuthStore>>) {
        self.store = store;
    }

    pub async fn login(
        &self,
        cancellation: &LoginCancellation,
        provider: &str,
        config: &LoginConfig,
        options: &LoginOptions,
    ) -> Result<(Auth, String), ManagerError> {
        let Some(authenticator) = self.authenticators.get(provider) else {
            return Err(ManagerError {
                kind: ManagerErrorKind::AuthenticatorNotRegistered,
                provider: provider.to_owned(),
                authentication: None,
                store: None,
            });
        };
        let record = authenticator
            .login(cancellation, config, options)
            .await
            .map_err(|error| ManagerError {
                kind: ManagerErrorKind::Authentication,
                provider: provider.to_owned(),
                authentication: Some(error),
                store: None,
            })?
            .ok_or_else(|| ManagerError {
                kind: ManagerErrorKind::InvalidRecord,
                provider: provider.to_owned(),
                authentication: Some(AuthenticatorError::new(
                    AuthenticatorErrorKind::InvalidRecord,
                )),
                store: None,
            })?;
        let Some(store) = &self.store else {
            return Ok((record, String::new()));
        };
        let path = store.save(&record).map_err(|error| ManagerError {
            kind: ManagerErrorKind::Store,
            provider: provider.to_owned(),
            authentication: None,
            store: Some(error),
        })?;
        Ok((record, path))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::sdk::auth::LoginFuture;

    struct FixedAuthenticator {
        provider: &'static str,
        record: Option<Auth>,
    }

    impl Authenticator for FixedAuthenticator {
        fn provider(&self) -> &str {
            self.provider
        }

        fn login<'a>(
            &'a self,
            _cancellation: &'a LoginCancellation,
            _config: &'a LoginConfig,
            _options: &'a LoginOptions,
        ) -> LoginFuture<'a> {
            Box::pin(async { Ok(self.record.clone()) })
        }
    }

    #[derive(Default)]
    struct RecordingStore(Mutex<Vec<String>>);

    impl AuthStore for RecordingStore {
        fn list(&self) -> Result<Vec<Auth>, AuthStoreError> {
            Ok(Vec::new())
        }

        fn save(&self, auth: &Auth) -> Result<String, AuthStoreError> {
            self.0.lock().unwrap().push(auth.provider.clone());
            Ok("ctox-secret://auth/codex".to_owned())
        }

        fn delete(&self, _id: &str) -> Result<(), AuthStoreError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn login_selects_provider_and_persists_through_injected_store() {
        let store = Arc::new(RecordingStore::default());
        let mut record = Auth::default();
        record.id = "account-1".to_owned();
        record.provider = "codex".to_owned();
        let authenticator: Arc<dyn Authenticator> = Arc::new(FixedAuthenticator {
            provider: "codex",
            record: Some(record),
        });
        let manager = Manager::new(Some(store.clone()), [authenticator]);
        let (record, saved_path) = manager
            .login(
                &LoginCancellation::default(),
                "codex",
                &LoginConfig::default(),
                &LoginOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(record.id, "account-1");
        assert_eq!(saved_path, "ctox-secret://auth/codex");
        assert_eq!(*store.0.lock().unwrap(), vec!["codex"]);
    }

    #[tokio::test]
    async fn missing_provider_and_invalid_record_are_distinct() {
        let manager = Manager::default();
        let missing = manager
            .login(
                &LoginCancellation::default(),
                "missing",
                &LoginConfig::default(),
                &LoginOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(missing.kind, ManagerErrorKind::AuthenticatorNotRegistered);

        let invalid: Arc<dyn Authenticator> = Arc::new(FixedAuthenticator {
            provider: "codex",
            record: None,
        });
        let manager = Manager::new(None, [invalid]);
        let invalid = manager
            .login(
                &LoginCancellation::default(),
                "codex",
                &LoginConfig::default(),
                &LoginOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(invalid.kind, ManagerErrorKind::InvalidRecord);
    }
}
