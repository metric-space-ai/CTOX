// ref: sdk/auth/filestore.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: replaced_by_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;

use crate::sdk::cliproxy::auth::{Auth, AuthStore, AuthStoreError};

/// CTOX replacement for upstream's process-global plaintext `FileTokenStore`.
///
/// The host injects its existing encrypted/typed auth store. This facade keeps
/// the SDK persistence contract without gaining filesystem, directory or
/// credential ownership inside the port.
#[derive(Clone)]
pub struct InjectedTokenStore {
    inner: Arc<dyn AuthStore>,
}

impl InjectedTokenStore {
    #[must_use]
    pub fn new(inner: Arc<dyn AuthStore>) -> Self {
        Self { inner }
    }
}

impl fmt::Debug for InjectedTokenStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InjectedTokenStore")
            .field("backend", &"[INJECTED]")
            .finish()
    }
}

impl AuthStore for InjectedTokenStore {
    fn list(&self) -> Result<Vec<Auth>, AuthStoreError> {
        self.inner.list()
    }

    fn save(&self, auth: &Auth) -> Result<String, AuthStoreError> {
        self.inner.save(auth)
    }

    fn delete(&self, id: &str) -> Result<(), AuthStoreError> {
        self.inner.delete(id)
    }
}
