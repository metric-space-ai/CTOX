// ref: sdk/auth/store_registry.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;

use crate::sdk::cliproxy::auth::AuthStore;

/// Instance-owned replacement for upstream's process-global token-store slot.
#[derive(Clone, Default)]
pub struct TokenStoreRegistry {
    store: Option<Arc<dyn AuthStore>>,
}

impl TokenStoreRegistry {
    #[must_use]
    pub fn new(store: Option<Arc<dyn AuthStore>>) -> Self {
        Self { store }
    }

    pub fn register(&mut self, store: Arc<dyn AuthStore>) {
        self.store = Some(store);
    }

    pub fn clear(&mut self) {
        self.store = None;
    }

    #[must_use]
    pub fn get(&self) -> Option<Arc<dyn AuthStore>> {
        self.store.clone()
    }
}

impl fmt::Debug for TokenStoreRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TokenStoreRegistry")
            .field("has_store", &self.store.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdk::cliproxy::auth::{Auth, AuthStoreError};

    struct Store;

    impl AuthStore for Store {
        fn list(&self) -> Result<Vec<Auth>, AuthStoreError> {
            Ok(Vec::new())
        }

        fn save(&self, _auth: &Auth) -> Result<String, AuthStoreError> {
            Ok(String::new())
        }

        fn delete(&self, _id: &str) -> Result<(), AuthStoreError> {
            Ok(())
        }
    }

    #[test]
    fn store_authority_is_explicit_and_replaceable() {
        let mut registry = TokenStoreRegistry::default();
        assert!(registry.get().is_none());
        let store: Arc<dyn AuthStore> = Arc::new(Store);
        registry.register(store.clone());
        assert!(Arc::ptr_eq(&registry.get().unwrap(), &store));
        registry.clear();
        assert!(registry.get().is_none());
    }
}
