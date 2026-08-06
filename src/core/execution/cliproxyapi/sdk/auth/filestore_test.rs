// ref: sdk/auth/filestore_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use crate::sdk::cliproxy::auth::{Auth, AuthStore, AuthStoreError};

use super::filestore::InjectedTokenStore;

#[derive(Default)]
struct MemoryStore {
    records: Mutex<Vec<Auth>>,
    deleted: Mutex<Vec<String>>,
}

impl AuthStore for MemoryStore {
    fn list(&self) -> Result<Vec<Auth>, AuthStoreError> {
        Ok(self.records.lock().unwrap().clone())
    }

    fn save(&self, auth: &Auth) -> Result<String, AuthStoreError> {
        self.records.lock().unwrap().push(auth.clone());
        Ok(format!("ctox-secret://auth/{}", auth.id))
    }

    fn delete(&self, id: &str) -> Result<(), AuthStoreError> {
        self.deleted.lock().unwrap().push(id.to_owned());
        Ok(())
    }
}

#[test]
fn injected_store_delegates_without_filesystem_authority() {
    let backend = Arc::new(MemoryStore::default());
    let store = InjectedTokenStore::new(backend.clone());
    let mut auth = Auth::default();
    auth.id = "codex-account".to_owned();
    auth.provider = "codex".to_owned();

    assert_eq!(
        store.save(&auth).unwrap(),
        "ctox-secret://auth/codex-account"
    );
    assert_eq!(store.list().unwrap().len(), 1);
    store.delete("codex-account").unwrap();
    assert_eq!(*backend.deleted.lock().unwrap(), vec!["codex-account"]);
    assert!(!format!("{store:?}").contains("codex-account"));
}

#[test]
fn wrapper_does_not_mutate_auth_path_or_metadata() {
    let backend = Arc::new(MemoryStore::default());
    let store = InjectedTokenStore::new(backend);
    let mut auth = Auth::default();
    auth.id = "kimi-account".to_owned();
    auth.provider = "kimi".to_owned();
    auth.file_name = "/tmp/must-not-be-created.json".to_owned();
    let before = serde_json::to_value(&auth).unwrap();
    store.save(&auth).unwrap();
    assert_eq!(serde_json::to_value(&auth).unwrap(), before);
}
