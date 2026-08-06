// ref: internal/api/handlers/management/auth_files_filter_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use super::{
    ManagementCredentialError, ManagementCredentialFilter, ManagementCredentialRecord,
    ManagementCredentialService, ManagementCredentialStore, ManagementCredentialStoreError,
};

#[derive(Debug)]
struct Store(Mutex<Vec<ManagementCredentialRecord>>);

impl ManagementCredentialStore for Store {
    fn load(&self) -> Result<Vec<ManagementCredentialRecord>, ManagementCredentialStoreError> {
        Ok(self.0.lock().unwrap().clone())
    }

    fn replace_all(
        &self,
        records: &[ManagementCredentialRecord],
    ) -> Result<(), ManagementCredentialStoreError> {
        *self.0.lock().unwrap() = records.to_vec();
        Ok(())
    }
}

fn record(id: &str, index: &str) -> ManagementCredentialRecord {
    ManagementCredentialRecord {
        id: id.to_owned(),
        auth_index: index.to_owned(),
        label: "shared-codex.json".to_owned(),
        provider: "codex".to_owned(),
        disabled: false,
        models: Vec::new(),
    }
}

#[test]
fn list_filter_combines_name_and_auth_index() {
    let store = Arc::new(Store(Mutex::new(vec![
        record("auth-a", "idx-a"),
        record("auth-b", "idx-b"),
    ])));
    let service = ManagementCredentialService::new(store);
    let records = service
        .list(&ManagementCredentialFilter {
            name: Some("shared-codex.json".to_owned()),
            auth_index: Some("idx-b".to_owned()),
            provider: None,
        })
        .unwrap();
    assert_eq!(records, vec![record("auth-b", "idx-b")]);
}

#[test]
fn status_mutation_requires_matching_auth_index() {
    let store = Arc::new(Store(Mutex::new(vec![record("auth-b", "idx-b")])));
    let service = ManagementCredentialService::new(store.clone());
    assert_eq!(
        service.set_disabled("auth-b", "idx-a", true),
        Err(ManagementCredentialError::AuthIndexMismatch)
    );
    assert!(!store.load().unwrap()[0].disabled);
    assert!(
        service
            .set_disabled("auth-b", "idx-b", true)
            .unwrap()
            .disabled
    );
}
