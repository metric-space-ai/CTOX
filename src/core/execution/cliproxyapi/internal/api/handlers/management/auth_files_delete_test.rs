// ref: internal/api/handlers/management/auth_files_delete_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use super::{
    ManagementCredentialError, ManagementCredentialRecord, ManagementCredentialService,
    ManagementCredentialStore, ManagementCredentialStoreError,
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

fn record(id: &str) -> ManagementCredentialRecord {
    ManagementCredentialRecord {
        id: id.to_owned(),
        auth_index: format!("index-{id}"),
        label: format!("{id}.json"),
        provider: "codex".to_owned(),
        disabled: false,
        models: Vec::new(),
    }
}

#[test]
fn batch_delete_uses_stable_ids_and_returns_partial_failures() {
    let store = Arc::new(Store(Mutex::new(vec![record("alpha"), record("beta")])));
    let service = ManagementCredentialService::new(store.clone());
    let result = service
        .delete_batch(&["alpha".to_owned(), "missing".to_owned()])
        .unwrap();
    assert_eq!(result.accepted, ["alpha"]);
    assert_eq!(result.failed[0].error, ManagementCredentialError::NotFound);
    assert_eq!(store.load().unwrap(), vec![record("beta")]);
}
