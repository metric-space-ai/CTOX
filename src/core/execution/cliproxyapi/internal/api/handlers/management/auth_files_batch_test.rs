// ref: internal/api/handlers/management/auth_files_batch_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use super::{
    ManagementCredentialError, ManagementCredentialRecord, ManagementCredentialService,
    ManagementCredentialStore, ManagementCredentialStoreError,
};

#[derive(Debug, Default)]
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

fn record(id: &str, label: &str) -> ManagementCredentialRecord {
    ManagementCredentialRecord {
        id: id.to_owned(),
        auth_index: String::new(),
        label: label.to_owned(),
        provider: "codex".to_owned(),
        disabled: false,
        models: Vec::new(),
    }
}

#[test]
fn batch_upsert_preserves_valid_records_and_reports_invalid_items() {
    let store = Arc::new(Store::default());
    let service = ManagementCredentialService::new(store.clone());
    let result = service
        .upsert_batch(vec![record("alpha", "Alpha"), record("", "Broken")])
        .unwrap();
    assert_eq!(result.accepted, ["alpha"]);
    assert_eq!(result.failed.len(), 1);
    assert_eq!(
        result.failed[0].error,
        ManagementCredentialError::InvalidRecord
    );
    assert_eq!(store.load().unwrap().len(), 1);
}

#[test]
fn duplicate_ids_in_one_batch_fail_closed_without_overwriting_first_item() {
    let store = Arc::new(Store::default());
    let service = ManagementCredentialService::new(store.clone());
    let result = service
        .upsert_batch(vec![record("alpha", "First"), record("alpha", "Second")])
        .unwrap();
    assert_eq!(result.accepted, ["alpha"]);
    assert_eq!(result.failed.len(), 1);
    assert_eq!(store.load().unwrap()[0].label, "First");
}
