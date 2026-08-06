// ref: internal/api/handlers/management/auth_files_patch_fields_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use super::{
    ManagementCredentialFieldError, ManagementCredentialPatch, ManagementCredentialRecord,
    ManagementCredentialService, ManagementCredentialStore, ManagementCredentialStoreError,
};

#[derive(Debug, Default)]
pub(super) struct Store(pub(super) Mutex<Vec<ManagementCredentialRecord>>);

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

pub(super) fn record(id: &str) -> ManagementCredentialRecord {
    ManagementCredentialRecord {
        id: id.to_owned(),
        auth_index: super::management_auth_index_for_id(id).unwrap(),
        label: format!("{id}.json"),
        provider: "codex".to_owned(),
        disabled: false,
        models: vec!["old".to_owned()],
    }
}

pub(super) fn service_with(id: &str) -> (Arc<Store>, ManagementCredentialService) {
    let store = Arc::new(Store(Mutex::new(vec![record(id)])));
    let service = ManagementCredentialService::new(store.clone());
    (store, service)
}

#[test]
fn patch_merges_typed_fields_and_applies_explicit_false() {
    let (store, service) = service_with("alpha");
    let auth_index = record("alpha").auth_index;
    let patched = service
        .patch_fields(
            "alpha",
            &auth_index,
            ManagementCredentialPatch {
                label: Some("renamed.json".to_owned()),
                disabled: Some(false),
                models: Some(vec![
                    " zeta ".to_owned(),
                    String::new(),
                    "alpha".to_owned(),
                    "zeta".to_owned(),
                ]),
            },
        )
        .unwrap();
    assert_eq!(patched.label, "renamed.json");
    assert!(!patched.disabled);
    assert_eq!(patched.models, ["alpha", "zeta"]);
    assert_eq!(store.0.lock().unwrap()[0], patched);
}

#[test]
fn patch_rejects_auth_index_mismatch_without_writing() {
    let (store, service) = service_with("alpha");
    let before = store.0.lock().unwrap().clone();
    assert!(service
        .patch_fields(
            "alpha",
            "wrong",
            ManagementCredentialPatch {
                disabled: Some(true),
                ..Default::default()
            }
        )
        .is_err());
    assert_eq!(*store.0.lock().unwrap(), before);
}

#[test]
fn patch_rejects_path_like_labels() {
    let (_, service) = service_with("alpha");
    let error = service
        .patch_fields(
            "alpha",
            &record("alpha").auth_index,
            ManagementCredentialPatch {
                label: Some("../secret.json".to_owned()),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(error, ManagementCredentialFieldError::InvalidFilename);
}
