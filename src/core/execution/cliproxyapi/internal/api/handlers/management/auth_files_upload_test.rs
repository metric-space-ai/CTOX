// ref: internal/api/handlers/management/auth_files_upload_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth_files_patch_fields_test::Store;
use super::{ManagementCredentialFieldError, ManagementCredentialService};
use std::sync::Arc;

#[test]
fn upload_preserves_typed_public_attributes() {
    let store = Arc::new(Store::default());
    let service = ManagementCredentialService::new(store);
    let payload = br#"{"id":"alpha","auth_index":"","label":"ignored","provider":"codex","disabled":true,"models":["gpt-5","gpt-5"]}"#;
    let uploaded = service.upload_projection("alpha.json", payload).unwrap();
    assert_eq!(uploaded.label, "alpha.json");
    assert!(uploaded.disabled);
    assert_eq!(uploaded.models, ["gpt-5"]);
    assert!(!uploaded.auth_index.is_empty());
}

#[test]
fn upload_rejects_raw_or_unknown_auth_file_attributes() {
    let service = ManagementCredentialService::new(Arc::new(Store::default()));
    let payload = br#"{"id":"alpha","label":"alpha","provider":"codex","priority":10}"#;
    assert_eq!(
        service
            .upload_projection("alpha.json", payload)
            .unwrap_err(),
        ManagementCredentialFieldError::InvalidPayload
    );
}
