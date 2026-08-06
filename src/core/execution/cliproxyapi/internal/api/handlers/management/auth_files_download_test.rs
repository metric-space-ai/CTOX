// ref: internal/api/handlers/management/auth_files_download_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth_files_patch_fields_test::{record, service_with, Store};
use super::{ManagementCredentialFieldError, ManagementCredentialService};
use std::sync::{Arc, Mutex};

#[test]
fn download_returns_only_the_secret_free_projection() {
    let (_, service) = service_with("alpha");
    let download = service
        .download_projection("alpha", &record("alpha").auth_index)
        .unwrap();
    let text = String::from_utf8(download.payload).unwrap();
    assert_eq!(download.filename, "alpha.json");
    assert!(text.contains("\"provider\": \"codex\""));
    assert!(!text.contains("token"));
    assert!(!text.contains("api_key"));
}

#[test]
fn download_rejects_path_separator_ids() {
    let unsafe_record = record("../alpha");
    let auth_index = unsafe_record.auth_index.clone();
    let service =
        ManagementCredentialService::new(Arc::new(Store(Mutex::new(vec![unsafe_record]))));
    assert_eq!(
        service
            .download_projection("../alpha", &auth_index)
            .unwrap_err(),
        ManagementCredentialFieldError::InvalidFilename
    );
}
