// ref: internal/api/handlers/management/auth_files_download_windows_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth_files_patch_fields_test::{record, Store};
use super::{ManagementCredentialFieldError, ManagementCredentialService};
use std::sync::{Arc, Mutex};

#[test]
fn download_rejects_windows_backslash_traversal_on_every_platform() {
    let unsafe_record = record("folder\\secret");
    let auth_index = unsafe_record.auth_index.clone();
    let service =
        ManagementCredentialService::new(Arc::new(Store(Mutex::new(vec![unsafe_record]))));
    assert_eq!(
        service
            .download_projection("folder\\secret", &auth_index)
            .unwrap_err(),
        ManagementCredentialFieldError::InvalidFilename
    );
}
