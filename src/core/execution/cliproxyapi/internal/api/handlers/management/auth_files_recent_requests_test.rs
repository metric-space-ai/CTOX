// ref: internal/api/handlers/management/auth_files_recent_requests_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth_files_patch_fields_test::service_with;
use super::{
    ManagementCredentialFilter, ManagementCredentialRuntimeDetails,
    ManagementCredentialRuntimeSource, ManagementRecentRequestBucket,
};

#[derive(Debug)]
struct Runtime;

impl ManagementCredentialRuntimeSource for Runtime {
    fn details(&self, _: &str) -> ManagementCredentialRuntimeDetails {
        ManagementCredentialRuntimeDetails {
            recent_requests: vec![ManagementRecentRequestBucket {
                time: "2026-08-04T10:00:00Z".to_owned(),
                success: 3,
                failed: 1,
            }],
            ..Default::default()
        }
    }
}

#[test]
fn list_keeps_recent_request_buckets_in_the_runtime_projection() {
    let (_, service) = service_with("alpha");
    let views = service
        .list_views(&ManagementCredentialFilter::default(), &Runtime)
        .unwrap();
    assert_eq!(
        views[0].runtime.recent_requests,
        [ManagementRecentRequestBucket {
            time: "2026-08-04T10:00:00Z".to_owned(),
            success: 3,
            failed: 1,
        }]
    );
}
