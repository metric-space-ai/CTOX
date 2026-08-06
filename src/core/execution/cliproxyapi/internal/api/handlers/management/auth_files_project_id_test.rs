// ref: internal/api/handlers/management/auth_files_project_id_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth_files_patch_fields_test::service_with;
use super::{
    ManagementCredentialFilter, ManagementCredentialRuntimeDetails,
    ManagementCredentialRuntimeSource,
};

#[derive(Debug)]
struct Runtime;

impl ManagementCredentialRuntimeSource for Runtime {
    fn details(&self, auth_id: &str) -> ManagementCredentialRuntimeDetails {
        ManagementCredentialRuntimeDetails {
            project_id: (auth_id == "alpha").then(|| "project-42".to_owned()),
            websockets: true,
            recent_requests: Vec::new(),
        }
    }
}

#[test]
fn list_projects_runtime_fields_from_the_injected_authority() {
    let (_, service) = service_with("alpha");
    let views = service
        .list_views(&ManagementCredentialFilter::default(), &Runtime)
        .unwrap();
    assert_eq!(views[0].runtime.project_id.as_deref(), Some("project-42"));
    assert!(views[0].runtime.websockets);
}
