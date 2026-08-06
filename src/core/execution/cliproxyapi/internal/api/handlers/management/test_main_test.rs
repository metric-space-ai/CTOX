// ref: internal/api/handlers/management/test_main_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: Rust tests need no process-global web framework mode
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use super::test_store_test::{Runtime, Store};
use super::ManagementPluginService;

#[test]
fn management_plugin_test_owner_is_instance_scoped() {
    let first =
        ManagementPluginService::new(Arc::new(Store::default()), Arc::new(Runtime::default()));
    let second =
        ManagementPluginService::new(Arc::new(Store::default()), Arc::new(Runtime::default()));
    first.set_enabled("sample", true).unwrap();
    assert!(second.list().unwrap().is_empty());
}
