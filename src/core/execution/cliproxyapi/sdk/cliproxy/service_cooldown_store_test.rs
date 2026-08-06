// ref: sdk/cliproxy/service_cooldown_store_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use super::auth::{CooldownStateRecord, CooldownStateStore, CooldownStoreError};
use super::service_test_support::runtime_fixture;

struct TestCooldownStore;

impl CooldownStateStore for TestCooldownStore {
    fn load(&self) -> Result<Vec<CooldownStateRecord>, CooldownStoreError> {
        Ok(Vec::new())
    }

    fn save(&self, _records: &[CooldownStateRecord]) -> Result<(), CooldownStoreError> {
        Ok(())
    }
}

#[test]
fn captured_cooldown_backend_is_stable_and_policy_gated() {
    let captured: Arc<dyn CooldownStateStore> = Arc::new(TestCooldownStore);
    let fixture = runtime_fixture(Some(captured.clone()));
    let resolved = fixture
        .runtime
        .resolve_cooldown_state_store(true, false)
        .expect("captured store");
    assert!(Arc::ptr_eq(&captured, &resolved));
    assert!(fixture.runtime.generic_auth_runtime().is_some());
    assert!(fixture
        .runtime
        .resolve_cooldown_state_store(false, false)
        .is_none());
    assert!(fixture
        .runtime
        .resolve_cooldown_state_store(true, true)
        .is_none());

    let without_store = runtime_fixture(None);
    assert!(without_store.runtime.generic_auth_runtime().is_none());
}
