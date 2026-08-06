// ref: internal/runtime/executor/claude_executor_auth_race_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::auth::claude::has_canonical_device_id_pool;

#[test]
fn exclusive_auth_mutation_is_the_race_boundary() {
    fn requires_exclusive(_: &mut crate::sdk::cliproxy::auth::Auth) {}
    let mut auth = crate::sdk::cliproxy::auth::Auth::default();
    requires_exclusive(&mut auth);
    assert!(!has_canonical_device_id_pool(None));
}
