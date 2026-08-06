// ref: sdk/cliproxy/auth/home_session_alias_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: alias reconciliation, expiry and clear behavior
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::Duration;

use super::HomeSessionAliasCache;

#[test]
fn aliases_reconcile_to_one_canonical_session_until_expiry() {
    let cache = HomeSessionAliasCache::default();
    let ttl = Duration::from_millis(100);
    assert_eq!(
        cache.canonical("primary", "fallback", ttl, 0).as_deref(),
        Some("primary")
    );
    assert_eq!(
        cache.canonical("fallback", "third", ttl, 50).as_deref(),
        Some("primary")
    );
    assert_eq!(
        cache.canonical("fallback", "", ttl, 151).as_deref(),
        Some("fallback")
    );
    cache.clear();
    assert!(cache.is_empty());
}
