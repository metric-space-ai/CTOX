// ref: sdk/cliproxy/auth/conductor_recent_requests_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::Auth;

fn totals(auth: &Auth, now: i64) -> (i64, i64) {
    auth.recent_requests_snapshot(now)
        .into_iter()
        .fold((0, 0), |(success, failed), bucket| {
            (success + bucket.success, failed + bucket.failed)
        })
}

#[test]
fn result_records_recent_requests() {
    let now = 1_700_000_000;
    let mut auth = Auth::default();
    auth.id = "auth-1".to_owned();
    auth.provider = "antigravity".to_owned();
    auth.record_recent_request(now, true);
    auth.record_recent_request(now, false);
    auth.success += 1;
    auth.failed += 1;
    assert_eq!((auth.success, auth.failed), (1, 1));
    assert_eq!(totals(&auth, now), (1, 1));
}

#[test]
fn update_preserves_recent_requests_and_totals() {
    let now = 1_700_000_000;
    let mut existing = Auth::default();
    existing.id = "auth-1".to_owned();
    existing.provider = "antigravity".to_owned();
    existing.record_recent_request(now, true);
    existing.success = 1;

    let mut updated = Auth::default();
    updated.id = existing.id.clone();
    updated.provider = existing.provider.clone();
    updated.preserve_runtime_state_from(&existing);
    assert_eq!((updated.success, updated.failed), (1, 0));
    assert_eq!(totals(&updated, now), (1, 0));
}
