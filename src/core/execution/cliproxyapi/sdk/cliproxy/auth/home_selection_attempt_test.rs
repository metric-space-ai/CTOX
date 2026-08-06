// ref: sdk/cliproxy/auth/home_selection_attempt_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: attempt leases release idempotently and are cancelled by selection teardown
// License: MIT (upstream); modifications AGPL-3.0-only

#[test]
fn attempt_release_is_idempotent_and_removes_tracking() {
    let (selection, _registry) = super::home_selection_test::selection();
    let attempt = selection.attempt().unwrap();
    assert_eq!(selection.attempt_count(), 1);
    attempt.release();
    attempt.release();
    assert!(attempt.cancelled());
    assert_eq!(selection.attempt_count(), 0);
    selection.end("done");
}

#[test]
fn ending_selection_cancels_every_live_attempt() {
    let (selection, _registry) = super::home_selection_test::selection();
    let first = selection.attempt().unwrap();
    let second = selection.attempt().unwrap();
    selection.end("closed");
    assert!(first.cancelled());
    assert!(second.cancelled());
    assert_eq!(selection.attempt_count(), 0);
    assert!(selection.attempt().is_err());
}
