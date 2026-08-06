// ref: examples/plugin/request-lifecycle/go/main_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only
use super::{Admission, Lifecycle};
#[test]
fn enforces_policy_concurrency_and_completion() {
    let mut lifecycle = Lifecycle::new(1, "blocked").unwrap();
    assert_eq!(lifecycle.admit("a", b"ok"), Ok(Admission::Accepted));
    assert_eq!(lifecycle.admit("a", b"ok"), Ok(Admission::Duplicate));
    assert_eq!(lifecycle.admit("b", b"ok"), Ok(Admission::RejectedBusy));
    lifecycle.complete("a");
    assert_eq!(
        lifecycle.admit("b", b"blocked"),
        Ok(Admission::RejectedPolicy)
    );
}
