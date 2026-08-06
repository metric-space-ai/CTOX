// ref: internal/runtime/executor/xai_status_err_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::xai_executor::xai_status_error;
use std::time::Duration;

#[test]
fn free_usage_exhausted_sets_24h_retry_after() {
    let error = xai_status_error(
        429,
        br#"{"error":{"message":"Free usage quota exhausted"}}"#,
    );
    assert_eq!(error.retry_after, Some(Duration::from_secs(86_400)));
}
#[test]
fn generic_429_has_no_retry_after() {
    assert_eq!(xai_status_error(429, b"rate limited").retry_after, None);
}
#[test]
fn non_429_is_unchanged() {
    let error = xai_status_error(500, b"Free usage quota exhausted");
    assert_eq!(error.status, 500);
    assert_eq!(error.retry_after, None);
}
