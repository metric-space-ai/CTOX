// ref: sdk/cliproxy/auth/request_termination_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::cliproxy::executor::RequestTerminatedError;

use super::{is_request_terminated_error, should_attempt_antigravity_credits_fallback};

#[test]
fn request_terminated_error_skips_credits_fallback() {
    let error = RequestTerminatedError {
        http_status: 429,
        headers: Default::default(),
        body: Vec::new(),
    };
    assert!(is_request_terminated_error(&error));
    assert!(!should_attempt_antigravity_credits_fallback(
        true,
        &error,
        &["antigravity".to_owned()]
    ));
}
