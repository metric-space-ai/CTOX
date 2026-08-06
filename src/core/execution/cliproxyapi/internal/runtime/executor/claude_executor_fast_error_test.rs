// ref: internal/runtime/executor/claude_executor_fast_error_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{
    classify_claude_upstream_error, claude_fast_direct_response_error, ClaudeFastRequestError,
};
use crate::sdk::cliproxy::executor::{RequestScopedError, StatusError};

#[test]
fn fast_entitlement_refusal_is_request_scoped() {
    let error = classify_claude_upstream_error(
        429,
        br#"{"error":{"message":"Usage credits are required for fast mode"}}"#,
    )
    .unwrap();
    assert!(error.is_request_scoped());
    assert_eq!(error.status_code(), 429);
}

#[test]
fn ordinary_rate_limit_is_not_reclassified() {
    assert!(
        classify_claude_upstream_error(429, br#"{"error":{"message":"rate limit"}}"#).is_none()
    );
    let _type_contract: Option<ClaudeFastRequestError> = None;
}

#[test]
fn fast_direct_response_keeps_body_and_drops_stale_representation_headers() {
    let mut headers = crate::sdk::cliproxy::executor::Headers::new();
    headers.insert("Content-Type".into(), vec!["application/json".into()]);
    headers.insert("Content-Encoding".into(), vec!["gzip".into()]);
    headers.insert("content-length".into(), vec!["999".into()]);
    let error = claude_fast_direct_response_error(429, headers, br#"{"error":"credits"}"#);
    assert_eq!(error.status_code(), 429);
    assert_eq!(error.response_body(), br#"{"error":"credits"}"#);
    assert_eq!(
        error.response_headers().get("Content-Type").unwrap(),
        &["application/json"]
    );
    assert!(error
        .response_headers()
        .keys()
        .all(|name| !name.eq_ignore_ascii_case("content-encoding")
            && !name.eq_ignore_ascii_case("content-length")));
}
