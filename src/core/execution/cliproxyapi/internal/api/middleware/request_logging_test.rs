// ref: internal/api/middleware/request_logging_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::request_logging::{
    decode_captured_request_body_for_log_with_limit, mask_sensitive_query,
    should_capture_request_body, RequestMetadata, MAX_ERROR_ONLY_CAPTURED_REQUEST_BODY_BYTES,
};
use std::collections::BTreeMap;

#[test]
fn upstream_error_only_capture_boundaries_are_exact() {
    let mut request = RequestMetadata {
        method: "POST".to_owned(),
        path: "/v1/responses".to_owned(),
        headers: BTreeMap::from([(
            "Content-Type".to_owned(),
            vec!["application/json".to_owned()],
        )]),
        content_length: MAX_ERROR_ONLY_CAPTURED_REQUEST_BODY_BYTES,
        has_body: true,
    };
    assert!(should_capture_request_body(false, Some(&request)));
    request.content_length += 1;
    assert!(!should_capture_request_body(false, Some(&request)));
    request.content_length = -1;
    assert!(!should_capture_request_body(false, Some(&request)));
}

#[test]
fn upstream_mask_and_zstd_failure_paths_fail_closed() {
    let masked = mask_sensitive_query("key=ab&api_key=abcdef&plain=a+b");
    assert_eq!(masked, "key=ab&api_key=ab...ef&plain=a+b");
    assert!(!masked.contains("abcdef"));

    let invalid = b"not-a-zstd-frame";
    assert_eq!(
        decode_captured_request_body_for_log_with_limit(invalid, "zstd", 64),
        invalid
    );
}
