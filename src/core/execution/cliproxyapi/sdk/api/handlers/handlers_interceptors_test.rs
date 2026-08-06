// ref: sdk/api/handlers/handlers_interceptors_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use super::*;

#[test]
fn request_headers_support_case_insensitive_clear_and_replace() {
    let current = BTreeMap::from([
        ("Authorization".to_owned(), vec!["old".to_owned()]),
        ("X-Stable".to_owned(), vec!["yes".to_owned()]),
    ]);
    let updates = BTreeMap::from([("authorization".to_owned(), vec!["new".to_owned()])]);
    let result = apply_interceptor_headers(&current, &updates, &["x-stable".to_owned()]);
    assert_eq!(result.len(), 1);
    assert_eq!(result["authorization"], ["new"]);
}

#[test]
fn termination_defaults_to_forbidden_and_filters_reserved_headers() {
    let intercepted = RequestInterceptResponse {
        terminate: true,
        response_headers: BTreeMap::from([
            ("X-Plugin".to_owned(), vec!["yes".to_owned()]),
            ("X-Cpa-Trace-Id".to_owned(), vec!["bad".to_owned()]),
        ]),
        response_body: b"blocked".to_vec(),
        ..RequestInterceptResponse::default()
    };
    let response = termination_response(&intercepted).unwrap();
    assert_eq!(response.status, 403);
    assert_eq!(response.body, b"blocked");
    assert_eq!(response.headers["X-Plugin"], ["yes"]);
    assert!(!response.headers.contains_key("X-Cpa-Trace-Id"));
}

#[test]
fn stream_history_is_bounded_by_chunk_count() {
    let mut history = Vec::new();
    for index in 0..80 {
        history = append_stream_interceptor_history(&history, &[index]);
    }
    assert_eq!(history.len(), MAX_STREAM_INTERCEPTOR_HISTORY_CHUNKS);
    assert_eq!(history[0], [16]);
}
