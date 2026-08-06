// ref: sdk/api/handlers/handlers_error_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::io;
use std::sync::Arc;

use super::*;

#[test]
fn addon_headers_are_disabled_by_default_and_enabled_explicitly() {
    let message = ErrorMessage {
        status_code: 429,
        error: Some(Arc::new(io::Error::other("rate limit"))),
        addon: BTreeMap::from([
            ("Retry-After".to_owned(), vec!["30".to_owned()]),
            ("X-Request-Id".to_owned(), vec!["req-1".to_owned()]),
        ]),
        ..ErrorMessage::default()
    };
    let response = build_error_response(None, Some(&message), &HeaderMap::new());
    assert_eq!(response.status, 429);
    assert!(!response.headers.contains_key("Retry-After"));

    let config = SdkConfig {
        passthrough_headers: true,
        ..SdkConfig::default()
    };
    let response = build_error_response(Some(&config), Some(&message), &HeaderMap::new());
    assert_eq!(response.headers["Retry-After"], ["30"]);
    assert_eq!(response.headers["X-Request-Id"], ["req-1"]);
}

#[test]
fn direct_response_preserves_body_and_filters_reserved_headers() {
    let existing = BTreeMap::from([
        ("X-Cpa-Trace-Id".to_owned(), vec!["local-trace".to_owned()]),
        (
            "Access-Control-Allow-Origin".to_owned(),
            vec!["https://trusted.example".to_owned()],
        ),
    ]);
    let message = ErrorMessage {
        status_code: 403,
        direct_response: true,
        body: br#"{"error":"blocked"}"#.to_vec(),
        headers: BTreeMap::from([
            (
                "Content-Type".to_owned(),
                vec!["application/problem+json".to_owned()],
            ),
            ("X-Plugin-Policy".to_owned(), vec!["blocked".to_owned()]),
            ("X-Cpa-Trace-Id".to_owned(), vec!["plugin-trace".to_owned()]),
        ]),
        ..ErrorMessage::default()
    };
    let response = build_error_response(None, Some(&message), &existing);
    assert_eq!(response.status, 403);
    assert_eq!(response.body, br#"{"error":"blocked"}"#);
    assert_eq!(
        response.headers["Content-Type"],
        ["application/problem+json"]
    );
    assert_eq!(response.headers["X-Plugin-Policy"], ["blocked"]);
    assert_eq!(response.headers["X-Cpa-Trace-Id"], ["local-trace"]);
}

#[test]
fn missing_message_fails_closed_as_json_500() {
    let response = build_error_response(None, None, &HeaderMap::new());
    assert_eq!(response.status, 500);
    assert_eq!(response.headers["Content-Type"], ["application/json"]);
    let payload: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(payload["error"]["type"], "server_error");
}
