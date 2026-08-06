// ref: internal/runtime/executor/helps/logging_helpers_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use super::logging_helpers::*;
use crate::internal::logging::{
    get_response_headers, with_request_id, with_response_headers_holder, RequestContext,
};

struct FixedClock;
impl ApiLogClock for FixedClock {
    fn now(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(123)
    }
}

fn context(request: RequestContext) -> ApiLogContext {
    ApiLogContext::new(Arc::new(FixedClock), request)
}

#[test]
fn record_api_request_clones_deferred_body_when_request_log_disabled() {
    let context = context(RequestContext::default());
    let mut body = br#"{"model":"original"}"#.to_vec();
    record_api_request(
        Some(&context),
        RequestLogPolicy::default(),
        UpstreamRequestLog {
            url: "https://api.example.com/v1/responses".into(),
            method: "POST".into(),
            body: body.clone(),
            ..UpstreamRequestLog::default()
        },
    );
    body[10] = b'X';
    let deferred = context.deferred_requests();
    assert_eq!(deferred.len(), 1);
    let captured = String::from_utf8((deferred[0])()).unwrap();
    assert!(captured.contains(r#"{"model":"original"}"#));
}

#[test]
fn response_metadata_clones_headers_when_request_log_disabled() {
    let request = with_response_headers_holder(None);
    let context = context(request);
    let mut headers = LogHeaders::from([(
        "X-Upstream-Request-Id".into(),
        vec!["upstream-req-1".into()],
    )]);
    record_api_response_metadata(Some(&context), RequestLogPolicy::default(), 200, &headers);
    headers.insert("X-Upstream-Request-Id".into(), vec!["mutated".into()]);
    assert_eq!(
        get_response_headers(Some(context.request_context())).unwrap()["X-Upstream-Request-Id"],
        ["upstream-req-1"]
    );
}

#[test]
fn attempts_sse_websocket_auth_and_credits_preserve_upstream_shape() {
    let request = with_request_id(None, "req-1");
    let context = context(request);
    let policy = RequestLogPolicy {
        request_log: true,
        commercial_mode: false,
    };
    record_api_request(
        Some(&context),
        policy,
        UpstreamRequestLog {
            url: "https://api.example".into(),
            method: "POST".into(),
            headers: LogHeaders::from([("Authorization".into(), vec!["Bearer secret".into()])]),
            body: b"{}".to_vec(),
            auth_type: "oauth".into(),
            ..UpstreamRequestLog::default()
        },
    );
    record_api_response_metadata(Some(&context), policy, 200, &LogHeaders::new());
    append_api_response_chunk(Some(&context), policy, b"event: message");
    append_api_response_chunk(Some(&context), policy, b"data: {}");
    let response = String::from_utf8(context.aggregated_response()).unwrap();
    assert!(response.contains("Status: 200"));
    assert!(response.contains("event: message\ndata: {}"));
    assert!(!String::from_utf8(context.aggregated_request())
        .unwrap()
        .contains("Bearer secret"));
    assert_eq!(request_id(Some(&context)), "req-1");
    assert!(!credits_used(Some(&context)));
    mark_credits_used(Some(&context));
    assert!(credits_used(Some(&context)));

    append_api_websocket_response(Some(&context), policy, b"frame");
    assert!(String::from_utf8(context.websocket_timeline())
        .unwrap()
        .contains("api.websocket.response"));
}

#[test]
fn error_summaries_and_websocket_url_are_safe() {
    assert_eq!(
        summarize_error_body("application/json", br#"{"error":{"message":"overloaded"}}"#),
        "overloaded"
    );
    assert_eq!(
        summarize_error_body(
            "text/html",
            b"<html><title> Cloud &amp; Error </title></html>"
        ),
        "Cloud & Error"
    );
    assert_eq!(
        summarize_error_body("text/html", b"<html>secret body</html>"),
        "[html body omitted]"
    );
    assert_eq!(
        websocket_upgrade_request_url("wss://api.example/ws?q=1"),
        "https://api.example/ws?q=1"
    );
}
