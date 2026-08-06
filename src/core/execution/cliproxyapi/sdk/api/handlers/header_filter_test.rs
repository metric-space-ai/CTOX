// ref: sdk/api/handlers/header_filter_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{
    filter_upstream_headers, is_cpa_reserved_response_header, write_upstream_headers, HeaderMap,
};

fn headers(entries: &[(&str, &[&str])]) -> HeaderMap {
    entries
        .iter()
        .map(|(key, values)| {
            (
                (*key).to_owned(),
                values.iter().map(|value| (*value).to_owned()).collect(),
            )
        })
        .collect()
}

#[test]
fn removes_connection_scoped_hop_by_hop_and_reserved_headers() {
    let source = headers(&[
        ("Connection", &["keep-alive, x-hop-a, x-hop-b", "x-hop-c"]),
        ("Keep-Alive", &["timeout=5"]),
        ("X-Hop-A", &["a"]),
        ("x-hop-b", &["b"]),
        ("X-HOP-C", &["c"]),
        ("X-Request-Id", &["req-1"]),
        ("Set-Cookie", &["session=secret"]),
        ("x-cpa-trace-id", &["upstream-trace"]),
        ("Access-Control-Expose-Headers", &["upstream-header"]),
    ]);

    let filtered = filter_upstream_headers(&source).expect("preserved request ID");
    assert_eq!(filtered["X-Request-Id"], ["req-1"]);
    for blocked in [
        "Connection",
        "Keep-Alive",
        "X-Hop-A",
        "x-hop-b",
        "X-HOP-C",
        "Set-Cookie",
        "x-cpa-trace-id",
        "Access-Control-Expose-Headers",
    ] {
        assert!(!filtered.contains_key(blocked), "unexpected {blocked}");
    }
}

#[test]
fn returns_none_when_every_header_is_blocked() {
    let source = headers(&[
        ("Connection", &["x-hop-a"]),
        ("X-Hop-A", &["a"]),
        ("Set-Cookie", &["session=secret"]),
    ]);
    assert_eq!(filter_upstream_headers(&source), None);
}

#[test]
fn gateway_metadata_and_cpa_owned_headers_are_case_insensitively_blocked() {
    let source = headers(&[
        ("X-LiteLLM-Model", &["hidden"]),
        ("helicone-cache", &["hit"]),
        ("CF-AIG-Metadata", &["gateway"]),
        ("Content-Encoding", &["gzip"]),
        ("X-Safe-Upstream", &["kept"]),
    ]);
    let filtered = filter_upstream_headers(&source).unwrap();
    assert_eq!(filtered, headers(&[("X-Safe-Upstream", &["kept"])]));
    assert!(is_cpa_reserved_response_header(
        "ACCESS-CONTROL-ALLOW-ORIGIN"
    ));
    assert!(!is_cpa_reserved_response_header("x-safe-upstream"));
}

#[test]
fn writer_never_overwrites_a_handler_owned_header() {
    let mut destination = headers(&[("content-type", &["application/json"])]);
    let source = headers(&[
        ("Content-Type", &["text/plain"]),
        ("X-Request-Id", &["req-a", "req-b"]),
    ]);
    write_upstream_headers(&mut destination, &source);
    assert_eq!(destination["content-type"], ["application/json"]);
    assert_eq!(destination["X-Request-Id"], ["req-a", "req-b"]);
}
