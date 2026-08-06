// Origin: CTOX
// License: AGPL-3.0-only

use super::{ensure_header, scrub_proxy_and_fingerprint_headers, Headers};

#[test]
fn scrub_removes_every_pinned_family_case_insensitively() {
    let mut headers = Headers::from([
        ("x-forwarded-for".into(), vec!["203.0.113.7".into()]),
        ("X-Stainless-Os".into(), vec!["MacOS".into()]),
        ("SEC-FETCH-SITE".into(), vec!["same-origin".into()]),
        ("accept-encoding".into(), vec!["gzip, zstd".into()]),
        ("Authorization".into(), vec!["Bearer token".into()]),
    ]);

    scrub_proxy_and_fingerprint_headers(&mut headers);

    assert_eq!(
        headers,
        Headers::from([("Authorization".into(), vec!["Bearer token".into()])])
    );
}

#[test]
fn ensure_header_preserves_the_pinned_priority_and_trimming() {
    let source = Headers::from([("x-client".into(), vec![" source ".into()])]);
    let mut target = Headers::from([("X-Client".into(), vec!["target".into()])]);

    ensure_header(&mut target, Some(&source), "X-Client", "default");
    assert_eq!(target.get("X-Client"), Some(&vec!["source".into()]));

    ensure_header(&mut target, None, "X-Client", "replacement");
    assert_eq!(target.get("X-Client"), Some(&vec!["source".into()]));
}

#[test]
fn ensure_header_uses_default_only_after_blank_source_and_target() {
    let source = Headers::from([("X-Client".into(), vec!["  ".into()])]);
    let mut target = Headers::from([("x-client".into(), vec!["\t".into()])]);

    ensure_header(&mut target, Some(&source), "x-client", " default ");

    assert_eq!(target.len(), 1);
    assert_eq!(target.get("X-Client"), Some(&vec!["default".into()]));
}
