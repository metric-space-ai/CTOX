// ref: internal/util/header_helpers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use super::{
    apply_custom_headers, apply_custom_headers_from_attrs, canonical_header_name,
    extract_custom_headers, HeaderRequest,
};

#[test]
fn extracts_only_exact_prefixed_non_empty_trimmed_attributes() {
    let attrs = BTreeMap::from([
        ("Header:X-Wrong".to_owned(), "wrong".to_owned()),
        ("header:   ".to_owned(), "ignored".to_owned()),
        ("header: Host ".to_owned(), " example.test ".to_owned()),
        ("header:X-Empty".to_owned(), "\t  ".to_owned()),
        ("header:X-Test".to_owned(), " value ".to_owned()),
        ("other".to_owned(), "ignored".to_owned()),
    ]);

    assert_eq!(
        extract_custom_headers(&attrs),
        Some(BTreeMap::from([
            ("Host".to_owned(), "example.test".to_owned()),
            ("X-Test".to_owned(), "value".to_owned()),
        ]))
    );
    assert_eq!(extract_custom_headers(&BTreeMap::new()), None);
}

#[test]
fn set_replaces_all_values_canonicalizes_and_mirrors_host() {
    let mut request = HeaderRequest {
        host: "old.test".to_owned(),
        headers: BTreeMap::from([
            ("Host".to_owned(), vec!["old.test".to_owned()]),
            (
                "X-Test".to_owned(),
                vec!["default-1".to_owned(), "default-2".to_owned()],
            ),
        ]),
    };
    let attrs = BTreeMap::from([
        ("header:host".to_owned(), "new.test".to_owned()),
        ("header:x-test".to_owned(), "custom".to_owned()),
    ]);

    apply_custom_headers_from_attrs(&mut request, &attrs);

    assert_eq!(request.host, "new.test");
    assert_eq!(request.headers.get("Host"), Some(&vec!["new.test".into()]));
    assert_eq!(request.headers.get("X-Test"), Some(&vec!["custom".into()]));
}

#[test]
fn direct_apply_skips_empty_entries_like_go_helper() {
    let mut request = HeaderRequest::default();
    apply_custom_headers(
        &mut request,
        &BTreeMap::from([
            (String::new(), "value".to_owned()),
            ("X-Empty".to_owned(), String::new()),
            ("X-Good".to_owned(), "yes".to_owned()),
        ]),
    );
    assert_eq!(
        request.headers,
        BTreeMap::from([("X-Good".to_owned(), vec!["yes".to_owned()])])
    );
}

#[test]
fn canonicalization_matches_http_for_tokens_and_preserves_invalid_names() {
    assert_eq!(canonical_header_name("content-TYPE"), "Content-Type");
    assert_eq!(canonical_header_name("x_api-key"), "X_api-Key");
    assert_eq!(canonical_header_name("bad header"), "bad header");
    assert_eq!(canonical_header_name("X-Ünicode"), "X-Ünicode");
}
