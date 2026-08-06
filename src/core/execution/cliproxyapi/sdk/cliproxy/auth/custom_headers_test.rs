// ref: sdk/cliproxy/auth/custom_headers_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::json;

use super::{apply_custom_headers_from_metadata, extract_custom_headers_from_metadata, Auth};

#[test]
fn extract_custom_headers_matches_pinned_filtering() {
    let metadata = [(
        "headers".to_owned(),
        json!({" X-Test ": " value ", "": "ignored", "X-Empty": "   ", "X-Num": 1}),
    )]
    .into_iter()
    .collect();

    assert_eq!(
        extract_custom_headers_from_metadata(&metadata),
        [("X-Test".to_owned(), "value".to_owned())]
            .into_iter()
            .collect()
    );
}

#[test]
fn apply_custom_headers_replaces_matching_attribute_and_preserves_others() {
    let mut auth = Auth::default();
    auth.metadata
        .insert("headers".into(), json!({"X-Test": "new", "X-Empty": "   "}));
    auth.attributes.insert("header:X-Test".into(), "old".into());
    auth.attributes.insert("keep".into(), "1".into());

    apply_custom_headers_from_metadata(&mut auth);

    assert_eq!(
        auth.attributes.get("header:X-Test").map(String::as_str),
        Some("new")
    );
    assert!(!auth.attributes.contains_key("header:X-Empty"));
    assert_eq!(auth.attributes.get("keep").map(String::as_str), Some("1"));
}

#[test]
fn executable_header_metadata_rejects_name_and_value_injection() {
    let metadata = [(
        "headers".to_owned(),
        json!({
            "X-Good": "yes",
            "X-Evil\r\nInjected": "yes",
            "X-Value": "safe\r\nInjected: yes"
        }),
    )]
    .into_iter()
    .collect();

    let headers = extract_custom_headers_from_metadata(&metadata);
    assert_eq!(headers.len(), 1);
    assert_eq!(headers.get("X-Good").map(String::as_str), Some("yes"));
}
