// ref: internal/translator/common/bytes_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{join_raw_array, new_raw_array_items, set_raw_array_items};

#[test]
fn join_raw_array_matches_empty_single_and_multiple_contracts() {
    assert_eq!(join_raw_array(&[]), b"[]");
    assert_eq!(join_raw_array(&[br#"{"id":1}"#.to_vec()]), br#"[{"id":1}]"#);
    assert_eq!(
        join_raw_array(&[br#"{"id":1}"#.to_vec(), br#"{"id":2}"#.to_vec()]),
        br#"[{"id":1},{"id":2}]"#
    );
}

#[test]
fn new_raw_array_items_preserves_nil_and_capacity() {
    assert!(new_raw_array_items(0).is_none());
    assert!(new_raw_array_items(-1).is_none());
    let items = new_raw_array_items(3).expect("positive capacity");
    assert_eq!(items.len(), 0);
    assert_eq!(items.capacity(), 3);
}

#[test]
fn set_raw_array_items_preserves_surrounding_bytes_and_dotted_paths() {
    type RawArrayCase<'a> = (&'a [u8], &'a str, Vec<Vec<u8>>, &'a [u8]);
    let cases: &[RawArrayCase<'_>] = &[
        (br#"{"items":[]}"#, "items", vec![], br#"{"items":[]}"#),
        (
            br#"{"before":1,"request":{"contents":[]},"after":2}"#,
            "request.contents",
            vec![br#"{"id":1}"#.to_vec()],
            br#"{"before":1,"request":{"contents":[{"id":1}]},"after":2}"#,
        ),
        (
            br#"{"items":[{"old":1},{"old":2}]}"#,
            "items",
            vec![br#"{"id":1}"#.to_vec()],
            br#"{"items":[{"id":1}]}"#,
        ),
        (
            br#"{"items":[]}"#,
            "items",
            vec![br#"{"id":1}"#.to_vec(), br#"{"id":2}"#.to_vec()],
            br#"{"items":[{"id":1},{"id":2}]}"#,
        ),
    ];
    for (data, path, items, expected) in cases {
        assert_eq!(set_raw_array_items(data, path, items), *expected);
    }
}
