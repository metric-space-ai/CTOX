// ref: internal/pluginstore/version_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::version::update_available;

#[test]
fn pinned_update_availability_cases_match_upstream() {
    for (installed, latest, expected) in [
        ("", "0.2.0", false),
        ("0.1.0", "0.1.0", false),
        ("v0.1.0", "0.1.0", false),
        ("0.1.0", "0.2.0", true),
        ("v0.1.0", "0.2.0", true),
        ("0.1.9", "0.1.10", true),
        ("0.2.0", "0.1.0", false),
        ("0.1", "0.1.0", false),
        ("0.1.0-rc1", "0.1.0", true),
        ("dev", "0.1.0", true),
    ] {
        assert_eq!(
            update_available(installed, latest),
            expected,
            "installed={installed:?}, latest={latest:?}"
        );
    }
}

#[test]
fn numeric_equality_and_prefix_whitespace_are_normalized() {
    assert!(!update_available(" V01.2 ", "1.2.0.0"));
    assert!(!update_available("1.2.0", "v1.2"));
    assert!(update_available(" 1.2 ", " V1.3 "));
}

#[test]
fn malformed_negative_empty_and_overflow_segments_fall_back_to_inequality() {
    for installed in ["1.-1", "1..0", "9223372036854775808.0", "v"] {
        assert!(update_available(installed, "1.0.0"), "{installed:?}");
        assert!(!update_available(installed, installed));
    }
}
