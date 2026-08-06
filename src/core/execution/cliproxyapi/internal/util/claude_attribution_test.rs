// ref: internal/util/claude_attribution_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::is_claude_code_attribution_system_text;

#[test]
fn pinned_attribution_cases_match_upstream() {
    for (text, expected) in [
        (
            "x-anthropic-billing-header: cc_version=2.1.63.abc; cc_entrypoint=cli; cch=12345;",
            true,
        ),
        (
            "\n\t x-anthropic-billing-header: cc_version=2.1.63.abc; cch=12345;",
            true,
        ),
        ("You are helpful.", false),
        ("", false),
    ] {
        assert_eq!(is_claude_code_attribution_system_text(text), expected);
    }
}

#[test]
fn unicode_whitespace_and_exact_case_sensitive_prefix_are_preserved() {
    assert!(is_claude_code_attribution_system_text(
        "\u{2003}\u{00a0}x-anthropic-billing-header: cch=1"
    ));
    assert!(!is_claude_code_attribution_system_text(
        "X-Anthropic-Billing-Header: cch=1"
    ));
    assert!(!is_claude_code_attribution_system_text(
        "prefix x-anthropic-billing-header: cch=1"
    ));
    assert!(!is_claude_code_attribution_system_text(
        "\u{200b}x-anthropic-billing-header: cch=1"
    ));
}
