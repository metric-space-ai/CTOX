// ref: internal/runtime/executor/helps/claude_mcp_alias_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox

use super::{claude_mcp_tool_alias, is_claude_mcp_tool_name};

#[test]
fn claude_mcp_tool_name_validation_matches_candidate_table() {
    for name in [
        "mcp__context7__query-docs",
        "mcp__amber_cedar__quiet_harbor",
        "mcp__server__tool__variant",
    ] {
        assert!(is_claude_mcp_tool_name(name), "{name}");
    }
    for name in [
        "context7__query-docs".to_owned(),
        "mcp____query-docs".to_owned(),
        "mcp__context7__".to_owned(),
        "mcp__context7__query.docs".to_owned(),
        format!("mcp__context7__{}", "x".repeat(64)),
    ] {
        assert!(!is_claude_mcp_tool_name(&name), "{name}");
    }
}

#[test]
fn claude_mcp_tool_alias_is_stable_scoped_and_retryable() {
    let first = claude_mcp_tool_alias("credential-secret", "search_web", 0);
    assert_eq!(
        claude_mcp_tool_alias("credential-secret", "search_web", 0),
        first
    );
    let case_distinct = claude_mcp_tool_alias("credential-secret", "Search_Web", 0);
    let retry = claude_mcp_tool_alias("credential-secret", "search_web", 1);
    assert_ne!(first, case_distinct);
    assert_ne!(first, retry);
    assert!(is_claude_mcp_tool_name(&first));
    assert!(first.ends_with("_search_web"));
    let parts = first.split("__").collect::<Vec<_>>();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[1].len(), 12);
    assert!(parts[1]
        .bytes()
        .all(|byte| byte.is_ascii_lowercase() || (b'2'..=b'7').contains(&byte)));
    assert!(parts[2].starts_with(|character: char| {
        character.is_ascii_lowercase() || ('2'..='7').contains(&character)
    }));
    assert_eq!(case_distinct.split("__").nth(1), Some(parts[1]));
    assert_eq!(retry.split("__").nth(1), Some(parts[1]));
    assert_ne!(
        claude_mcp_tool_alias("other-caller", "search_web", 0)
            .split("__")
            .nth(1),
        Some(parts[1])
    );
}

#[test]
fn claude_mcp_tool_alias_semantic_suffix_is_safe_and_bounded() {
    for (original, suffix, expected_length) in [
        ("browser.open URL", "_browser_open_URL".to_owned(), None),
        (
            "search.网页/tool with spaces",
            "_search_tool_with_spaces".to_owned(),
            None,
        ),
        ("搜索网页", "_tool".to_owned(), None),
        (&"a".repeat(100), format!("_{}", "a".repeat(32)), Some(64)),
    ] {
        let alias = claude_mcp_tool_alias("credential-secret", original, 0);
        assert!(is_claude_mcp_tool_name(&alias), "{alias}");
        assert!(alias.len() <= 64, "{alias}");
        assert!(alias.ends_with(&suffix), "{alias} does not end in {suffix}");
        if let Some(expected_length) = expected_length {
            assert_eq!(alias.len(), expected_length);
        }
    }
}
