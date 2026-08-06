// ref: internal/util/claude_tool_id_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::claude_tool_id::{
    gemini_claude_tool_use_id, is_gemini_claude_tool_use_id, sanitize_claude_tool_id,
};

#[test]
fn stable_gemini_id_is_bound_to_native_call_name_and_canonical_args() {
    let first = gemini_claude_tool_use_id(
        "native-call-1",
        "Edit",
        r#"{"file_path":"/tmp/a","old_string":"x","new_string":"y"}"#,
    );
    let reordered = gemini_claude_tool_use_id(
        "native-call-1",
        "Edit",
        r#"{"new_string":"y","old_string":"x","file_path":"/tmp/a"}"#,
    );
    assert_eq!(first, reordered);
    assert!(is_gemini_claude_tool_use_id(&first));
    assert_ne!(
        first,
        gemini_claude_tool_use_id(
            "native-call-1",
            "Edit",
            r#"{"file_path":"/tmp/a","old_string":"x","new_string":"z"}"#,
        )
    );
    assert!(gemini_claude_tool_use_id("", "Edit", "{}").is_empty());
}

#[test]
fn claude_tool_ids_replace_only_non_protocol_characters() {
    assert_eq!(sanitize_claude_tool_id("tool/a:b c"), "tool_a_b_c");
    assert!(sanitize_claude_tool_id("")
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')));
    assert!(!is_gemini_claude_tool_use_id("toolu_client_value"));
}
