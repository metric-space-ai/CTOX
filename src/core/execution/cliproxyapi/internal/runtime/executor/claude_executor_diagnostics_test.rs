// ref: internal/runtime/executor/claude_executor_diagnostics_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{claude_message_id_from_sse, commit_claude_diagnostics, inject_claude_diagnostics};

#[test]
fn diagnostics_preserve_native_order_and_advance_only_on_commit() {
    let body = br#"{"context_management":{"edits":[]},"max_tokens":1}"#;
    let (first, state) = inject_claude_diagnostics(
        body,
        "credential-diagnostics-test",
        "session-diagnostics-test",
    );
    assert!(String::from_utf8_lossy(&first).contains("\"context_management\":{\"edits\":[]},\"diagnostics\":{\"previous_message_id\":null},\"max_tokens\""));
    commit_claude_diagnostics(&state, "msg_01");
    let (second, _) = inject_claude_diagnostics(
        body,
        "credential-diagnostics-test",
        "session-diagnostics-test",
    );
    assert!(String::from_utf8_lossy(&second).contains("\"previous_message_id\":\"msg_01\""));
}

#[test]
fn incomplete_sse_does_not_commit_an_id() {
    assert_eq!(
        claude_message_id_from_sse(
            b"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\"}}\n"
        ),
        ""
    );
}

#[test]
fn caller_diagnostics_are_replaced_without_duplicate_members() {
    let body = br#"{"context_management":{},"diagnostics":{"previous_message_id":"caller"},"max_tokens":1}"#;
    let (updated, _) =
        inject_claude_diagnostics(body, "credential-replace-test", "session-replace-test");
    let text = String::from_utf8(updated).unwrap();
    assert_eq!(text.matches("\"diagnostics\"").count(), 1);
    assert!(!text.contains("caller"));
    assert!(text.contains("\"context_management\":{},\"diagnostics\":"));
}
