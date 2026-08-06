// ref: internal/translator/codex/claude/noop_optimization_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_claude_request_to_codex;

#[test]
fn codex_required_invariants_are_always_canonical() {
    let output: Value = serde_json::from_slice(&convert_claude_request_to_codex(
        "gpt-5",
        br#"{"messages":[{"role":"user","content":"hi"}]}"#,
        false,
    ))
    .unwrap();
    assert_eq!(output["instructions"], "");
    assert_eq!(output["stream"], true);
    assert_eq!(output["store"], false);
    assert_eq!(output["parallel_tool_calls"], true);
    assert_eq!(output["include"][0], "reasoning.encrypted_content");
}
