// ref: internal/translator/codex/openai/chat-completions/noop_optimization_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_openai_chat_request_to_codex;

#[test]
fn request_has_codex_invariants_and_no_legacy_chat_fields() {
    let output: Value = serde_json::from_slice(&convert_openai_chat_request_to_codex(
        "gpt-5",
        br#"{"messages":[{"role":"user","content":"hi"}]}"#,
        true,
    ))
    .unwrap();
    assert_eq!(output["instructions"], "");
    assert_eq!(output["store"], false);
    assert_eq!(output["include"][0], "reasoning.encrypted_content");
    assert!(output.get("messages").is_none());
}
