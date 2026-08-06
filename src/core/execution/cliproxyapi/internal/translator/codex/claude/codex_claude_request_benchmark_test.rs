// ref: internal/translator/codex/claude/codex_claude_request_benchmark_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::json;

use super::convert_claude_request_to_codex;

#[test]
fn large_history_smoke_is_bounded_and_complete() {
    let messages: Vec<_> = (0..512)
        .map(|index| json!({"role":"user","content":format!("message {index}")}))
        .collect();
    let raw = serde_json::to_vec(&json!({"messages":messages})).unwrap();
    let output: serde_json::Value =
        serde_json::from_slice(&convert_claude_request_to_codex("gpt-5", &raw, false)).unwrap();
    assert_eq!(output["input"].as_array().unwrap().len(), 512);
}
