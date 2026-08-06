// ref: internal/translator/codex/interactions/noop_optimization_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_interactions_request_to_codex;

#[test]
fn canonical_tool_schema_is_preserved_without_foreign_shape() {
    let output: Value = serde_json::from_slice(&convert_interactions_request_to_codex(
        "gpt-5.6",
        br#"{"input":"hi","tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}}]}"#,
        false,
    ))
    .unwrap();
    assert_eq!(output["tools"][0]["parameters"]["required"][0], "q");
    assert_eq!(output["tool_choice"], "auto");
    assert!(output.get("contents").is_none());
    assert!(output.get("systemInstruction").is_none());
}
