// ref: internal/runtime/executor/codex_executor_input_ids_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_executor_request::{prepare_codex_responses_body, CodexRequestPolicy};

#[test]
fn overlong_input_ids_are_deterministically_shortened() {
    let raw = format!(
        r#"{{"input":[{{"type":"message","id":"{}"}}]}}"#,
        "x".repeat(100)
    );
    let policy = CodexRequestPolicy {
        model: "gpt-5.3-codex",
        plan_type: "free",
        responses_lite: false,
        disable_image_generation: true,
    };
    let first = prepare_codex_responses_body(raw.as_bytes(), policy).unwrap();
    let second = prepare_codex_responses_body(raw.as_bytes(), policy).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&first).unwrap();
    assert_eq!(first, second);
    assert!(value["input"][0]["id"].as_str().unwrap().chars().count() <= 64);
}
