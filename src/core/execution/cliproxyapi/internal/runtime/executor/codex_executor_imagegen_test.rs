// ref: internal/runtime/executor/codex_executor_imagegen_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_executor_request::{prepare_codex_responses_body, CodexRequestPolicy};

fn prepare(responses_lite: bool, plan_type: &str, model: &str) -> serde_json::Value {
    let body = prepare_codex_responses_body(
        br#"{"input":[],"tools":[]}"#,
        CodexRequestPolicy {
            model,
            plan_type,
            responses_lite,
            disable_image_generation: false,
        },
    )
    .unwrap();
    serde_json::from_slice(&body).unwrap()
}

#[test]
fn image_tool_is_injected_only_for_eligible_requests() {
    assert_eq!(
        prepare(false, "pro", "gpt-5.3-codex")["tools"][0]["type"],
        "image_generation"
    );
    assert!(prepare(true, "pro", "gpt-5.3-codex")["tools"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(prepare(false, "free", "gpt-5.3-codex")["tools"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(prepare(false, "pro", "gpt-5.3-codex-spark")["tools"]
        .as_array()
        .unwrap()
        .is_empty());
}
