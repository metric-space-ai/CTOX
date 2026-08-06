// ref: internal/runtime/executor/codex_executor_instructions_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_executor_request::{prepare_codex_responses_body, CodexRequestPolicy};

#[test]
fn null_instructions_become_empty_string() {
    let body = prepare_codex_responses_body(
        br#"{"instructions":null}"#,
        CodexRequestPolicy {
            model: "gpt-5.3-codex",
            plan_type: "free",
            responses_lite: false,
            disable_image_generation: true,
        },
    )
    .unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).unwrap()["instructions"],
        ""
    );
}
