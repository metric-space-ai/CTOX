// ref: internal/runtime/executor/codex_executor_parallel_tool_calls_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_executor_request::{prepare_codex_responses_body, CodexRequestPolicy};

fn policy(lite: bool) -> CodexRequestPolicy<'static> {
    CodexRequestPolicy {
        model: "gpt-5.3-codex",
        plan_type: "free",
        responses_lite: lite,
        disable_image_generation: true,
    }
}

#[test]
fn parallel_tool_calls_follow_tool_and_lite_policy() {
    let absent =
        prepare_codex_responses_body(br#"{"parallel_tool_calls":true}"#, policy(false)).unwrap();
    assert!(serde_json::from_slice::<serde_json::Value>(&absent)
        .unwrap()
        .get("parallel_tool_calls")
        .is_none());
    let lite = prepare_codex_responses_body(
        br#"{"tools":[{"type":"function"}],"parallel_tool_calls":true}"#,
        policy(true),
    )
    .unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&lite).unwrap()["parallel_tool_calls"],
        false
    );
}
