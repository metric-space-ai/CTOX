// ref: internal/runtime/executor/codex_executor_signature_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_executor_request::{
    prepare_codex_compact_body, prepare_codex_responses_body, CodexRequestPolicy,
};

#[test]
fn invalid_overlong_encrypted_reasoning_is_dropped_for_all_paths() {
    let raw = format!(
        r#"{{"input":[{{"type":"reasoning","id":"{}","encrypted_content":"sig"}}]}}"#,
        "r".repeat(80)
    );
    let responses = prepare_codex_responses_body(
        raw.as_bytes(),
        CodexRequestPolicy {
            model: "gpt-5.3-codex",
            plan_type: "free",
            responses_lite: false,
            disable_image_generation: true,
        },
    )
    .unwrap();
    let compact = prepare_codex_compact_body(raw.as_bytes(), "gpt-5.3-codex").unwrap();
    assert!(
        serde_json::from_slice::<serde_json::Value>(&responses).unwrap()["input"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(
        serde_json::from_slice::<serde_json::Value>(&compact).unwrap()["input"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}
