// ref: internal/runtime/executor/codex_executor_translate_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_executor_request::{prepare_codex_responses_body, CodexRequestPolicy};

#[test]
fn equal_translation_reuses_payload_and_different_translation_changes_it() {
    let policy = CodexRequestPolicy {
        model: "gpt-5.3-codex",
        plan_type: "free",
        responses_lite: false,
        disable_image_generation: true,
    };
    let raw = br#"{"input":"hello"}"#;
    assert_eq!(
        prepare_codex_responses_body(raw, policy).unwrap(),
        prepare_codex_responses_body(raw, policy).unwrap()
    );
    let different = br#"{"input":"different"}"#;
    assert_ne!(
        prepare_codex_responses_body(raw, policy).unwrap(),
        prepare_codex_responses_body(different, policy).unwrap()
    );
}
