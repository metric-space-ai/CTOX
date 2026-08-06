// ref: internal/runtime/executor/codex_openai_images_extract_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_openai_images::{extract_codex_image_results, CodexImageError};

#[test]
fn completed_output_is_validated_and_extracted() {
    let images = extract_codex_image_results(
        br#"{"output":[{"type":"image_generation_call","result":"cG5n","output_format":"png"}]}"#,
    )
    .unwrap();
    assert_eq!(images[0].base64_data, "cG5n");
    assert_eq!(
        extract_codex_image_results(br#"{"output":[]}"#),
        Err(CodexImageError::InvalidCompletion)
    );
}
