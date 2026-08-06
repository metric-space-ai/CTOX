// ref: internal/runtime/executor/codex_openai_images_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_openai_images::{
    codex_is_images_endpoint_path, prepare_codex_openai_image_request, CodexImageAction,
};

#[test]
fn generation_and_edit_requests_use_responses_image_tool() {
    assert!(codex_is_images_endpoint_path("/v1/images/generations"));
    let generation = prepare_codex_openai_image_request(
        br#"{"prompt":"draw"}"#,
        "gpt-image-2",
        CodexImageAction::Generate,
    )
    .unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&generation.responses_body).unwrap()["tools"]
            [0]["action"],
        "generate"
    );
    let edit = prepare_codex_openai_image_request(
        br#"{"prompt":"edit","image":"data:image/png;base64,cG5n"}"#,
        "gpt-image-2",
        CodexImageAction::Edit,
    )
    .unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&edit.responses_body).unwrap()["tools"][0]
            ["action"],
        "edit"
    );
}
