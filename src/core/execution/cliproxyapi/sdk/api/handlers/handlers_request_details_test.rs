// ref: sdk/api/handlers/handlers_request_details_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;

#[test]
fn route_base_strips_valid_suffix_and_preserves_plain_model() {
    assert_eq!(route_model_base_name("gpt-5.2(high)"), "gpt-5.2");
    assert_eq!(
        route_model_base_name("gemini-2.5-pro(8192)"),
        "gemini-2.5-pro"
    );
    assert_eq!(
        route_model_base_name("claude-sonnet-4-5"),
        "claude-sonnet-4-5"
    );
}

#[test]
fn image_only_models_fail_closed_outside_image_endpoints() {
    for model in [
        "gpt-image-1.5",
        "gpt-image-2",
        "codex/gpt-image-2",
        "grok-imagine-image",
        "xai/grok-imagine-image-quality",
    ] {
        assert!(is_openai_image_only_model(model));
        assert!(validate_image_only_model(model, false).is_err());
        assert!(validate_image_only_model(model, true).is_ok());
    }
    assert!(!is_openai_image_only_model("grok-imagine-video"));
}
