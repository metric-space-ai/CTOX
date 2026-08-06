// ref: sdk/api/handlers/openai/openai_images_handlers_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;

#[test]
fn model_validation_allows_gpt_and_xai_images() {
    for model in [
        "gpt-image-2",
        "codex/gpt-image-1.5",
        "xai/grok-imagine-image-quality",
    ] {
        assert!(is_supported_images_model(model));
    }
    assert!(!is_supported_images_model("gpt-5.4"));
}

#[test]
fn compat_request_controls_stream_without_losing_fields() {
    let raw = br#"{"model":"old","prompt":"draw","stream":false}"#;
    let stream: Value = serde_json::from_slice(&build_openai_compat_images_json_request(
        raw,
        "gpt-image-2",
        true,
    ))
    .unwrap();
    assert_eq!(stream["model"], "gpt-image-2");
    assert_eq!(stream["prompt"], "draw");
    assert_eq!(stream["stream"], true);
    let buffered: Value = serde_json::from_slice(&build_openai_compat_images_json_request(
        raw,
        "gpt-image-2",
        false,
    ))
    .unwrap();
    assert!(buffered.get("stream").is_none());
}

#[test]
fn sse_accumulator_reassembles_split_frames() {
    let mut accumulator = SseFrameAccumulator::default();
    assert!(accumulator.add_chunk(b"data: {\"a\":").is_empty());
    assert_eq!(accumulator.add_chunk(b"1}\n\ndata: {}\n\n").len(), 2);
    assert!(accumulator.flush().is_empty());
}
