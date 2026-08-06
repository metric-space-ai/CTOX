// ref: sdk/api/handlers/openai/openai_responses_handlers_stream_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::api::handlers::openai::SseFrameAccumulator;

#[test]
fn responses_stream_reassembles_split_sse_events() {
    let mut accumulator = SseFrameAccumulator::default();
    assert!(accumulator
        .add_chunk(b"event: response.output_item.done\nda")
        .is_empty());
    let frames = accumulator.add_chunk(b"ta: {\"type\":\"response.output_item.done\"}\n\n");
    assert_eq!(frames.len(), 1);
    assert!(String::from_utf8_lossy(&frames[0]).contains("response.output_item.done"));
}

#[test]
fn responses_stream_drops_no_complete_frame_on_normal_flush() {
    let mut accumulator = SseFrameAccumulator::default();
    assert!(accumulator.add_chunk(b"data: {\"partial\":").is_empty());
    assert_eq!(accumulator.flush(), [b"data: {\"partial\":".to_vec()]);
}
