// ref: sdk/cliproxy/auth/response_model_rewriter_antigravity_sim_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::internal::translator::gemini::openai::responses::{
    convert_gemini_response_to_openai_responses_stream, GeminiToResponsesState,
};

use super::{
    finish_force_mapped_stream_chunks, rewrite_force_mapped_stream_chunk, StreamRewriteOptions,
    StreamRewriter,
};

fn antigravity_live_chunks() -> Vec<Vec<u8>> {
    let request = br#"{"model":"gemini-3.5-flash","input":[]}"#;
    let mut state = GeminiToResponsesState::with_identity("resp_live", 1_700_000_000);
    [
        br#"{"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"OK"}]}}],"usageMetadata":{"promptTokenCount":21,"candidatesTokenCount":1,"totalTokenCount":131,"thoughtsTokenCount":109},"modelVersion":"gemini-3-flash-a","responseId":"live"}}"#.as_slice(),
        br#"{"response":{"candidates":[{"content":{"role":"model","parts":[{"thoughtSignature":"sig","text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":21,"candidatesTokenCount":1,"totalTokenCount":131,"thoughtsTokenCount":109},"modelVersion":"gemini-3-flash-a","responseId":"live"}}"#.as_slice(),
    ]
    .into_iter()
    .flat_map(|raw| {
        convert_gemini_response_to_openai_responses_stream(request, request, raw, &mut state)
    })
    .collect()
}

fn has_completed(payload: &[u8]) -> bool {
    payload.split(|byte| *byte == b'\n').any(|line| {
        let line = line
            .trim_ascii()
            .strip_prefix(b"data:")
            .unwrap_or(line)
            .trim_ascii();
        serde_json::from_slice::<Value>(line)
            .ok()
            .and_then(|value| value.get("type")?.as_str().map(str::to_owned))
            .as_deref()
            == Some("response.completed")
    })
}

#[test]
fn translator_emits_completed_without_rewriter() {
    assert!(has_completed(&antigravity_live_chunks().concat()));
}

#[test]
fn translator_event_chunks_preserve_completed() {
    let mut rewriter = StreamRewriter::new(StreamRewriteOptions {
        rewrite_model: "gemini-3.5-flash".to_owned(),
    });
    let mut output = antigravity_live_chunks()
        .iter()
        .flat_map(|chunk| rewrite_force_mapped_stream_chunk(Some(&mut rewriter), chunk))
        .collect::<Vec<_>>();
    output.extend(finish_force_mapped_stream_chunks(Some(&mut rewriter)));
    assert!(has_completed(&output));
}

#[test]
fn glued_event_frames_flush_completed() {
    let chunks = antigravity_live_chunks();
    let glued = chunks.concat();
    let mut rewriter = StreamRewriter::new(StreamRewriteOptions {
        rewrite_model: "gemini-3.5-flash".to_owned(),
    });
    let mut output = rewrite_force_mapped_stream_chunk(Some(&mut rewriter), &glued);
    output.extend(finish_force_mapped_stream_chunks(Some(&mut rewriter)));
    assert!(has_completed(&output));
}
