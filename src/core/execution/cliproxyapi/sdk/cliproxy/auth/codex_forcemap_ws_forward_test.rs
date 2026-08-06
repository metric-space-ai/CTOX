// ref: sdk/cliproxy/auth/codex_forcemap_ws_forward_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{
    finish_force_mapped_stream_chunks, normalize_glued_sse_events,
    rewrite_force_mapped_stream_chunk, StreamRewriteOptions, StreamRewriter,
};

fn event_types(chunks: &[Vec<u8>]) -> Vec<String> {
    chunks
        .iter()
        .flat_map(|chunk| normalize_glued_sse_events(chunk))
        .collect::<Vec<_>>()
        .split(|byte| *byte == b'\n')
        .filter_map(|line| {
            let line = line.strip_prefix(b"data:")?;
            serde_json::from_slice::<Value>(line.trim_ascii())
                .ok()?
                .get("type")?
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

fn replay(lines: &[&[u8]]) -> Vec<String> {
    let mut rewriter = StreamRewriter::new(StreamRewriteOptions {
        rewrite_model: "gpt-5.4-fast".to_owned(),
    });
    let mut forwarded = lines
        .iter()
        .map(|line| rewrite_force_mapped_stream_chunk(Some(&mut rewriter), line))
        .filter(|output| !output.is_empty())
        .collect::<Vec<_>>();
    let tail = finish_force_mapped_stream_chunks(Some(&mut rewriter));
    if !tail.is_empty() {
        forwarded.push(tail);
    }
    event_types(&forwarded)
}

#[test]
fn per_line_sse_forwards_completed() {
    let types = replay(&[
        b"event: response.created",
        b"data: {\"type\":\"response.created\",\"response\":{\"model\":\"gpt-5.4\"}}",
        b"event: response.output_text.delta",
        b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"OK\"}",
        b"event: response.completed",
        b"data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5.4\",\"output\":[]}}",
    ]);
    assert!(types.iter().any(|kind| kind == "response.completed"));
}

#[test]
fn fallback_when_pending_buffers_event() {
    let mut rewriter = StreamRewriter::new(StreamRewriteOptions {
        rewrite_model: "gpt-5.4-fast".to_owned(),
    });
    let _ = rewrite_force_mapped_stream_chunk(Some(&mut rewriter), b"event: response.completed");
    let mut output = rewrite_force_mapped_stream_chunk(
        Some(&mut rewriter),
        b"data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5.4\",\"output\":[]}}",
    );
    if output.is_empty() {
        output = finish_force_mapped_stream_chunks(Some(&mut rewriter));
    }
    assert!(String::from_utf8(output)
        .unwrap()
        .contains("response.completed"));
}
