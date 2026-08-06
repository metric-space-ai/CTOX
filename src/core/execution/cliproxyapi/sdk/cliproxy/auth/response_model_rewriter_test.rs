// ref: sdk/cliproxy/auth/response_model_rewriter_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::cliproxy::executor::Response;

use super::{
    finish_force_mapped_stream_chunks, normalize_glued_sse_events, rewrite_force_mapped_response,
    rewrite_force_mapped_stream_chunk, rewrite_model_in_response, rewrite_sse_payload_lines,
    OAuthModelAliasResult, StreamRewriteOptions, StreamRewriter,
};

fn rewriter(model: &str) -> StreamRewriter {
    StreamRewriter::new(StreamRewriteOptions {
        rewrite_model: model.to_owned(),
    })
}

#[test]
fn kimi_messages_data_prefix_without_space() {
    let mut rewriter = rewriter("k2.5");
    let output = rewriter.rewrite_chunk(
        b"event:message_start\ndata:{\"type\":\"message_start\",\"message\":{\"model\":\"kimi-k2.5\"}}\n\n",
    );
    let output = String::from_utf8(output).unwrap();
    assert!(output.contains("\"model\":\"k2.5\""));
    assert!(!output.contains("kimi-k2.5"));
    assert!(output.contains("data:{"));
}

#[test]
fn anthropic_messages_data_prefix_with_space() {
    let mut rewriter = rewriter("grok-latest");
    let output = rewriter.rewrite_chunk(
        b"data: {\"type\":\"message_start\",\"message\":{\"model\":\"grok-4.3\"}}\n\n",
    );
    let output = String::from_utf8(output).unwrap();
    assert!(output.contains("\"model\":\"grok-latest\""));
    assert!(!output.contains("grok-4.3"));
    assert!(output.contains("data: {"));
}

#[test]
fn finish_flushes_codex_responses_event_chunk() {
    let mut rewriter = rewriter("gpt-5.4-fast");
    assert!(rewriter
        .rewrite_chunk(b"event: response.created\n")
        .is_empty());
    let mut output = rewriter.rewrite_chunk(
        b"data: {\"type\":\"response.created\",\"response\":{\"model\":\"gpt-5.4\"}}\n\n",
    );
    output.extend(rewriter.finish());
    let output = String::from_utf8(output).unwrap();
    assert!(output.contains("gpt-5.4-fast"));
    assert!(!output.contains("\"model\":\"gpt-5.4\""));
}

#[test]
fn codex_responses_line_chunks() {
    let mut rewriter = rewriter("gpt-5.4-fast");
    let mut output = Vec::new();
    for line in [
        b"event: response.created\n".as_slice(),
        b"data: {\"type\":\"response.created\",\"response\":{\"model\":\"gpt-5.4\"}}\n".as_slice(),
        b"\n".as_slice(),
        b"event: response.completed\n".as_slice(),
        b"data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5.4\"}}\n"
            .as_slice(),
        b"\n".as_slice(),
    ] {
        output.extend(rewriter.rewrite_chunk(line));
    }
    output.extend(rewriter.finish());
    let output = String::from_utf8(output).unwrap();
    assert!(output.contains("gpt-5.4-fast"));
    assert!(!output.contains("\"model\":\"gpt-5.4\""));
}

#[test]
fn force_mapped_chunks_do_not_duplicate_buffered_event() {
    let mut rewriter = rewriter("gpt-5.4-fast");
    let mut output =
        rewrite_force_mapped_stream_chunk(Some(&mut rewriter), b"event: response.created\n");
    output.extend(rewrite_force_mapped_stream_chunk(
        Some(&mut rewriter),
        b"data: {\"type\":\"response.created\",\"response\":{\"model\":\"gpt-5.4\"}}\n\n",
    ));
    output.extend(finish_force_mapped_stream_chunks(Some(&mut rewriter)));
    let output = String::from_utf8(output).unwrap();
    assert_eq!(output.matches("event: response.created").count(), 1);
    assert!(output.ends_with("\n\n"));
    assert!(output.contains("\"model\":\"gpt-5.4-fast\""));
}

#[test]
fn rewrites_antigravity_model_version() {
    let output = rewrite_model_in_response(
        b"{\"response\":{\"modelVersion\":\"gemini-3-flash\",\"candidates\":[]}}",
        "claude-haiku-4-5-20251001",
    );
    let output = String::from_utf8(output).unwrap();
    assert!(output.contains("claude-haiku-4-5-20251001"));
    assert!(!output.contains("gemini-3-flash"));
}

#[test]
fn live_derived_provider_chunks() {
    for (target, upstream, chunk) in [
        (
            "k2.5",
            "kimi-k2.5",
            b"data:{\"model\":\"kimi-k2.5\"}\n\n".as_slice(),
        ),
        (
            "k2.5",
            "kimi-k2.5",
            b"event:message_start\ndata:{\"message\":{\"model\":\"kimi-k2.5\"}}\n\n".as_slice(),
        ),
        (
            "grok-latest",
            "grok-4.3",
            b"data: {\"message\":{\"model\":\"grok-4.3\"}}\n\n".as_slice(),
        ),
    ] {
        let output = String::from_utf8(rewriter(target).rewrite_chunk(chunk)).unwrap();
        assert!(output.contains(target));
        assert!(!output.contains(upstream));
    }
}

#[test]
fn rewrites_codex_live_sse_frame() {
    let output = rewrite_sse_payload_lines(
        b"event: response.created\ndata: {\"response\":{\"model\":\"gpt-5.4\"}}\n\nevent: response.completed\ndata: {\"response\":{\"model\":\"gpt-5.4\"}}\n\n",
        "gpt-5.4-fast",
    );
    let output = String::from_utf8(output).unwrap();
    assert!(output.contains("gpt-5.4-fast"));
    assert!(!output.contains("\"model\":\"gpt-5.4\""));
}

#[test]
fn response_unchanged_when_force_mapping_disabled() {
    let original = b"{\"model\":\"gpt-5.4\",\"choices\":[]}".to_vec();
    let mut response = Response {
        payload: original.clone(),
        ..Response::default()
    };
    rewrite_force_mapped_response(
        Some(&mut response),
        &OAuthModelAliasResult {
            upstream_model: "gpt-5.4".to_owned(),
            force_mapping: false,
            original_alias: "gpt-5.4-fast".to_owned(),
        },
    );
    assert_eq!(response.payload, original);
}

#[test]
fn stream_chunk_unchanged_without_rewriter() {
    let chunk = b"data: {\"model\":\"gpt-5.4\"}\n\n";
    assert_eq!(rewrite_force_mapped_stream_chunk(None, chunk), chunk);
}

#[test]
fn valid_event_glue_only_is_split() {
    let glued = b"event: a\ndata: {\"type\":\"a\"}event: b\ndata: {\"type\":\"b\"}";
    assert!(String::from_utf8(normalize_glued_sse_events(glued))
        .unwrap()
        .contains("}\n\nevent:"));
    let inside = b"event: d\ndata: {\"text\":\"literal }event: inside\"}";
    assert_eq!(normalize_glued_sse_events(inside), inside);
}

#[test]
fn valid_data_glue_only_is_split() {
    let glued = b"data: {\"type\":\"a\"}data: {\"type\":\"b\"}";
    assert!(String::from_utf8(normalize_glued_sse_events(glued))
        .unwrap()
        .contains("}\ndata:"));
    let inside = b"data: {\"text\":\"literal }data: inside\"}";
    assert_eq!(normalize_glued_sse_events(inside), inside);
}

#[test]
fn data_lines_without_newlines_finish_with_completed_event() {
    let mut rewriter = rewriter("gpt-5.4-fast");
    let mut output = Vec::new();
    for line in [
        b"data: {\"type\":\"response.created\",\"response\":{\"model\":\"gpt-5.4\"}}".as_slice(),
        b"data: {\"type\":\"response.in_progress\",\"response\":{\"model\":\"gpt-5.4\"}}".as_slice(),
        b"data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5.4\",\"output\":[]}}".as_slice(),
    ] {
        output.extend(rewrite_force_mapped_stream_chunk(Some(&mut rewriter), line));
    }
    output.extend(finish_force_mapped_stream_chunks(Some(&mut rewriter)));
    assert!(String::from_utf8(output)
        .unwrap()
        .contains("response.completed"));
}
