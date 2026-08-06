// ref: sdk/api/handlers/handlers_stream_bootstrap_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::io;
use std::sync::Arc;

use super::*;

#[tokio::test]
async fn bootstrap_does_not_commit_an_initial_error() {
    let (sender, mut receiver) = mpsc::channel(1);
    sender
        .send(StreamChunk {
            payload: Vec::new(),
            error: Some(Arc::new(io::Error::other("upstream failed"))),
        })
        .await
        .unwrap();
    drop(sender);
    let error = bootstrap_stream(&super::super::HandlerCancellation::default(), &mut receiver)
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "upstream failed");
}

#[tokio::test]
async fn bootstrap_commits_only_after_first_success_or_clean_eof() {
    let (sender, mut receiver) = mpsc::channel(1);
    sender
        .send(StreamChunk {
            payload: b"data: {}\n\n".to_vec(),
            error: None,
        })
        .await
        .unwrap();
    let bootstrap = bootstrap_stream(&super::super::HandlerCancellation::default(), &mut receiver)
        .await
        .unwrap();
    assert!(bootstrap.committed);
    assert_eq!(bootstrap.first_chunk.unwrap().payload, b"data: {}\n\n");
}

#[test]
fn validates_every_sse_data_json_line() {
    assert!(
        validate_sse_data_json(b"event: response\ndata: {\"ok\":true}\n\ndata: [DONE]\n\n").is_ok()
    );
    assert!(validate_sse_data_json(b"data: {broken}\n\n").is_err());
}
