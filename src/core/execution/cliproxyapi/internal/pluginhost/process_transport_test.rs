// Origin: CTOX
// Port-Status: adapted_to_ctox
// Port-Note: supplemental process-frame transport evidence
// License: AGPL-3.0-only

use serde_json::json;
use tokio::io::{duplex, AsyncWriteExt};

use super::process_transport::*;
use super::rpc_schema::{encode_upstream_json, ProcessMessage, PROCESS_PROTOCOL_VERSION};
use crate::sdk::pluginabi::Envelope;

fn cancel(request_id: &str) -> ProcessMessage {
    ProcessMessage::Cancel {
        protocol_version: PROCESS_PROTOCOL_VERSION,
        request_id: request_id.into(),
    }
}

#[tokio::test]
async fn duplex_transport_reads_fragmented_frame_and_clean_eof() {
    let (mut writer, mut reader) = duplex(512);
    let message = cancel("request-1");
    let frame = super::rpc_schema::encode_process_frame(&message).unwrap();
    let split = frame.len() / 2;
    writer.write_all(&frame[..split]).await.unwrap();
    writer.write_all(&frame[split..]).await.unwrap();
    writer.shutdown().await.unwrap();

    let decoded = read_process_message(&mut reader).await.unwrap().unwrap();
    assert_eq!(
        super::rpc_schema::encode_process_frame(&decoded).unwrap(),
        frame
    );
    assert!(read_process_message(&mut reader).await.unwrap().is_none());
}

#[tokio::test]
async fn reader_rejects_length_before_allocating_payload() {
    let (mut writer, mut reader) = duplex(16);
    writer
        .write_all(&((super::rpc_schema::MAX_FRAME_BYTES as u32) + 1).to_be_bytes())
        .await
        .unwrap();
    assert_eq!(
        read_process_message(&mut reader).await.unwrap_err(),
        ProcessTransportError::FrameTooLarge
    );
}

#[tokio::test]
async fn writer_and_reader_round_trip_without_payload_in_debug() {
    let secret = "transport-secret-never-debug";
    let message = ProcessMessage::Request {
        protocol_version: PROCESS_PROTOCOL_VERSION,
        request_id: "request-2".into(),
        method: "plugin.register".into(),
        deadline_unix_ms: Some(20),
        payload: encode_upstream_json(&json!({"token": secret})).unwrap(),
    };
    assert!(!format!("{message:?}").contains(secret));

    let (mut host, mut plugin) = duplex(512);
    write_process_message(&mut host, &message).await.unwrap();
    let decoded = read_process_message(&mut plugin).await.unwrap().unwrap();
    assert!(!format!("{decoded:?}").contains(secret));
}

#[test]
fn inflight_limit_duplicate_deadline_and_cancel_are_deterministic() {
    let mut requests = InflightRequests::with_limit(1).unwrap();
    requests
        .begin("a".into(), RequestMode::Unary, Some(20), 10)
        .unwrap();
    assert_eq!(
        requests.begin("a".into(), RequestMode::Unary, None, 10),
        Err(ProcessSessionError::DuplicateRequest)
    );
    assert_eq!(
        requests.begin("b".into(), RequestMode::Unary, None, 10),
        Err(ProcessSessionError::InflightLimit)
    );

    match requests.observe(cancel("a"), 10).unwrap() {
        ProcessEvent::Cancelled { was_active, .. } => assert!(was_active),
        _ => panic!("expected cancellation"),
    }
    match requests.observe(cancel("a"), 10).unwrap() {
        ProcessEvent::Cancelled { was_active, .. } => assert!(!was_active),
        _ => panic!("expected idempotent cancellation"),
    }
    assert_eq!(
        requests.begin("late".into(), RequestMode::Unary, Some(10), 10),
        Err(ProcessSessionError::DeadlineExpired)
    );
}

#[test]
fn session_revalidates_messages_not_received_through_the_codec() {
    let mut requests = InflightRequests::new();
    requests
        .begin("a".into(), RequestMode::Unary, None, 0)
        .unwrap();
    let future_cancel = ProcessMessage::Cancel {
        protocol_version: PROCESS_PROTOCOL_VERSION + 1,
        request_id: "a".into(),
    };
    assert_eq!(
        requests.observe(future_cancel, 0).unwrap_err(),
        ProcessSessionError::InvalidMessage
    );
    assert_eq!(requests.len(), 1);
}

#[test]
fn unary_response_is_terminal_and_unknown_replay_is_rejected() {
    let mut requests = InflightRequests::new();
    requests
        .begin("unary".into(), RequestMode::Unary, None, 0)
        .unwrap();
    let response = || ProcessMessage::Response {
        protocol_version: PROCESS_PROTOCOL_VERSION,
        request_id: "unary".into(),
        envelope: Envelope::success(None),
    };
    assert!(matches!(
        requests.observe(response(), 0).unwrap(),
        ProcessEvent::UnaryResponse { .. }
    ));
    assert_eq!(
        requests.observe(response(), 0).unwrap_err(),
        ProcessSessionError::UnknownRequest
    );
}

#[test]
fn stream_requires_exact_sequence_and_one_terminal_end() {
    let mut requests = InflightRequests::new();
    requests
        .begin("stream".into(), RequestMode::Stream, Some(100), 0)
        .unwrap();
    let chunk = |sequence| ProcessMessage::StreamChunk {
        protocol_version: PROCESS_PROTOCOL_VERSION,
        request_id: "stream".into(),
        sequence,
        payload: encode_upstream_json(&json!({"sequence":sequence})).unwrap(),
    };
    assert_eq!(
        requests.observe(chunk(1), 1).unwrap_err(),
        ProcessSessionError::InvalidStreamSequence
    );
    assert!(matches!(
        requests.observe(chunk(0), 1).unwrap(),
        ProcessEvent::StreamChunk { sequence: 0, .. }
    ));
    let end = || ProcessMessage::StreamEnd {
        protocol_version: PROCESS_PROTOCOL_VERSION,
        request_id: "stream".into(),
        next_sequence: 1,
        error: None,
    };
    assert!(matches!(
        requests.observe(end(), 2).unwrap(),
        ProcessEvent::StreamEnd {
            next_sequence: 1,
            ..
        }
    ));
    assert_eq!(
        requests.observe(end(), 2).unwrap_err(),
        ProcessSessionError::UnknownRequest
    );
}

#[test]
fn deadline_expiry_removes_requests_in_stable_order() {
    let mut requests = InflightRequests::new();
    requests
        .begin("b".into(), RequestMode::Unary, Some(10), 0)
        .unwrap();
    requests
        .begin("a".into(), RequestMode::Stream, Some(10), 0)
        .unwrap();
    requests
        .begin("keep".into(), RequestMode::Unary, Some(11), 0)
        .unwrap();
    assert_eq!(requests.expire(10), vec!["a", "b"]);
    assert_eq!(requests.len(), 1);
    assert_eq!(requests.abort_all(), vec!["keep"]);
    assert!(requests.is_empty());
}
