// ref: internal/runtime/executor/xai_websockets_executor_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use super::xai_websockets_executor::*;
use crate::sdk::cliproxy::auth::Auth;
use crate::sdk::cliproxy::executor::{Headers, RequestScopedError, StatusError};

fn parse(payload: &[u8]) -> Value {
    serde_json::from_slice(payload).expect("valid JSON")
}

#[test]
fn websocket_enablement_prefers_attributes_and_supports_metadata() {
    let mut auth = Auth::default();
    auth.provider = "xai".into();
    auth.metadata.insert("websockets".into(), json!(true));
    assert!(xai_websockets_enabled(Some(&auth)));

    auth.attributes.insert("websockets".into(), "false".into());
    assert!(!xai_websockets_enabled(Some(&auth)));
    auth.attributes.insert("websockets".into(), "true".into());
    assert!(xai_websockets_enabled(Some(&auth)));
    assert!(!xai_websockets_enabled(None));
}

#[test]
fn auto_decision_rejects_required_http_fallback() {
    let auth = Auth::default();
    let error = decide_xai_stream_transport(Some(&auth), true, true).unwrap_err();
    assert_eq!(StatusError::status_code(&error), 426);
    assert!(RequestScopedError::is_request_scoped(&error));
    assert_eq!(error.code, "upstream_http_replay_required");

    assert_eq!(
        decide_xai_stream_transport(Some(&auth), true, false).unwrap(),
        XaiStreamTransportDecision::Http
    );
}

#[test]
fn request_body_sets_store_and_preserves_prompt_cache_key() {
    let body = br#"{"model":"grok-4.3","stream":true,"stream_options":{"include_usage":true},"background":true,"prompt_cache_key":"cache-1","previous_response_id":"resp-prev","instructions":"system prompt","input":[]}"#;
    let payload = parse(&build_xai_websocket_request_body(body));
    assert_eq!(payload["type"], "response.create");
    assert_eq!(payload["store"], true);
    assert_eq!(payload["prompt_cache_key"], "cache-1");
    assert!(payload.get("stream").is_none());
    assert!(payload.get("stream_options").is_none());
    assert!(payload.get("background").is_none());
    assert!(payload.get("instructions").is_none());
}

#[test]
fn request_body_invalid_json_is_not_destroyed() {
    let invalid = b"not-json";
    assert_eq!(build_xai_websocket_request_body(invalid), invalid);
}

#[test]
fn response_urls_validate_and_convert_schemes() {
    assert_eq!(
        build_xai_responses_websocket_url("https://api.x.ai/v1/responses?x=1").unwrap(),
        "wss://api.x.ai/v1/responses?x=1"
    );
    assert_eq!(
        build_xai_responses_websocket_url("ws://127.0.0.1:8080/responses").unwrap(),
        "ws://127.0.0.1:8080/responses"
    );
    assert!(build_xai_responses_websocket_url("file:///tmp/responses").is_err());
}

#[test]
fn headers_do_not_require_global_credential_authority() {
    let headers = apply_xai_websocket_headers(Headers::new(), "secret", "session-1");
    assert_eq!(headers["content-type"], ["application/json"]);
    assert_eq!(headers["authorization"], ["Bearer secret"]);
    assert_eq!(headers["x-grok-conv-id"], ["session-1"]);
}

#[test]
fn transcript_records_request_and_response_and_can_reset() {
    let state = XaiWebsocketIdState::default();
    state.record_transcript_turn(
        br#"{"input":[{"id":"msg-1"}]}"#,
        br#"{"response":{"output":[{"id":"out-1"}]}}"#,
        false,
    );
    assert_eq!(
        parse(&state.snapshot_transcript_input())
            .as_array()
            .unwrap()
            .len(),
        2
    );

    let replay = parse(&state.prepend_transcript_input(br#"{"input":[{"id":"msg-2"}]}"#));
    let ids: Vec<_> = replay["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["msg-1", "out-1", "msg-2"]);

    state.record_transcript_turn(
        br#"{"input":[{"id":"fresh"}]}"#,
        br#"{"response":{"output":[]}}"#,
        true,
    );
    assert_eq!(
        parse(&state.snapshot_transcript_input()),
        json!([{"id":"fresh"}])
    );
}

#[test]
fn compacted_transcript_replays_once_reset_is_requested() {
    let state = Arc::new(XaiWebsocketIdState::default());
    state.replace_transcript_with_items([json!({"type":"compaction","id":"cmp-1"})]);
    let mut mapper = XaiWebsocketRequestIdMapper::new(
        state,
        br#"{"type":"response.append","input":[{"id":"new"}]}"#,
    );
    let payload =
        mapper.upstream_request_payload(br#"{"type":"response.append","input":[{"id":"new"}]}"#);
    assert!(mapper.replayed_compacted_transcript);
    assert_eq!(parse(&payload)["input"].as_array().unwrap().len(), 2);
}

#[test]
fn empty_full_reset_clears_pending_compaction_replay() {
    let state = Arc::new(XaiWebsocketIdState::default());
    state.replace_transcript_with_items([json!({"id":"stale"})]);
    state.record_transcript_turn(br#"{"input":[]}"#, br#"{"response":{"output":[]}}"#, true);
    let mut mapper = XaiWebsocketRequestIdMapper::new(
        state,
        br#"{"type":"response.append","input":[{"id":"new"}]}"#,
    );
    let payload =
        mapper.upstream_request_payload(br#"{"type":"response.append","input":[{"id":"new"}]}"#);
    assert_eq!(parse(&payload)["input"], json!([{"id":"new"}]));
}

#[test]
fn missing_upstream_previous_id_replays_recorded_context() {
    let state = Arc::new(XaiWebsocketIdState::default());
    state.record_transcript_turn(
        br#"{"input":[{"id":"old"}]}"#,
        br#"{"response":{"output":[]}}"#,
        false,
    );
    state.map_downstream_to_upstream("downstream-prev", "");
    let request = br#"{"type":"response.create","previous_response_id":"downstream-prev","input":[{"id":"new"}]}"#;
    let mut mapper = XaiWebsocketRequestIdMapper::new(state, request);
    let upstream = parse(&mapper.upstream_request_payload(request));
    assert!(upstream.get("previous_response_id").is_none());
    assert_eq!(upstream["input"].as_array().unwrap().len(), 2);
}

#[test]
fn repeated_upstream_response_ids_are_stable_downstream() {
    let state = Arc::new(XaiWebsocketIdState::default());
    let first_request = br#"{"input":[]}"#;
    let mut first = XaiWebsocketRequestIdMapper::new(state.clone(), first_request);
    let first_event = first.downstream_response_payload(
        br#"{"type":"response.completed","response":{"id":"resp-1","output":[{"id":"item-resp-1"}]}}"#,
    );
    assert_eq!(parse(&first_event)["response"]["id"], "resp-1");

    let second_request = br#"{"previous_response_id":"resp-1","input":[]}"#;
    let mut second = XaiWebsocketRequestIdMapper::new(state, second_request);
    let second_event = second.downstream_response_payload(
        br#"{"type":"response.completed","previous_response_id":"resp-1","response":{"id":"resp-1","output":[{"item_id":"item-resp-1"}]}}"#,
    );
    let event = parse(&second_event);
    assert_eq!(event["response"]["id"], "resp-1-xai-1");
    assert_eq!(
        event["response"]["output"][0]["item_id"],
        "item-resp-1-xai-1"
    );
    assert_eq!(event["previous_response_id"], "resp-1");
}

#[test]
fn id_rewrite_only_touches_id_fields() {
    let payload = br#"{"id":"resp-1","item_id":"item-resp-1","description":"resp-1","nested":[{"id":"resp-1"}]}"#;
    let rewritten = parse(&rewrite_xai_websocket_downstream_ids(
        payload,
        "resp-1",
        "resp-1-xai-2",
        "",
        "",
    ));
    assert_eq!(rewritten["id"], "resp-1-xai-2");
    assert_eq!(rewritten["item_id"], "item-resp-1-xai-2");
    assert_eq!(rewritten["description"], "resp-1");
    assert_eq!(rewritten["nested"][0]["id"], "resp-1-xai-2");
}

#[test]
fn validates_and_normalizes_compaction_response() {
    let valid = br#"{"id":"cmp_compact","output":[{"type":"compaction","encrypted_content":"opaque-state"}]}"#;
    let state = validate_xai_websocket_compaction_response(valid).unwrap();
    assert_eq!(state.response_id, "resp_compact");
    assert_eq!(state.item["id"], "cmp_compact");
    assert_eq!(state.item["encrypted_content"], "opaque-state");

    for invalid in [
        b"{}".as_slice(),
        br#"{"id":"resp","output":[]}"#,
        br#"{"id":123,"output":[{"type":"compaction","encrypted_content":"x"}]}"#,
        br#"{"id":"resp","output":[{"type":"compaction"}]}"#,
    ] {
        assert!(validate_xai_websocket_compaction_response(invalid).is_err());
    }
}

#[test]
fn compaction_payload_replaces_input_and_previous_id() {
    let payload = build_xai_websocket_compaction_payload(
        br#"{"model":"grok","previous_response_id":"resp-old","input":[]}"#,
        br#"[{"id":"msg-1"}]"#,
    )
    .unwrap();
    let payload = parse(&payload);
    assert_eq!(payload["input"], json!([{"id":"msg-1"}]));
    assert!(payload.get("previous_response_id").is_none());
}

#[test]
fn warmup_completion_preserves_response_identity() {
    let created = br#"{"type":"response.created","sequence_number":7,"response":{"id":"resp-warm","status":"in_progress"}}"#;
    assert!(xai_websocket_generate_false(br#"{"generate":false}"#));
    let completed = parse(&build_xai_websocket_warmup_completed_payload(created));
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(completed["sequence_number"], 8);
    assert_eq!(completed["response"]["id"], "resp-warm");
    assert_eq!(completed["response"]["status"], "completed");
    assert_eq!(completed["response"]["output"], json!([]));
}

#[test]
fn message_too_big_is_request_scoped_and_not_retryable() {
    let error = map_xai_websocket_close(XAI_CLOSE_MESSAGE_TOO_BIG, "message too big");
    assert_eq!(StatusError::status_code(&error), 413);
    assert_eq!(error.code, "message_too_big");
    assert!(RequestScopedError::is_request_scoped(&error));
    assert!(!error.should_retry_send());

    let stale = map_xai_websocket_close(XAI_CLOSE_NORMAL, "stale");
    assert!(stale.should_retry_send());

    let write = XaiWebsocketError::transport("broken pipe", true);
    let mapped = map_xai_websocket_write_error(
        write,
        Some((XAI_CLOSE_MESSAGE_TOO_BIG, "peer rejected payload")),
    );
    assert_eq!(mapped.status, 413);
    assert!(!should_retry_xai_websocket_send(&mapped));
}

#[test]
fn parses_typed_and_bare_error_statuses_and_headers() {
    let typed = parse_xai_websocket_error(
        br#"{"type":"error","status":409,"headers":{"retry-after":"2","x-count":3},"error":{"code":"conflict","message":"busy"}}"#,
    )
    .unwrap();
    assert_eq!(typed.status, 409);
    assert_eq!(typed.headers["retry-after"], ["2"]);
    assert_eq!(typed.headers["x-count"], ["3"]);

    let bare = parse_xai_websocket_error(
        br#"{"error":{"message":"Request validation error: {\"code\":\"400\"}"}}"#,
    )
    .unwrap();
    assert_eq!(bare.status, 400);

    let body_error = parse_xai_websocket_error(
        br#"{"type":"error","status":503,"body":{"error":{"code":"overloaded","message":"later"}}}"#,
    )
    .unwrap();
    assert_eq!(body_error.code, "overloaded");

    let default_error = parse_xai_websocket_error(br#"{"type":"error","status":500}"#).unwrap();
    assert_eq!(default_error.code, "upstream_error");
}

#[test]
fn response_done_is_normalized_to_response_completed() {
    let normalized = normalize_xai_websocket_completion(
        br#"{"type":"response.done","response":{"id":"resp-1"}}"#,
    );
    assert_eq!(parse(&normalized)["type"], "response.completed");
}

#[test]
fn free_usage_exhaustion_carries_24_hour_retry_hint() {
    let error = parse_xai_websocket_error(
        br#"{"status":429,"error":{"code":"subscription:free-usage-exhausted","message":"included free usage exhausted"}}"#,
    )
    .unwrap();
    assert_eq!(error.retry_after, Some(Duration::from_secs(86_400)));
}

#[derive(Debug)]
struct FixedClock(i64);

impl XaiWebsocketClock for FixedClock {
    fn now_millis(&self) -> i64 {
        self.0
    }
}

#[test]
fn session_store_uses_injected_clock_values_and_expires_idle_state() {
    let store = XaiWebsocketSessionStore::default();
    let first = store.state("s1", 10).unwrap();
    let second = store.state("s1", 20).unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert!(!store.update_target("s1", "auth-1", "wss://one", 20));
    assert!(store.update_target("s1", "auth-2", "wss://two", 30));
    assert_eq!(store.expire_idle(130, Duration::from_millis(99)), 1);
}

#[derive(Clone, Debug, Default)]
struct ConnectionState {
    writes: Arc<Mutex<Vec<XaiWebsocketFrame>>>,
    closes: Arc<Mutex<Vec<String>>>,
}

struct ScriptedConnection {
    state: ConnectionState,
    reads: VecDeque<Result<XaiWebsocketFrame, XaiWebsocketError>>,
    write_error: Option<XaiWebsocketError>,
}

impl XaiWebsocketConnection for ScriptedConnection {
    fn write(&mut self, frame: XaiWebsocketFrame) -> Result<(), XaiWebsocketError> {
        if let Some(error) = self.write_error.take() {
            return Err(error);
        }
        self.state.writes.lock().unwrap().push(frame);
        Ok(())
    }

    fn read(&mut self) -> Result<XaiWebsocketFrame, XaiWebsocketError> {
        self.reads
            .pop_front()
            .unwrap_or_else(|| Err(XaiWebsocketError::transport("script exhausted", false)))
    }

    fn close(&mut self, reason: &str) -> Result<(), XaiWebsocketError> {
        self.state.closes.lock().unwrap().push(reason.to_owned());
        Ok(())
    }
}

#[derive(Default)]
struct ScriptedTransport {
    connections: Mutex<VecDeque<ScriptedConnection>>,
    requests: Mutex<Vec<XaiWebsocketConnectRequest>>,
}

impl ScriptedTransport {
    fn push(&self, connection: ScriptedConnection) {
        self.connections.lock().unwrap().push_back(connection);
    }
}

impl XaiWebsocketTransport for ScriptedTransport {
    fn connect(
        &self,
        request: XaiWebsocketConnectRequest,
    ) -> Result<Box<dyn XaiWebsocketConnection>, XaiWebsocketError> {
        self.requests.lock().unwrap().push(request);
        self.connections
            .lock()
            .unwrap()
            .pop_front()
            .map(|connection| Box::new(connection) as Box<dyn XaiWebsocketConnection>)
            .ok_or_else(|| XaiWebsocketError::transport("no scripted connection", false))
    }
}

fn connection(
    frames: impl IntoIterator<Item = XaiWebsocketFrame>,
) -> (ScriptedConnection, ConnectionState) {
    let state = ConnectionState::default();
    (
        ScriptedConnection {
            state: state.clone(),
            reads: frames.into_iter().map(Ok).collect(),
            write_error: None,
        },
        state,
    )
}

fn execution_request(payload: &[u8]) -> XaiWebsocketExecutionRequest {
    XaiWebsocketExecutionRequest {
        session_id: "session-1".into(),
        url: "wss://api.x.ai/v1/responses".into(),
        headers: Headers::new(),
        credential: XaiWebsocketCredential {
            auth_id: "auth-1".into(),
            bearer_token: "token".into(),
            proxy_url: Some("direct://".into()),
        },
        payload: payload.to_vec(),
    }
}

#[test]
fn executor_sends_response_create_and_completes_turn() {
    let transport = Arc::new(ScriptedTransport::default());
    let (connection, state) = connection([
        XaiWebsocketFrame::Ping(b"hello".to_vec()),
        XaiWebsocketFrame::Text(
            br#"{"type":"response.created","response":{"id":"resp-1","output":[]}}"#.to_vec(),
        ),
        XaiWebsocketFrame::Text(
            br#"{"type":"response.completed","response":{"id":"resp-1","output":[]}}"#.to_vec(),
        ),
    ]);
    transport.push(connection);
    let executor = XaiWebsocketsExecutor::new(
        transport.clone(),
        Arc::new(FixedClock(100)),
        Arc::new(XaiWebsocketSessionStore::default()),
    );
    let result = executor
        .execute_stream(execution_request(
            br#"{"model":"grok","stream":true,"input":[{"id":"msg-1"}]}"#,
        ))
        .unwrap();
    assert_eq!(result.events.len(), 2);
    let writes = state.writes.lock().unwrap();
    let XaiWebsocketFrame::Text(outbound) = &writes[0] else {
        panic!("expected text request")
    };
    assert_eq!(parse(outbound)["type"], "response.create");
    assert_eq!(parse(outbound)["store"], true);
    assert!(matches!(&writes[1], XaiWebsocketFrame::Pong(value) if value == b"hello"));
    assert_eq!(state.closes.lock().unwrap().as_slice(), ["turn_complete"]);
}

#[test]
fn executor_synthesizes_generate_false_completion() {
    let transport = Arc::new(ScriptedTransport::default());
    let (connection, _) = connection([XaiWebsocketFrame::Text(
        br#"{"type":"response.created","sequence_number":1,"response":{"id":"resp-warm"}}"#
            .to_vec(),
    )]);
    transport.push(connection);
    let executor = XaiWebsocketsExecutor::new(
        transport,
        Arc::new(FixedClock(1)),
        Arc::new(XaiWebsocketSessionStore::default()),
    );
    let result = executor
        .execute_stream(execution_request(
            br#"{"generate":false,"input":[{"id":"warm"}]}"#,
        ))
        .unwrap();
    assert_eq!(result.events.len(), 2);
    assert_eq!(parse(&result.events[1])["type"], "response.completed");
}

#[test]
fn executor_maps_message_too_big_close_without_reconnect() {
    let transport = Arc::new(ScriptedTransport::default());
    let (connection, _) = connection([XaiWebsocketFrame::Close {
        code: XAI_CLOSE_MESSAGE_TOO_BIG,
        reason: "large".into(),
    }]);
    transport.push(connection);
    let executor = XaiWebsocketsExecutor::new(
        transport.clone(),
        Arc::new(FixedClock(1)),
        Arc::new(XaiWebsocketSessionStore::default()),
    );
    let error = executor
        .execute_stream(execution_request(br#"{"input":"hello"}"#))
        .unwrap_err();
    assert_eq!(error.status, 413);
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}

#[test]
fn executor_reconnects_once_before_any_response_event() {
    let transport = Arc::new(ScriptedTransport::default());
    let (first, _) = connection([XaiWebsocketFrame::Close {
        code: XAI_CLOSE_NORMAL,
        reason: "stale".into(),
    }]);
    let (second, _) = connection([XaiWebsocketFrame::Text(
        br#"{"type":"response.completed","response":{"id":"resp-ok","output":[]}}"#.to_vec(),
    )]);
    transport.push(first);
    transport.push(second);
    let executor = XaiWebsocketsExecutor::new(
        transport.clone(),
        Arc::new(FixedClock(1)),
        Arc::new(XaiWebsocketSessionStore::default()),
    );
    let result = executor
        .execute_stream(execution_request(br#"{"input":"hello"}"#))
        .unwrap();
    assert_eq!(result.reconnects, 1);
    assert_eq!(transport.requests.lock().unwrap().len(), 2);
}

#[test]
fn auth_target_change_is_explicit_and_keeps_transcript_state() {
    let transport = Arc::new(ScriptedTransport::default());
    let (first, _) = connection([XaiWebsocketFrame::Text(
        br#"{"type":"response.completed","response":{"id":"resp-1","output":[{"id":"out-1"}]}}"#
            .to_vec(),
    )]);
    let (second, second_state) = connection([XaiWebsocketFrame::Text(
        br#"{"type":"response.completed","response":{"id":"resp-2","output":[]}}"#.to_vec(),
    )]);
    transport.push(first);
    transport.push(second);
    let sessions = Arc::new(XaiWebsocketSessionStore::default());
    let executor = XaiWebsocketsExecutor::new(transport, Arc::new(FixedClock(1)), sessions);
    executor
        .execute_stream(execution_request(br#"{"input":[{"id":"msg-1"}]}"#))
        .unwrap();

    let mut second_request =
        execution_request(br#"{"previous_response_id":"resp-1","input":[{"id":"msg-2"}]}"#);
    second_request.credential.auth_id = "auth-2".into();
    let result = executor.execute_stream(second_request).unwrap();
    assert!(result.target_changed);
    let writes = second_state.writes.lock().unwrap();
    let XaiWebsocketFrame::Text(payload) = &writes[0] else {
        panic!("expected request")
    };
    assert_eq!(parse(payload)["previous_response_id"], "resp-1");
}
