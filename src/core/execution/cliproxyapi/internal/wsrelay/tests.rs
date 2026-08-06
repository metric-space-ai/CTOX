// Origin: CTOX
// License: AGPL-3.0-only

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Map, Value};
use tokio::io::DuplexStream;
use tokio_tungstenite::tungstenite::protocol::frame::coding::{Data, OpCode};
use tokio_tungstenite::tungstenite::protocol::frame::Frame;
use tokio_tungstenite::tungstenite::Message as WebSocketMessage;
use tokio_tungstenite::WebSocketStream;

use super::*;

struct FixedAuthority(&'static str);

impl RelayAuthority for FixedAuthority {
    fn authorize(&self, _request: &RelayHandshake) -> Result<String, RelayError> {
        Ok(self.0.into())
    }
}

#[derive(Default)]
struct CaptureEvents {
    connected: Mutex<Vec<String>>,
    disconnected: Mutex<Vec<(String, RelayError)>>,
    unknown: Mutex<Vec<(String, String)>>,
}

impl RelayEventSink for CaptureEvents {
    fn connected(&self, provider: &str) {
        self.connected.lock().unwrap().push(provider.into());
    }

    fn disconnected(&self, provider: &str, cause: &RelayError) {
        self.disconnected
            .lock()
            .unwrap()
            .push((provider.into(), cause.clone()));
    }

    fn unknown_terminal_message(&self, provider: &str, message_id: &str) {
        self.unknown
            .lock()
            .unwrap()
            .push((provider.into(), message_id.into()));
    }
}

fn manager(events: Option<Arc<CaptureEvents>>, limits: RelayLimits) -> Manager {
    Manager::new(ManagerOptions {
        path: "v1/ws".into(),
        authority: Arc::new(FixedAuthority("  Studio-A  ")),
        events: events.map(|events| events as Arc<dyn RelayEventSink>),
        clock: None,
        limits,
    })
    .unwrap()
}

async fn websocket_pair() -> (WebSocketStream<DuplexStream>, WebSocketStream<DuplexStream>) {
    let (server_io, client_io) = tokio::io::duplex(256 * 1024);
    let (server, client) = tokio::join!(
        tokio_tungstenite::accept_async(server_io),
        tokio_tungstenite::client_async("ws://localhost/v1/ws", client_io)
    );
    (server.unwrap(), client.unwrap().0)
}

async fn connect(manager: &Manager) -> WebSocketStream<DuplexStream> {
    let (server, client) = websocket_pair().await;
    let provider = manager
        .accept_websocket(
            RelayHandshake {
                method: "get".into(),
                path: "/v1/ws".into(),
                ..RelayHandshake::default()
            },
            server,
        )
        .await
        .unwrap();
    assert_eq!(provider, "studio-a");
    client
}

fn payload(entries: impl IntoIterator<Item = (&'static str, Value)>) -> Map<String, Value> {
    entries
        .into_iter()
        .map(|(key, value)| (key.into(), value))
        .collect()
}

fn response_message(id: &str, kind: &str, body: &str) -> Message {
    Message::with_payload(
        id,
        kind,
        payload([
            ("status", json!(201)),
            ("headers", json!({"X-Test":["a","b"]})),
            ("body", json!(body)),
        ]),
    )
}

#[test]
fn message_wire_shape_and_terminal_contract_match_upstream() {
    let ping = Message::new("abc", MESSAGE_TYPE_PING);
    assert_eq!(
        serde_json::to_value(&ping).unwrap(),
        json!({"id":"abc","type":"ping"})
    );
    assert!(!ping.is_terminal());
    assert!(Message::new("abc", MESSAGE_TYPE_HTTP_RESPONSE).is_terminal());
    assert!(Message::new("abc", MESSAGE_TYPE_ERROR).is_terminal());
    assert!(Message::new("abc", MESSAGE_TYPE_STREAM_END).is_terminal());
}

#[test]
fn http_envelopes_preserve_headers_defaults_and_missing_payload() {
    let request = HttpRequest {
        method: "POST".into(),
        url: "https://example.invalid/generate".into(),
        headers: HeaderMap::from([("X-Multi".into(), vec!["one".into(), "two".into()])]),
        body: b"hello".to_vec(),
    };
    let sent_at = "2026-08-04T01:02:03.000000004Z".parse().unwrap();
    let encoded = encode_request(&request, sent_at);
    assert_eq!(encoded["body"], "hello");
    assert_eq!(encoded["headers"]["X-Multi"], json!(["one", "two"]));
    assert_eq!(encoded["sent_at"], "2026-08-04T01:02:03.000000004Z");

    assert_eq!(decode_response(None).status, 502);
    let decoded = decode_response(Some(&payload([
        ("status", json!(204)),
        ("headers", json!({"X-A":["1",4,"2"],"X-B":"single"})),
        ("body", json!("done")),
    ])));
    assert_eq!(decoded.status, 204);
    assert_eq!(decoded.headers["X-A"], ["1", "2"]);
    assert_eq!(decoded.headers["X-B"], ["single"]);
    assert_eq!(decoded.body, b"done");
    assert_eq!(decode_chunk(None), Vec::<u8>::new());
    assert_eq!(
        decode_error(None).to_string(),
        "wsrelay: unknown error (status=0)"
    );
}

#[tokio::test]
async fn websocket_loopback_relays_non_stream_response() {
    let events = Arc::new(CaptureEvents::default());
    let manager = manager(Some(Arc::clone(&events)), RelayLimits::default());
    let mut client = connect(&manager).await;
    let client_task = tokio::spawn(async move {
        let frame = client.next().await.unwrap().unwrap();
        let request: Message = serde_json::from_slice(&frame.into_data()).unwrap();
        assert_eq!(request.kind, MESSAGE_TYPE_HTTP_REQUEST);
        let body = request.payload.as_ref().unwrap()["body"].as_str().unwrap();
        assert_eq!(body, "request-body");
        client
            .send(WebSocketMessage::Text(
                serde_json::to_string(&response_message(
                    &request.id,
                    MESSAGE_TYPE_HTTP_RESPONSE,
                    "response-body",
                ))
                .unwrap()
                .into(),
            ))
            .await
            .unwrap();
    });
    let response = manager
        .non_stream(
            RelayCancellation::default(),
            "STUDIO-A",
            Some(&HttpRequest {
                method: "POST".into(),
                url: "https://example.invalid".into(),
                headers: HeaderMap::new(),
                body: b"request-body".to_vec(),
            }),
        )
        .await
        .unwrap();
    assert_eq!(response.status, 201);
    assert_eq!(response.headers["X-Test"], ["a", "b"]);
    assert_eq!(response.body, b"response-body");
    client_task.await.unwrap();
    assert_eq!(events.connected.lock().unwrap().as_slice(), ["studio-a"]);
    manager.stop(RelayCancellation::default()).await.unwrap();
}

#[tokio::test]
async fn fragmented_websocket_message_yields_ordered_stream_and_shutdown() {
    let manager = manager(None, RelayLimits::default());
    let mut client = connect(&manager).await;
    let client_task = tokio::spawn(async move {
        let request_frame = client.next().await.unwrap().unwrap();
        let request: Message = serde_json::from_slice(&request_frame.into_data()).unwrap();
        let start = serde_json::to_vec(&response_message(
            &request.id,
            MESSAGE_TYPE_STREAM_START,
            "",
        ))
        .unwrap();
        let midpoint = start.len() / 2;
        client
            .send(WebSocketMessage::Frame(Frame::message(
                start[..midpoint].to_vec(),
                OpCode::Data(Data::Text),
                false,
            )))
            .await
            .unwrap();
        client
            .send(WebSocketMessage::Frame(Frame::message(
                start[midpoint..].to_vec(),
                OpCode::Data(Data::Continue),
                true,
            )))
            .await
            .unwrap();
        for (kind, data) in [
            (MESSAGE_TYPE_STREAM_CHUNK, Some("one")),
            (MESSAGE_TYPE_STREAM_CHUNK, Some("two")),
            (MESSAGE_TYPE_STREAM_END, None),
        ] {
            let message = Message::with_payload(
                &request.id,
                kind,
                data.map_or_else(Map::new, |data| payload([("data", json!(data))])),
            );
            client
                .send(WebSocketMessage::Text(
                    serde_json::to_string(&message).unwrap().into(),
                ))
                .await
                .unwrap();
        }
    });
    let mut stream = manager
        .stream(
            RelayCancellation::default(),
            "studio-a",
            Some(&HttpRequest {
                method: "POST".into(),
                url: "https://example.invalid".into(),
                headers: HeaderMap::new(),
                body: Vec::new(),
            }),
        )
        .await
        .unwrap();
    let mut kinds = Vec::new();
    let mut data = Vec::new();
    while let Some(event) = stream.recv().await {
        kinds.push(event.kind);
        data.extend(event.payload);
    }
    assert_eq!(
        kinds,
        [
            MESSAGE_TYPE_STREAM_START,
            MESSAGE_TYPE_STREAM_CHUNK,
            MESSAGE_TYPE_STREAM_CHUNK,
            MESSAGE_TYPE_STREAM_END
        ]
    );
    assert_eq!(data, b"onetwo");
    client_task.await.unwrap();
    manager.stop(RelayCancellation::default()).await.unwrap();
    assert!(!manager.is_connected("studio-a"));
}

#[tokio::test]
async fn replacement_closes_old_session_without_removing_new_session() {
    let events = Arc::new(CaptureEvents::default());
    let manager = manager(Some(Arc::clone(&events)), RelayLimits::default());
    let mut old_client = connect(&manager).await;
    let mut new_client = connect(&manager).await;
    let old_closed = tokio::time::timeout(Duration::from_secs(1), old_client.next())
        .await
        .unwrap();
    assert!(matches!(
        old_closed,
        None | Some(Ok(WebSocketMessage::Close(_)))
    ));
    assert!(manager.is_connected("studio-a"));

    let request_task = {
        let manager = manager.clone();
        tokio::spawn(async move {
            manager
                .send(
                    RelayCancellation::default(),
                    "studio-a",
                    Message::new("new-session", MESSAGE_TYPE_HTTP_REQUEST),
                )
                .await
                .unwrap()
        })
    };
    let frame = new_client.next().await.unwrap().unwrap();
    let request: Message = serde_json::from_slice(&frame.into_data()).unwrap();
    assert_eq!(request.id, "new-session");
    drop(request_task.await.unwrap());
    manager.stop(RelayCancellation::default()).await.unwrap();
    assert_eq!(events.connected.lock().unwrap().len(), 2);
    assert!(!events.disconnected.lock().unwrap().is_empty());
}

#[tokio::test]
async fn request_cancellation_removes_pending_and_late_terminal_is_observable() {
    let events = Arc::new(CaptureEvents::default());
    let manager = manager(Some(Arc::clone(&events)), RelayLimits::default());
    let mut client = connect(&manager).await;
    let cancellation = RelayCancellation::default();
    let mut responses = manager
        .send(
            cancellation.clone(),
            "studio-a",
            Message::new("cancel-me", MESSAGE_TYPE_HTTP_REQUEST),
        )
        .await
        .unwrap();
    let _request = client.next().await.unwrap().unwrap();
    cancellation.cancel();
    assert!(
        tokio::time::timeout(Duration::from_secs(1), responses.recv())
            .await
            .unwrap()
            .is_none()
    );
    client
        .send(WebSocketMessage::Text(
            serde_json::to_string(&Message::new("cancel-me", MESSAGE_TYPE_HTTP_RESPONSE))
                .unwrap()
                .into(),
        ))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if !events.unknown.lock().unwrap().is_empty() {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        events.unknown.lock().unwrap().as_slice(),
        &[("studio-a".into(), "cancel-me".into())]
    );
    manager.stop(RelayCancellation::default()).await.unwrap();
}

#[tokio::test]
async fn response_backpressure_is_terminal_instead_of_dropping_frames() {
    let events = Arc::new(CaptureEvents::default());
    let limits = RelayLimits {
        response_capacity: 1,
        ..RelayLimits::default()
    };
    let manager = manager(Some(Arc::clone(&events)), limits);
    let mut client = connect(&manager).await;
    let _responses = manager
        .send(
            RelayCancellation::default(),
            "studio-a",
            Message::new("slow", MESSAGE_TYPE_HTTP_REQUEST),
        )
        .await
        .unwrap();
    let _request = client.next().await.unwrap().unwrap();
    for index in 0..3 {
        client
            .send(WebSocketMessage::Text(
                serde_json::to_string(&Message::with_payload(
                    "slow",
                    MESSAGE_TYPE_STREAM_CHUNK,
                    payload([("data", json!(index))]),
                ))
                .unwrap()
                .into(),
            ))
            .await
            .unwrap();
    }
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if events
                .disconnected
                .lock()
                .unwrap()
                .iter()
                .any(|(_, cause)| matches!(cause, RelayError::Backpressure(_)))
            {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(!manager.is_connected("studio-a"));
}

#[tokio::test]
async fn stop_closes_pending_requests_and_waits_for_session_exit() {
    let manager = manager(None, RelayLimits::default());
    let mut client = connect(&manager).await;
    let mut responses = manager
        .send(
            RelayCancellation::default(),
            "studio-a",
            Message::new("pending", MESSAGE_TYPE_HTTP_REQUEST),
        )
        .await
        .unwrap();
    let _request = client.next().await.unwrap().unwrap();
    manager.stop(RelayCancellation::default()).await.unwrap();
    let terminal = responses.recv().await.unwrap();
    assert_eq!(terminal.kind, MESSAGE_TYPE_ERROR);
    assert!(responses.recv().await.is_none());
    assert!(matches!(
        manager
            .send(
                RelayCancellation::default(),
                "studio-a",
                Message::new("later", MESSAGE_TYPE_HTTP_REQUEST),
            )
            .await,
        Err(RelayError::NotConnected(_))
    ));
}

#[tokio::test]
async fn handshake_policy_is_fail_closed_before_transport_registration() {
    let manager = manager(None, RelayLimits::default());
    let (server, _client) = websocket_pair().await;
    let error = manager
        .accept_websocket(
            RelayHandshake {
                method: "POST".into(),
                path: "/wrong".into(),
                ..RelayHandshake::default()
            },
            server,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, RelayError::InvalidRequest(_)));
    assert!(manager.connected_providers().is_empty());
}

#[tokio::test]
async fn heartbeat_enforces_injected_liveness_deadline() {
    let events = Arc::new(CaptureEvents::default());
    let limits = RelayLimits {
        heartbeat_interval: Duration::from_millis(10),
        read_timeout: Duration::from_millis(20),
        ..RelayLimits::default()
    };
    let manager = manager(Some(Arc::clone(&events)), limits);
    let _idle_client = connect(&manager).await;
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if events
                .disconnected
                .lock()
                .unwrap()
                .iter()
                .any(|(_, cause)| matches!(cause, RelayError::TimedOut("read")))
            {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(!manager.is_connected("studio-a"));
}
