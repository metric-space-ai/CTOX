// ref: internal/client/codex/live/live_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::json;

use super::*;
use crate::sdk::cliproxy::auth::Auth;
use crate::sdk::cliproxy::executor::Headers;

#[test]
fn prepare_call_request_rewrites_multipart() {
    let body = b"--edge\r\nContent-Disposition: form-data; name=\"sdp\"\r\n\r\nv=0\r\n--edge\r\nContent-Disposition: form-data; name=\"session\"\r\n\r\n{\"model\":\"gpt-live-test\"}\r\n--edge--\r\n";
    let prepared = prepare_call_request(body, "multipart/form-data; boundary=edge").unwrap();
    assert_eq!(prepared.content_type, "application/json");
    assert_eq!(prepared.model, "gpt-live-test");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&prepared.body).unwrap(),
        json!({"sdp":"v=0","session":{"model":"gpt-live-test"}})
    );
}

#[test]
fn prepare_call_request_preserves_raw_sdp() {
    let prepared = prepare_call_request(b"v=0\r\n", "application/sdp").unwrap();
    assert_eq!(prepared.body, b"v=0\r\n");
    assert_eq!(prepared.model, DEFAULT_LIVE_MODEL);
}

#[test]
fn media_relay_wraps_raw_sdp_for_backend() {
    let (body, content_type) = replace_call_request_sdp(b"v=0", "application/sdp", "v=1").unwrap();
    assert_eq!(content_type, "application/json");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
        json!({"sdp":"v=1"})
    );
}

#[test]
fn prepare_call_request_rejects_invalid_multipart() {
    assert!(prepare_call_request(b"broken", "multipart/form-data").is_err());
    assert!(prepare_call_request(
        b"--x\r\nContent-Disposition: form-data; name=\"session\"\r\n\r\n{}\r\n--x--\r\n",
        "multipart/form-data; boundary=x"
    )
    .is_err());
}

#[test]
fn headers_for_logging_redacts_attestation() {
    let headers = Headers::from_iter([
        ("x-oai-attestation".to_owned(), vec!["secret".to_owned()]),
        ("Originator".to_owned(), vec!["codex".to_owned()]),
    ]);
    let safe = headers_for_logging(&headers);
    assert_eq!(header_first(&safe, "x-oai-attestation"), Some("[REDACTED]"));
    assert_eq!(header_first(&safe, "originator"), Some("codex"));
}

#[test]
fn protocol_headers_are_whitelisted_and_account_is_injected() {
    let source = Headers::from_iter([
        ("Authorization".to_owned(), vec!["secret".to_owned()]),
        ("Thread-Id".to_owned(), vec!["thread".to_owned()]),
    ]);
    let mut selected = Auth::default();
    selected
        .metadata
        .insert("account_id".to_owned(), json!("acct"));
    let mut headers = protocol_headers(&source);
    set_account_header(&mut headers, &selected);
    assert!(header_first(&headers, "authorization").is_none());
    assert_eq!(header_first(&headers, "thread-id"), Some("thread"));
    assert_eq!(header_first(&headers, "chatgpt-account-id"), Some("acct"));
}

#[test]
fn media_credential_name_uses_safe_identity() {
    let mut auth = Auth::default();
    auth.file_name = "/private/credentials/codex.json".to_owned();
    assert_eq!(media_credential_name(Some(&auth), "index"), "codex.json");
    auth.label = "Work".to_owned();
    assert_eq!(media_credential_name(Some(&auth), "index"), "Work");
}

#[test]
fn proxy_url_prefers_credential_override() {
    let mut auth = Auth::default();
    assert_eq!(
        proxy_url_for_auth("socks5://default", &auth),
        "socks5://default"
    );
    auth.proxy_url = " http://credential ".to_owned();
    assert_eq!(
        proxy_url_for_auth("socks5://default", &auth),
        "http://credential"
    );
}

#[test]
fn session_store_claims_releases_and_expires() {
    let store = SessionStore::new(Duration::from_millis(10));
    let call_id = CallId::parse("call_1").unwrap();
    let stored = store.put(
        call_id.clone(),
        LiveSession::new("auth".to_owned(), "model".to_owned(), None, 100),
    );
    let (claimed, status) = store.claim(&call_id);
    assert_eq!(status, SessionClaim::Acquired);
    assert_eq!(store.claim(&call_id).1, SessionClaim::Busy);
    assert_eq!(store.expire_due(1_000), 0, "claimed sessions do not expire");
    assert!(store.release(claimed.as_ref().unwrap(), 1_000));
    assert_eq!(store.expire_due(1_009), 0);
    assert_eq!(store.expire_due(1_010), 1);
    assert!(!store.complete(&stored, "stale"));
}

struct FlagResource(AtomicBool);

impl SessionResource for FlagResource {
    fn close(&self) {
        self.0.store(true, Ordering::Release);
    }
}

#[test]
fn session_store_close_all_releases_resources() {
    let store = SessionStore::default();
    let resource = Arc::new(FlagResource(AtomicBool::new(false)));
    let session = LiveSession::new("a".to_owned(), "m".to_owned(), None, 0);
    session.resources.add(resource.clone());
    store.put(CallId::parse("id").unwrap(), session);
    assert_eq!(store.close_all("server_stopped"), 1);
    assert!(resource.0.load(Ordering::Acquire));
}

#[test]
fn sideband_url_shapes_and_location_parsing() {
    let id = CallId::parse("call_42").unwrap();
    assert_eq!(
        build_sideband_url("wss://api/v1/", SidebandStyle::Frameless, &id),
        "wss://api/v1/live/call_42"
    );
    assert_eq!(
        build_sideband_url("wss://api/v1", SidebandStyle::RealtimeCalls, &id),
        "wss://api/v1/realtime/calls/call_42"
    );
    assert_eq!(
        call_id_from_location("https://api/v1/realtime/calls/call_42"),
        Some(id.clone())
    );
    assert_eq!(
        call_id_from_location("https://api/v1/realtime?call_id=call_42"),
        Some(id)
    );
}

struct ClosedEndpoint {
    closed: AtomicBool,
}

impl WebSocketEndpoint for ClosedEndpoint {
    fn read(&self) -> Result<WebSocketFrame, WebSocketClose> {
        Err(WebSocketClose::Normal)
    }

    fn write(&self, _frame: WebSocketFrame) -> Result<(), WebSocketClose> {
        Ok(())
    }

    fn close(&self, _close: WebSocketClose) {
        self.closed.store(true, Ordering::Release);
    }
}

struct FixtureConnector {
    request: Mutex<Option<SidebandConnectRequest>>,
    endpoint: Arc<ClosedEndpoint>,
}

impl SidebandConnector for FixtureConnector {
    fn connect(&self, request: &SidebandConnectRequest) -> Result<SidebandHandshake, LiveError> {
        *self.request.lock().unwrap() = Some(request.clone());
        Ok(SidebandHandshake {
            endpoint: self.endpoint.clone(),
            subprotocol: Some("realtime".to_owned()),
        })
    }
}

#[test]
fn sideband_join_pins_auth_and_drop_allows_reconnect() {
    let sessions = Arc::new(SessionStore::new(Duration::from_millis(100)));
    let call_id = CallId::parse("call_join").unwrap();
    sessions.put(
        call_id.clone(),
        LiveSession::new("pinned-auth".to_owned(), "live-model".to_owned(), None, 0),
    );
    let endpoint = Arc::new(ClosedEndpoint {
        closed: AtomicBool::new(false),
    });
    let connector = Arc::new(FixtureConnector {
        request: Mutex::new(None),
        endpoint: endpoint.clone(),
    });
    let client = SidebandClient::new(sessions.clone(), connector.clone(), "wss://api/v1");
    let connection = client
        .join(
            SidebandStyle::Frameless,
            &call_id,
            vec!["realtime".to_owned()],
            10,
        )
        .unwrap();
    assert_eq!(connection.subprotocol.as_deref(), Some("realtime"));
    let request = connector.request.lock().unwrap().clone().unwrap();
    assert_eq!(request.auth_id, "pinned-auth");
    assert_eq!(request.model, "live-model");
    assert_eq!(sessions.claim(&call_id).1, SessionClaim::Busy);
    drop(connection);
    assert!(endpoint.closed.load(Ordering::Acquire));
    assert_eq!(sessions.claim(&call_id).1, SessionClaim::Acquired);
}

struct UnauthorizedThenConnected {
    requests: Mutex<Vec<SidebandConnectRequest>>,
    endpoint: Arc<ClosedEndpoint>,
}

impl SidebandConnector for UnauthorizedThenConnected {
    fn connect(&self, request: &SidebandConnectRequest) -> Result<SidebandHandshake, LiveError> {
        self.requests.lock().unwrap().push(request.clone());
        if request.auth_id != "fresh" {
            return Err(LiveError::new(LiveErrorKind::Unauthorized, "unauthorized"));
        }
        Ok(SidebandHandshake {
            endpoint: self.endpoint.clone(),
            subprotocol: None,
        })
    }
}

#[test]
fn sideband_refreshes_unauthorized_home_handshake_once() {
    let sessions = Arc::new(SessionStore::default());
    let call_id = CallId::parse("call_refresh_sideband").unwrap();
    sessions.put(
        call_id.clone(),
        LiveSession::new("stale".to_owned(), "live-model".to_owned(), None, 0),
    );
    let connector = Arc::new(UnauthorizedThenConnected {
        requests: Mutex::new(Vec::new()),
        endpoint: Arc::new(ClosedEndpoint {
            closed: AtomicBool::new(false),
        }),
    });
    let refresher = Arc::new(FixtureRefresher {
        calls: Mutex::new(Vec::new()),
    });
    let connection = SidebandClient::new(sessions, connector.clone(), "wss://api/v1")
        .with_credential_refresher(refresher.clone())
        .join(SidebandStyle::Frameless, &call_id, Vec::new(), 10)
        .expect("retry succeeds");
    assert_eq!(
        connector
            .requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| request.auth_id.as_str())
            .collect::<Vec<_>>(),
        vec!["stale", "fresh"]
    );
    assert_eq!(refresher.calls.lock().unwrap().len(), 2);
    drop(connection);
}

#[test]
fn sideband_releases_claim_when_refresh_fails() {
    let sessions = Arc::new(SessionStore::default());
    let call_id = CallId::parse("call_refresh_failure").unwrap();
    sessions.put(
        call_id.clone(),
        LiveSession::new("stale".to_owned(), "live-model".to_owned(), None, 0),
    );
    let connector = Arc::new(UnauthorizedThenConnected {
        requests: Mutex::new(Vec::new()),
        endpoint: Arc::new(ClosedEndpoint {
            closed: AtomicBool::new(false),
        }),
    });
    let refresher = Arc::new(FailingRefresher {
        reports: Mutex::new(Vec::new()),
    });
    let error = SidebandClient::new(sessions.clone(), connector, "wss://api/v1")
        .with_credential_refresher(refresher.clone())
        .join(SidebandStyle::Frameless, &call_id, Vec::new(), 10)
        .err()
        .expect("refresh fails");
    assert_eq!(error.kind, LiveErrorKind::Unavailable);
    assert_eq!(*refresher.reports.lock().unwrap(), ["stale"]);
    assert_eq!(sessions.claim(&call_id).1, SessionClaim::Acquired);
}

struct FixtureTransport {
    seen: Mutex<Option<LiveHttpRequest>>,
}

struct UnauthorizedThenSuccessTransport {
    requests: Mutex<Vec<LiveHttpRequest>>,
}

struct AlwaysUnauthorizedTransport;

impl LiveTransport for AlwaysUnauthorizedTransport {
    fn execute<'a>(&'a self, _: LiveHttpRequest) -> LiveTransportFuture<'a> {
        Box::pin(async {
            Ok(LiveHttpResponse {
                status: 401,
                headers: Headers::new(),
                body: Vec::new(),
            })
        })
    }
}

impl LiveTransport for UnauthorizedThenSuccessTransport {
    fn execute<'a>(&'a self, request: LiveHttpRequest) -> LiveTransportFuture<'a> {
        let status = if request.auth_id == "fresh" { 201 } else { 401 };
        self.requests.lock().unwrap().push(request);
        Box::pin(async move {
            Ok(LiveHttpResponse {
                status,
                headers: Headers::from_iter([(
                    "Location".to_owned(),
                    vec!["call_refresh".to_owned()],
                )]),
                body: Vec::new(),
            })
        })
    }
}

struct FixtureRefresher {
    calls: Mutex<Vec<(String, String, bool)>>,
}

impl LiveCredentialRefresher for FixtureRefresher {
    fn report_unauthorized(&self, auth_id: &str, model: &str) {
        self.calls
            .lock()
            .unwrap()
            .push((auth_id.to_owned(), model.to_owned(), false));
    }

    fn refresh_after_unauthorized(
        &self,
        auth_id: &str,
        model: &str,
        current: Option<&Auth>,
    ) -> Result<Auth, LiveError> {
        self.calls
            .lock()
            .unwrap()
            .push((auth_id.to_owned(), model.to_owned(), current.is_some()));
        let mut auth = current.cloned().unwrap_or_default();
        auth.id = "fresh".to_owned();
        Ok(auth)
    }
}

struct FailingRefresher {
    reports: Mutex<Vec<String>>,
}

impl LiveCredentialRefresher for FailingRefresher {
    fn report_unauthorized(&self, auth_id: &str, _: &str) {
        self.reports.lock().unwrap().push(auth_id.to_owned());
    }

    fn refresh_after_unauthorized(
        &self,
        _: &str,
        _: &str,
        _: Option<&Auth>,
    ) -> Result<Auth, LiveError> {
        Err(LiveError::new(LiveErrorKind::Unavailable, "refresh failed"))
    }
}

struct TrackingMediaSession {
    closed: Mutex<Vec<String>>,
}

impl MediaRelaySession for TrackingMediaSession {
    fn accept_upstream_answer<'a>(&'a self, answer: &'a str) -> MediaFuture<'a, String> {
        Box::pin(async move { Ok(answer.to_owned()) })
    }

    fn set_call_id(&self, _: &str) {}

    fn set_close_handler(&self, _: Arc<dyn Fn(&str) + Send + Sync>) {}

    fn close(&self, reason: &str) {
        self.closed.lock().unwrap().push(reason.to_owned());
    }
}

struct TrackingMediaRelay(Arc<TrackingMediaSession>);

impl MediaRelay for TrackingMediaRelay {
    fn new_session<'a>(
        &'a self,
        _: &'a str,
        _: &'a MediaRoute,
    ) -> MediaFuture<'a, (Arc<dyn MediaRelaySession>, String)> {
        let session: Arc<dyn MediaRelaySession> = self.0.clone();
        Box::pin(async move { Ok((session, "v=upstream-offer".to_owned())) })
    }
}

#[tokio::test]
async fn bootstrap_refreshes_unauthorized_home_selection_once() {
    let transport = Arc::new(UnauthorizedThenSuccessTransport {
        requests: Mutex::new(Vec::new()),
    });
    let refresher = Arc::new(FixtureRefresher {
        calls: Mutex::new(Vec::new()),
    });
    let sessions = Arc::new(SessionStore::default());
    let client = LiveClient::new(transport.clone(), None, sessions.clone(), "")
        .with_credential_refresher(refresher.clone());
    let mut auth = Auth::default();
    auth.id = "stale".to_owned();
    let response = client
        .bootstrap(
            br#"{"session":{"model":"gpt-live"},"sdp":"v=offer"}"#,
            "application/json",
            &Headers::new(),
            &auth,
            "stale",
            0,
        )
        .await
        .expect("retry succeeds");
    assert_eq!(response.status, 201);
    assert_eq!(
        transport
            .requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| request.auth_id.as_str())
            .collect::<Vec<_>>(),
        vec!["stale", "fresh"]
    );
    assert_eq!(refresher.calls.lock().unwrap().len(), 2);
    assert_eq!(
        sessions
            .peek(&CallId::parse("call_refresh").unwrap())
            .expect("stored")
            .auth_id,
        "fresh"
    );
}

#[tokio::test]
async fn bootstrap_reports_both_unauthorized_responses() {
    let refresher = Arc::new(FixtureRefresher {
        calls: Mutex::new(Vec::new()),
    });
    let client = LiveClient::new(
        Arc::new(AlwaysUnauthorizedTransport),
        None,
        Arc::new(SessionStore::default()),
        "",
    )
    .with_credential_refresher(refresher.clone());
    let mut auth = Auth::default();
    auth.id = "stale".to_owned();
    let response = client
        .bootstrap(
            br#"{"session":{"model":"gpt-live"},"sdp":"v=offer"}"#,
            "application/json",
            &Headers::new(),
            &auth,
            "stale",
            0,
        )
        .await
        .expect("401 response is forwarded");
    assert_eq!(response.status, 401);
    let calls = refresher.calls.lock().unwrap();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0].0, "stale");
    assert_eq!(calls[2].0, "fresh");
}

#[tokio::test]
async fn bootstrap_closes_media_when_refresh_fails() {
    let media_session = Arc::new(TrackingMediaSession {
        closed: Mutex::new(Vec::new()),
    });
    let refresher = Arc::new(FailingRefresher {
        reports: Mutex::new(Vec::new()),
    });
    let client = LiveClient::new(
        Arc::new(AlwaysUnauthorizedTransport),
        Some(Arc::new(TrackingMediaRelay(media_session.clone()))),
        Arc::new(SessionStore::default()),
        "",
    )
    .with_credential_refresher(refresher);
    let mut auth = Auth::default();
    auth.id = "stale".to_owned();
    let error = client
        .bootstrap(
            br#"{"session":{"model":"gpt-live"},"sdp":"v=offer"}"#,
            "application/json",
            &Headers::new(),
            &auth,
            "stale",
            0,
        )
        .await
        .expect_err("refresh fails");
    assert_eq!(error.kind, LiveErrorKind::Unavailable);
    assert_eq!(*media_session.closed.lock().unwrap(), ["refresh_failed"]);
}

impl LiveTransport for FixtureTransport {
    fn execute<'a>(&'a self, request: LiveHttpRequest) -> LiveTransportFuture<'a> {
        *self.seen.lock().unwrap() = Some(request);
        Box::pin(async {
            Ok(LiveHttpResponse {
                status: 201,
                headers: Headers::from_iter([("Location".to_owned(), vec!["call_9".to_owned()])]),
                body: b"v=answer".to_vec(),
            })
        })
    }
}

#[tokio::test]
async fn bootstrap_uses_model_and_pins_session_auth() {
    let transport = Arc::new(FixtureTransport {
        seen: Mutex::new(None),
    });
    let sessions = Arc::new(SessionStore::default());
    let client = LiveClient::new(transport.clone(), None, sessions.clone(), "");
    let mut auth = Auth::default();
    auth.id = "oauth-a".to_owned();
    let response = client
        .bootstrap(
            br#"{"session":{"model":"gpt-live-custom"},"sdp":"v=offer"}"#,
            "application/json",
            &Headers::new(),
            &auth,
            "idx",
            10,
        )
        .await
        .unwrap();
    assert_eq!(response.status, 201);
    assert_eq!(
        transport.seen.lock().unwrap().as_ref().unwrap().model,
        "gpt-live-custom"
    );
    let stored = sessions.peek(&CallId::parse("call_9").unwrap()).unwrap();
    assert_eq!(stored.auth_id, "oauth-a");
}
