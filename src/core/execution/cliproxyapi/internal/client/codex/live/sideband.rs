// ref: internal/client/codex/live/sideband.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};
use url::Url;

use super::{LiveCredentialRefresher, LiveError, LiveErrorKind, MediaRelaySession};

pub const DEFAULT_SIDEBAND_API_BASE_URL: &str = "wss://api.openai.com/v1";
pub const DEFAULT_SESSION_LIFETIME: Duration = Duration::from_secs(60 * 60);

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CallId(String);

impl CallId {
    pub fn parse(value: impl AsRef<str>) -> Result<Self, LiveError> {
        let value = value.as_ref().trim();
        let valid = !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-');
        valid
            .then(|| Self(value.to_owned()))
            .ok_or_else(|| LiveError::new(LiveErrorKind::BadRequest, "Invalid Codex live call ID"))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CallId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

pub trait SessionResource: Send + Sync {
    fn close(&self);
}

pub struct SessionResources {
    closed: AtomicBool,
    resources: Mutex<Vec<Arc<dyn SessionResource>>>,
}

impl SessionResources {
    #[must_use]
    pub fn new() -> Self {
        Self {
            closed: AtomicBool::new(false),
            resources: Mutex::new(Vec::new()),
        }
    }

    pub fn add(&self, resource: Arc<dyn SessionResource>) {
        if self.closed.load(Ordering::Acquire) {
            resource.close();
            return;
        }
        let mut resources = self
            .resources
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.closed.load(Ordering::Acquire) {
            drop(resources);
            resource.close();
        } else {
            resources.push(resource);
        }
    }

    pub fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let resources = std::mem::take(
            &mut *self
                .resources
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        for resource in resources {
            resource.close();
        }
    }

    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

impl Default for SessionResources {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct LiveSession {
    pub auth_id: String,
    pub model: String,
    pub media: Option<Arc<dyn MediaRelaySession>>,
    pub resources: Arc<SessionResources>,
    created_at_millis: u64,
    call_id: Option<CallId>,
    token: u64,
}

impl LiveSession {
    #[must_use]
    pub fn new(
        auth_id: String,
        model: String,
        media: Option<Arc<dyn MediaRelaySession>>,
        created_at_millis: u64,
    ) -> Self {
        Self {
            auth_id,
            model,
            media,
            resources: Arc::new(SessionResources::new()),
            created_at_millis,
            call_id: None,
            token: 0,
        }
    }

    #[must_use]
    pub fn call_id(&self) -> Option<&CallId> {
        self.call_id.as_ref()
    }

    #[must_use]
    pub fn token(&self) -> u64 {
        self.token
    }

    fn end(&self, reason: &str) {
        self.resources.close();
        if let Some(media) = &self.media {
            media.close(reason);
        }
    }
}

struct StoredSession {
    session: LiveSession,
    claimed: bool,
    expires_at_millis: u64,
}

struct StoreState {
    next: u64,
    sessions: BTreeMap<CallId, StoredSession>,
}

pub struct SessionStore {
    lifetime_millis: u64,
    state: Mutex<StoreState>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionClaim {
    Missing,
    Busy,
    Acquired,
}

impl SessionStore {
    #[must_use]
    pub fn new(lifetime: Duration) -> Self {
        let lifetime_millis = u64::try_from(lifetime.as_millis()).unwrap_or(u64::MAX);
        Self {
            lifetime_millis: lifetime_millis.max(1),
            state: Mutex::new(StoreState {
                next: 0,
                sessions: BTreeMap::new(),
            }),
        }
    }

    pub fn put(&self, call_id: CallId, mut session: LiveSession) -> LiveSession {
        let previous = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.next = state.next.wrapping_add(1).max(1);
            session.call_id = Some(call_id.clone());
            session.token = state.next;
            let expires_at_millis = session
                .created_at_millis
                .saturating_add(self.lifetime_millis);
            state.sessions.insert(
                call_id,
                StoredSession {
                    session: session.clone(),
                    claimed: false,
                    expires_at_millis,
                },
            )
        };
        if let Some(previous) = previous {
            previous.session.end("session_replaced");
        }
        session
    }

    #[must_use]
    pub fn claim(&self, call_id: &CallId) -> (Option<LiveSession>, SessionClaim) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(entry) = state.sessions.get_mut(call_id) else {
            return (None, SessionClaim::Missing);
        };
        if entry.claimed {
            return (None, SessionClaim::Busy);
        }
        entry.claimed = true;
        (Some(entry.session.clone()), SessionClaim::Acquired)
    }

    pub fn release(&self, session: &LiveSession, now_millis: u64) -> bool {
        let Some(call_id) = &session.call_id else {
            return false;
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(entry) = state.sessions.get_mut(call_id) else {
            return false;
        };
        if entry.session.token != session.token || !entry.claimed {
            return false;
        }
        entry.claimed = false;
        entry.expires_at_millis = now_millis.saturating_add(self.lifetime_millis);
        true
    }

    pub fn complete(&self, session: &LiveSession, reason: &str) -> bool {
        let Some(call_id) = &session.call_id else {
            session.end(reason);
            return false;
        };
        let removed = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state
                .sessions
                .get(call_id)
                .is_some_and(|entry| entry.session.token == session.token)
            {
                state.sessions.remove(call_id)
            } else {
                None
            }
        };
        if let Some(entry) = removed {
            entry.session.end(reason);
            true
        } else {
            false
        }
    }

    pub fn expire_due(&self, now_millis: u64) -> usize {
        let expired = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let keys: Vec<_> = state
                .sessions
                .iter()
                .filter(|(_, entry)| !entry.claimed && entry.expires_at_millis <= now_millis)
                .map(|(call_id, _)| call_id.clone())
                .collect();
            keys.into_iter()
                .filter_map(|call_id| state.sessions.remove(&call_id))
                .collect::<Vec<_>>()
        };
        let count = expired.len();
        for entry in expired {
            entry.session.end("session_expired");
        }
        count
    }

    pub fn close_all(&self, reason: &str) -> usize {
        let sessions = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::mem::take(&mut state.sessions)
        };
        let count = sessions.len();
        for (_, entry) in sessions {
            entry.session.end(reason);
        }
        count
    }

    #[must_use]
    pub fn peek(&self, call_id: &CallId) -> Option<LiveSession> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .sessions
            .get(call_id)
            .map(|entry| entry.session.clone())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .sessions
            .len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new(DEFAULT_SESSION_LIFETIME)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SidebandStyle {
    Frameless,
    RealtimeCalls,
    RealtimeQuery,
}

pub fn sideband_target(
    path: &str,
    query_call_id: Option<&str>,
) -> Result<(SidebandStyle, CallId), LiveError> {
    let normalized = path.trim_end_matches('/');
    if let Some(id) = normalized.strip_prefix("/live/") {
        return Ok((SidebandStyle::Frameless, CallId::parse(id)?));
    }
    if let Some(id) = normalized.strip_prefix("/realtime/calls/") {
        return Ok((SidebandStyle::RealtimeCalls, CallId::parse(id)?));
    }
    Ok((
        SidebandStyle::RealtimeQuery,
        CallId::parse(query_call_id.unwrap_or_default())?,
    ))
}

#[must_use]
pub fn build_sideband_url(base_url: &str, style: SidebandStyle, call_id: &CallId) -> String {
    let root = base_url.trim_end_matches('/');
    match style {
        SidebandStyle::RealtimeCalls => format!("{root}/realtime/calls/{call_id}"),
        SidebandStyle::RealtimeQuery => format!(
            "{root}/realtime?intent=quicksilver&call_id={}",
            utf8_percent_encode(call_id.as_str(), NON_ALPHANUMERIC)
        ),
        SidebandStyle::Frameless => format!("{root}/live/{call_id}"),
    }
}

#[must_use]
pub fn websocket_http_url(raw_url: &str) -> String {
    let Ok(mut url) = Url::parse(raw_url) else {
        return raw_url.to_owned();
    };
    match url.scheme().to_ascii_lowercase().as_str() {
        "ws" => {
            let _ = url.set_scheme("http");
        }
        "wss" => {
            let _ = url.set_scheme("https");
        }
        _ => {}
    }
    url.to_string()
}

#[must_use]
pub fn call_id_from_location(location: &str) -> Option<CallId> {
    let location = location.trim();
    if let Ok(call_id) = CallId::parse(location) {
        return Some(call_id);
    }
    let parsed = Url::parse(location).ok()?;
    if let Some(call_id) = parsed
        .query_pairs()
        .find(|(name, _)| name == "call_id")
        .and_then(|(_, value)| CallId::parse(value.as_ref()).ok())
    {
        return Some(call_id);
    }
    let segments: Vec<_> = parsed.path_segments()?.collect();
    let (&call_id, &previous) = (
        segments.last()?,
        segments.get(segments.len().checked_sub(2)?)?,
    );
    (previous == "live" || previous == "calls")
        .then(|| CallId::parse(call_id).ok())
        .flatten()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebSocketMessageType {
    Text,
    Binary,
    Ping,
    Pong,
    Close,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WebSocketFrame {
    pub message_type: WebSocketMessageType,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WebSocketClose {
    Normal,
    GoingAway(String),
    Protocol(u16, String),
    Internal(String),
}

pub trait WebSocketEndpoint: Send + Sync {
    fn read(&self) -> Result<WebSocketFrame, WebSocketClose>;
    fn write(&self, frame: WebSocketFrame) -> Result<(), WebSocketClose>;
    fn close(&self, close: WebSocketClose);
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SidebandConnectRequest {
    pub url: String,
    pub http_url: String,
    pub auth_id: String,
    pub model: String,
    pub call_id: CallId,
    pub subprotocols: Vec<String>,
}

#[derive(Clone)]
pub struct SidebandHandshake {
    pub endpoint: Arc<dyn WebSocketEndpoint>,
    pub subprotocol: Option<String>,
}

pub trait SidebandConnector: Send + Sync {
    fn connect(&self, request: &SidebandConnectRequest) -> Result<SidebandHandshake, LiveError>;
}

/// Claims the bootstrap session and performs the sideband handshake with the
/// exact auth/model pair captured at bootstrap time. A failed join or dropped
/// connection releases the claim for reconnect; `complete` consumes it.
pub struct SidebandClient {
    sessions: Arc<SessionStore>,
    connector: Arc<dyn SidebandConnector>,
    base_url: String,
    refresher: Option<Arc<dyn LiveCredentialRefresher>>,
}

impl SidebandClient {
    #[must_use]
    pub fn new(
        sessions: Arc<SessionStore>,
        connector: Arc<dyn SidebandConnector>,
        base_url: impl Into<String>,
    ) -> Self {
        let base_url = base_url.into();
        Self {
            sessions,
            connector,
            refresher: None,
            base_url: if base_url.trim().is_empty() {
                DEFAULT_SIDEBAND_API_BASE_URL.to_owned()
            } else {
                base_url.trim().to_owned()
            },
        }
    }

    #[must_use]
    pub fn with_credential_refresher(
        mut self,
        refresher: Arc<dyn LiveCredentialRefresher>,
    ) -> Self {
        self.refresher = Some(refresher);
        self
    }

    pub fn join(
        &self,
        style: SidebandStyle,
        call_id: &CallId,
        subprotocols: Vec<String>,
        now_millis: u64,
    ) -> Result<SidebandConnection, LiveError> {
        let (session, claim) = self.sessions.claim(call_id);
        let session = match (session, claim) {
            (Some(session), SessionClaim::Acquired) => session,
            (_, SessionClaim::Busy) => {
                return Err(LiveError::new(
                    LiveErrorKind::Conflict,
                    "Codex live session already joining",
                ));
            }
            _ => {
                return Err(LiveError::new(
                    LiveErrorKind::NotFound,
                    "Codex live session not found",
                ));
            }
        };
        let url = build_sideband_url(&self.base_url, style, call_id);
        let mut request = SidebandConnectRequest {
            http_url: websocket_http_url(&url),
            url,
            auth_id: session.auth_id.clone(),
            model: session.model.clone(),
            call_id: call_id.clone(),
            subprotocols,
        };
        let first = self.connector.connect(&request);
        let handshake = match first {
            Ok(handshake) => handshake,
            Err(error) if error.kind == LiveErrorKind::Unauthorized && self.refresher.is_some() => {
                let refresher = self.refresher.as_ref().expect("guarded above");
                refresher.report_unauthorized(&session.auth_id, &session.model);
                let refreshed = match refresher.refresh_after_unauthorized(
                    &session.auth_id,
                    &session.model,
                    None,
                ) {
                    Ok(auth) => auth,
                    Err(error) => {
                        self.sessions.release(&session, now_millis);
                        return Err(error);
                    }
                };
                request.auth_id = refreshed.id;
                match self.connector.connect(&request) {
                    Ok(handshake) => handshake,
                    Err(error) => {
                        if error.kind == LiveErrorKind::Unauthorized {
                            refresher.report_unauthorized(&request.auth_id, &session.model);
                        }
                        self.sessions.release(&session, now_millis);
                        return Err(error);
                    }
                }
            }
            Err(error) => {
                self.sessions.release(&session, now_millis);
                return Err(error);
            }
        };
        Ok(SidebandConnection {
            sessions: Arc::clone(&self.sessions),
            session: Some(session),
            endpoint: handshake.endpoint,
            subprotocol: handshake.subprotocol,
            release_at_millis: now_millis,
        })
    }
}

pub struct SidebandConnection {
    sessions: Arc<SessionStore>,
    session: Option<LiveSession>,
    pub endpoint: Arc<dyn WebSocketEndpoint>,
    pub subprotocol: Option<String>,
    release_at_millis: u64,
}

impl SidebandConnection {
    pub fn complete(mut self, reason: &str) {
        if let Some(session) = self.session.take() {
            self.sessions.complete(&session, reason);
        }
        self.endpoint.close(WebSocketClose::Normal);
    }

    pub fn set_reconnect_time(&mut self, now_millis: u64) {
        self.release_at_millis = now_millis;
    }
}

impl Drop for SidebandConnection {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            self.sessions.release(&session, self.release_at_millis);
        }
        self.endpoint.close(WebSocketClose::Normal);
    }
}

/// Runs both relay directions concurrently, propagates the first close reason,
/// and waits for the peer direction to drain before returning.
pub fn relay_websockets(
    downstream: Arc<dyn WebSocketEndpoint>,
    upstream: Arc<dyn WebSocketEndpoint>,
) -> Result<(), WebSocketClose> {
    let down_to_up = {
        let downstream = Arc::clone(&downstream);
        let upstream = Arc::clone(&upstream);
        std::thread::spawn(move || copy_websocket(upstream.as_ref(), downstream.as_ref()))
    };
    let up_to_down = {
        let downstream = Arc::clone(&downstream);
        let upstream = Arc::clone(&upstream);
        std::thread::spawn(move || copy_websocket(downstream.as_ref(), upstream.as_ref()))
    };
    let first = down_to_up
        .join()
        .unwrap_or_else(|_| Err(WebSocketClose::Internal("relay closed".to_owned())));
    let close = normalized_close(first.clone().err());
    downstream.close(close.clone());
    upstream.close(close);
    let second = up_to_down
        .join()
        .unwrap_or_else(|_| Err(WebSocketClose::Internal("relay closed".to_owned())));
    first.and(second)
}

/// Copies frames without decoding their payload, preserving text/binary and
/// control-frame boundaries exactly as the upstream Gorilla relay does.
pub fn copy_websocket(
    destination: &dyn WebSocketEndpoint,
    source: &dyn WebSocketEndpoint,
) -> Result<(), WebSocketClose> {
    loop {
        let frame = source.read()?;
        let terminal = frame.message_type == WebSocketMessageType::Close;
        destination.write(frame)?;
        if terminal {
            return Ok(());
        }
    }
}

#[must_use]
pub fn normalized_close(error: Option<WebSocketClose>) -> WebSocketClose {
    match error {
        None | Some(WebSocketClose::Normal) => WebSocketClose::Normal,
        Some(WebSocketClose::Protocol(1005 | 1006 | 1015, _)) => WebSocketClose::Normal,
        Some(error) => error,
    }
}
