// ref: internal/runtime/executor/xai_websockets_executor.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Transport-neutral xAI Responses WebSocket contract.
//!
//! Upstream stores sessions, credentials, proxy selection and the wall clock in
//! package globals.  The Rust port deliberately makes those authorities
//! constructor/request inputs.  A concrete WebSocket implementation only has
//! to implement [`XaiWebsocketTransport`]; protocol state and replay behavior
//! remain independently testable.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};
use url::Url;

use crate::sdk::cliproxy::auth::Auth;
use crate::sdk::cliproxy::executor::{Headers, RequestScopedError, StatusError};

pub const XAI_FREE_USAGE_EXHAUSTED_COOLDOWN: Duration = Duration::from_secs(24 * 60 * 60);
pub const XAI_CLOSE_NORMAL: u16 = 1000;
pub const XAI_CLOSE_MESSAGE_TOO_BIG: u16 = 1009;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XaiStreamTransportDecision {
    Http,
    Websocket,
}

/// Mirrors `xaiWebsocketsEnabled`. Attribute configuration wins over metadata.
#[must_use]
pub fn xai_websockets_enabled(auth: Option<&Auth>) -> bool {
    let Some(auth) = auth else { return false };
    if let Some(value) = auth.attributes.get("websockets") {
        if let Some(parsed) = parse_bool(value) {
            return parsed;
        }
    }
    auth.metadata.get("websockets").is_some_and(|value| {
        value
            .as_bool()
            .or_else(|| value.as_str().and_then(parse_bool))
            .unwrap_or(false)
    })
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "t" | "yes" | "y" | "on" => Some(true),
        "0" | "false" | "f" | "no" | "n" | "off" => Some(false),
        _ => None,
    }
}

pub fn decide_xai_stream_transport(
    auth: Option<&Auth>,
    downstream_is_websocket: bool,
    upstream_websocket_required: bool,
) -> Result<XaiStreamTransportDecision, XaiWebsocketError> {
    if downstream_is_websocket && xai_websockets_enabled(auth) {
        return Ok(XaiStreamTransportDecision::Websocket);
    }
    if upstream_websocket_required {
        return Err(XaiWebsocketError::replay_required());
    }
    Ok(XaiStreamTransportDecision::Http)
}

/// WebSocket payload shape required by xAI's Responses endpoint.
#[must_use]
pub fn build_xai_websocket_request_body(body: &[u8]) -> Vec<u8> {
    if body.is_empty() {
        return Vec::new();
    }
    let Ok(mut value) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    let Some(object) = value.as_object_mut() else {
        return body.to_vec();
    };
    object.insert("type".into(), Value::String("response.create".into()));
    object.remove("stream");
    object.remove("stream_options");
    object.remove("background");
    object.insert("store".into(), Value::Bool(true));
    if object
        .get("previous_response_id")
        .and_then(Value::as_str)
        .is_some_and(|id| !id.trim().is_empty())
    {
        object.remove("instructions");
    }
    serde_json::to_vec(&value).unwrap_or_else(|_| body.to_vec())
}

pub fn build_xai_responses_websocket_url(http_url: &str) -> Result<String, XaiWebsocketError> {
    let mut url = Url::parse(http_url.trim())
        .map_err(|error| XaiWebsocketError::protocol(format!("invalid responses URL: {error}")))?;
    let scheme = match url.scheme().to_ascii_lowercase().as_str() {
        "http" => "ws",
        "https" => "wss",
        "ws" => "ws",
        "wss" => "wss",
        other => {
            return Err(XaiWebsocketError::protocol(format!(
                "unsupported responses websocket URL scheme {other:?}"
            )))
        }
    };
    url.set_scheme(scheme)
        .map_err(|()| XaiWebsocketError::protocol("could not set websocket URL scheme"))?;
    if url.host_str().is_none_or(|host| host.trim().is_empty()) {
        return Err(XaiWebsocketError::protocol(
            "responses websocket URL host is empty",
        ));
    }
    Ok(url.into())
}

/// Secret-bearing connection material. Debug output never renders the token.
#[derive(Clone, Eq, PartialEq)]
pub struct XaiWebsocketCredential {
    pub auth_id: String,
    pub bearer_token: String,
    pub proxy_url: Option<String>,
}

impl fmt::Debug for XaiWebsocketCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiWebsocketCredential")
            .field("auth_id", &self.auth_id)
            .field("has_bearer_token", &!self.bearer_token.is_empty())
            .field("has_proxy_url", &self.proxy_url.is_some())
            .finish()
    }
}

#[must_use]
pub fn apply_xai_websocket_headers(mut headers: Headers, token: &str, session_id: &str) -> Headers {
    headers.insert("content-type".into(), vec!["application/json".into()]);
    if !token.trim().is_empty() {
        headers.insert(
            "authorization".into(),
            vec![format!("Bearer {}", token.trim())],
        );
    }
    if !session_id.trim().is_empty() {
        headers.insert("x-grok-conv-id".into(), vec![session_id.trim().to_owned()]);
    }
    headers
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct XaiWebsocketConnectRequest {
    pub url: String,
    pub headers: Headers,
    pub credential: XaiWebsocketCredential,
    pub session_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum XaiWebsocketFrame {
    Text(Vec<u8>),
    Binary(Vec<u8>),
    Ping(Vec<u8>),
    Pong(Vec<u8>),
    Close { code: u16, reason: String },
}

pub trait XaiWebsocketConnection: Send {
    fn write(&mut self, frame: XaiWebsocketFrame) -> Result<(), XaiWebsocketError>;
    fn read(&mut self) -> Result<XaiWebsocketFrame, XaiWebsocketError>;
    fn close(&mut self, reason: &str) -> Result<(), XaiWebsocketError>;
}

/// Injected network authority. It may implement TCP/TLS, HTTP CONNECT or
/// SOCKS, but protocol code never consults process environment or global state.
pub trait XaiWebsocketTransport: Send + Sync {
    fn connect(
        &self,
        request: XaiWebsocketConnectRequest,
    ) -> Result<Box<dyn XaiWebsocketConnection>, XaiWebsocketError>;
}

pub trait XaiWebsocketClock: Send + Sync {
    fn now_millis(&self) -> i64;
}

#[derive(Debug, Default)]
pub struct SystemXaiWebsocketClock;

impl XaiWebsocketClock for SystemXaiWebsocketClock {
    fn now_millis(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(i64::MAX)
    }
}

#[derive(Default)]
struct XaiIdStateInner {
    downstream_to_upstream: BTreeMap<String, String>,
    sequence: u64,
    transcript_input: Vec<Value>,
    replay_compacted_transcript_on_reset: bool,
}

#[derive(Default)]
pub struct XaiWebsocketIdState {
    inner: Mutex<XaiIdStateInner>,
}

impl fmt::Debug for XaiWebsocketIdState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        formatter
            .debug_struct("XaiWebsocketIdState")
            .field("mapped_ids", &inner.downstream_to_upstream.len())
            .field("sequence", &inner.sequence)
            .field("transcript_items", &inner.transcript_input.len())
            .field(
                "replay_compacted_transcript_on_reset",
                &inner.replay_compacted_transcript_on_reset,
            )
            .finish()
    }
}

impl XaiWebsocketIdState {
    #[must_use]
    pub fn upstream_id_for_downstream(&self, downstream_id: &str) -> String {
        let id = downstream_id.trim();
        if id.is_empty() {
            return String::new();
        }
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .downstream_to_upstream
            .get(id)
            .map(|value| value.trim().to_owned())
            .unwrap_or_else(|| id.to_owned())
    }

    pub fn map_downstream_to_upstream(&self, downstream_id: &str, upstream_id: &str) {
        let downstream_id = downstream_id.trim();
        if downstream_id.is_empty() {
            return;
        }
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .downstream_to_upstream
            .insert(downstream_id.to_owned(), upstream_id.trim().to_owned());
    }

    #[must_use]
    pub fn snapshot_transcript_input(&self) -> Vec<u8> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if inner.transcript_input.is_empty() {
            Vec::new()
        } else {
            serde_json::to_vec(&inner.transcript_input).unwrap_or_default()
        }
    }

    #[must_use]
    pub fn prepend_transcript_input(&self, payload: &[u8]) -> Vec<u8> {
        self.prepend_transcript(payload, false).0
    }

    fn prepend_transcript(&self, payload: &[u8], compacted_only: bool) -> (Vec<u8>, bool) {
        let Ok(mut value) = serde_json::from_slice::<Value>(payload) else {
            return (payload.to_vec(), false);
        };
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if inner.transcript_input.is_empty()
            || (compacted_only && !inner.replay_compacted_transcript_on_reset)
        {
            return (payload.to_vec(), false);
        }
        let mut merged = inner.transcript_input.clone();
        if let Some(input) = value.get("input").and_then(Value::as_array) {
            merged.extend(input.iter().cloned());
        }
        drop(inner);
        value["input"] = Value::Array(merged);
        (
            serde_json::to_vec(&value).unwrap_or_else(|_| payload.to_vec()),
            true,
        )
    }

    pub fn record_transcript_turn(
        &self,
        request_payload: &[u8],
        completed_payload: &[u8],
        reset: bool,
    ) {
        let request = serde_json::from_slice::<Value>(request_payload).unwrap_or(Value::Null);
        let completed = serde_json::from_slice::<Value>(completed_payload).unwrap_or(Value::Null);
        let input = request
            .get("input")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let output = completed
            .pointer("/response/output")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if reset {
            inner.transcript_input.clear();
            inner.replay_compacted_transcript_on_reset = false;
        }
        inner.transcript_input.extend(input);
        inner.transcript_input.extend(output);
    }

    pub fn replace_transcript_with_items(&self, items: impl IntoIterator<Item = Value>) {
        let next: Vec<Value> = items.into_iter().collect();
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.replay_compacted_transcript_on_reset = !next.is_empty();
        inner.transcript_input = next;
    }

    #[must_use]
    pub fn prepend_compacted_transcript_on_reset(&self, payload: &[u8]) -> (Vec<u8>, bool) {
        self.prepend_transcript(payload, true)
    }
}

#[derive(Clone, Debug)]
struct SessionEntry {
    state: Arc<XaiWebsocketIdState>,
    auth_id: String,
    url: String,
    touched_at_millis: i64,
}

/// Explicitly owned session/ID store; there is intentionally no process global.
#[derive(Default, Debug)]
pub struct XaiWebsocketSessionStore {
    sessions: Mutex<BTreeMap<String, SessionEntry>>,
}

impl XaiWebsocketSessionStore {
    #[must_use]
    pub fn state(&self, session_id: &str, now_millis: i64) -> Option<Arc<XaiWebsocketIdState>> {
        let session_id = session_id.trim();
        if session_id.is_empty() {
            return None;
        }
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = sessions
            .entry(session_id.to_owned())
            .or_insert_with(|| SessionEntry {
                state: Arc::new(XaiWebsocketIdState::default()),
                auth_id: String::new(),
                url: String::new(),
                touched_at_millis: now_millis,
            });
        entry.touched_at_millis = now_millis;
        Some(Arc::clone(&entry.state))
    }

    /// Returns true when auth or URL changed, which requires a fresh connection.
    pub fn update_target(
        &self,
        session_id: &str,
        auth_id: &str,
        url: &str,
        now_millis: i64,
    ) -> bool {
        let Some(_) = self.state(session_id, now_millis) else {
            return false;
        };
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(entry) = sessions.get_mut(session_id.trim()) else {
            return false;
        };
        let changed = (!entry.auth_id.is_empty() || !entry.url.is_empty())
            && (entry.auth_id.trim() != auth_id.trim() || entry.url.trim() != url.trim());
        entry.auth_id = auth_id.trim().to_owned();
        entry.url = url.trim().to_owned();
        entry.touched_at_millis = now_millis;
        changed
    }

    pub fn close_session(&self, session_id: &str) -> bool {
        self.sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(session_id.trim())
            .is_some()
    }

    pub fn close_all(&self) -> usize {
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let count = sessions.len();
        sessions.clear();
        count
    }

    pub fn expire_idle(&self, now_millis: i64, max_idle: Duration) -> usize {
        let max_idle = i64::try_from(max_idle.as_millis()).unwrap_or(i64::MAX);
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let before = sessions.len();
        sessions.retain(|_, entry| now_millis.saturating_sub(entry.touched_at_millis) <= max_idle);
        before - sessions.len()
    }
}

pub struct XaiWebsocketRequestIdMapper {
    state: Arc<XaiWebsocketIdState>,
    downstream_previous_id: String,
    upstream_previous_id: String,
    upstream_response_id: String,
    downstream_response_id: String,
    pub replayed_compacted_transcript: bool,
}

impl XaiWebsocketRequestIdMapper {
    #[must_use]
    pub fn new(state: Arc<XaiWebsocketIdState>, downstream_request: &[u8]) -> Self {
        let downstream_previous_id = json_string(downstream_request, "/previous_response_id");
        let upstream_previous_id = state.upstream_id_for_downstream(&downstream_previous_id);
        Self {
            state,
            downstream_previous_id,
            upstream_previous_id,
            upstream_response_id: String::new(),
            downstream_response_id: String::new(),
            replayed_compacted_transcript: false,
        }
    }

    #[must_use]
    pub fn upstream_request_payload(&mut self, payload: &[u8]) -> Vec<u8> {
        let request_type = json_string(payload, "/type");
        if self.downstream_previous_id == self.upstream_previous_id {
            if self.downstream_previous_id.is_empty() && request_type == "response.append" {
                let (out, replayed) = self.state.prepend_compacted_transcript_on_reset(payload);
                self.replayed_compacted_transcript = replayed;
                return out;
            }
            return payload.to_vec();
        }
        let Ok(mut value) = serde_json::from_slice::<Value>(payload) else {
            return payload.to_vec();
        };
        if self.upstream_previous_id.is_empty() {
            value
                .as_object_mut()
                .map(|object| object.remove("previous_response_id"));
            let out = serde_json::to_vec(&value).unwrap_or_else(|_| payload.to_vec());
            if !self.downstream_previous_id.is_empty() {
                self.replayed_compacted_transcript = true;
                return self.state.prepend_transcript_input(&out);
            }
            return out;
        }
        value["previous_response_id"] = Value::String(self.upstream_previous_id.clone());
        serde_json::to_vec(&value).unwrap_or_else(|_| payload.to_vec())
    }

    #[must_use]
    pub fn downstream_response_payload(&mut self, payload: &[u8]) -> Vec<u8> {
        let upstream_id = json_string(payload, "/response/id");
        let downstream_id = self.downstream_id_for_upstream_response(&upstream_id);
        if downstream_id.is_empty() {
            return payload.to_vec();
        }
        rewrite_xai_websocket_downstream_ids(
            payload,
            &self.upstream_response_id,
            &downstream_id,
            &self.upstream_previous_id,
            &self.downstream_previous_id,
        )
    }

    fn downstream_id_for_upstream_response(&mut self, upstream_response_id: &str) -> String {
        let upstream_response_id = upstream_response_id.trim();
        if !self.upstream_response_id.is_empty() {
            return self.downstream_response_id.clone();
        }
        if upstream_response_id.is_empty() {
            return String::new();
        }
        let mut inner = self
            .state
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.upstream_response_id = upstream_response_id.to_owned();
        self.downstream_response_id = upstream_response_id.to_owned();
        let seen = inner
            .downstream_to_upstream
            .contains_key(upstream_response_id);
        if (!self.downstream_previous_id.is_empty()
            && !self.upstream_previous_id.is_empty()
            && upstream_response_id == self.upstream_previous_id)
            || seen
        {
            inner.sequence += 1;
            self.downstream_response_id = format!("{upstream_response_id}-xai-{}", inner.sequence);
        }
        inner.downstream_to_upstream.insert(
            upstream_response_id.to_owned(),
            upstream_response_id.to_owned(),
        );
        inner.downstream_to_upstream.insert(
            self.downstream_response_id.clone(),
            upstream_response_id.to_owned(),
        );
        self.downstream_response_id.clone()
    }
}

#[must_use]
pub fn rewrite_xai_websocket_downstream_ids(
    payload: &[u8],
    upstream_response_id: &str,
    downstream_response_id: &str,
    upstream_previous_id: &str,
    downstream_previous_id: &str,
) -> Vec<u8> {
    if payload.is_empty()
        || (upstream_response_id.trim() == downstream_response_id.trim()
            && upstream_previous_id.trim() == downstream_previous_id.trim())
    {
        return payload.to_vec();
    }
    let Ok(mut value) = serde_json::from_slice::<Value>(payload) else {
        return payload.to_vec();
    };
    let changed = rewrite_id_value(
        &mut value,
        upstream_response_id.trim(),
        downstream_response_id.trim(),
        upstream_previous_id.trim(),
        downstream_previous_id.trim(),
    );
    if !changed {
        return payload.to_vec();
    }
    serde_json::to_vec(&value).unwrap_or_else(|_| payload.to_vec())
}

fn rewrite_id_value(
    value: &mut Value,
    upstream_id: &str,
    downstream_id: &str,
    upstream_previous: &str,
    downstream_previous: &str,
) -> bool {
    match value {
        Value::Object(object) => object.iter_mut().fold(false, |changed, (key, child)| {
            let child_changed = if let Some(text) = child.as_str() {
                let replacement = match key.as_str() {
                    "id" | "item_id" if !upstream_id.is_empty() && text.contains(upstream_id) => {
                        text.replace(upstream_id, downstream_id)
                    }
                    "previous_response_id"
                        if !upstream_previous.is_empty() && text == upstream_previous =>
                    {
                        downstream_previous.to_owned()
                    }
                    _ => text.to_owned(),
                };
                if replacement != text {
                    *child = Value::String(replacement);
                    true
                } else {
                    false
                }
            } else {
                rewrite_id_value(
                    child,
                    upstream_id,
                    downstream_id,
                    upstream_previous,
                    downstream_previous,
                )
            };
            changed || child_changed
        }),
        Value::Array(array) => array.iter_mut().fold(false, |changed, child| {
            rewrite_id_value(
                child,
                upstream_id,
                downstream_id,
                upstream_previous,
                downstream_previous,
            ) || changed
        }),
        _ => false,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct XaiCompactionState {
    pub response_id: String,
    pub item: Value,
}

pub fn validate_xai_websocket_compaction_response(
    data: &[u8],
) -> Result<XaiCompactionState, XaiWebsocketError> {
    let value: Value = serde_json::from_slice(data).map_err(|_| {
        XaiWebsocketError::bad_gateway("xai websocket compaction returned invalid JSON")
    })?;
    let response_id = value
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(compaction_missing)?;
    let mut item = value
        .get("output")
        .and_then(Value::as_array)
        .and_then(|output| output.first())
        .cloned()
        .ok_or_else(compaction_missing)?;
    let valid = item.get("type").and_then(Value::as_str) == Some("compaction")
        && item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .is_some_and(|content| !content.trim().is_empty());
    if !valid {
        return Err(compaction_missing());
    }
    let response_id = if response_id.starts_with("resp_") {
        response_id.to_owned()
    } else {
        format!("resp_{}", response_id.trim_start_matches("cmp_"))
    };
    if item.get("id").is_none() {
        item["id"] = Value::String(format!(
            "cmp_{}",
            response_id.strip_prefix("resp_").unwrap_or(&response_id)
        ));
    }
    Ok(XaiCompactionState { response_id, item })
}

fn compaction_missing() -> XaiWebsocketError {
    XaiWebsocketError::bad_gateway("xai websocket compaction response is missing compacted state")
}

pub fn build_xai_websocket_compaction_payload(
    payload: &[u8],
    transcript_input: &[u8],
) -> Result<Vec<u8>, XaiWebsocketError> {
    let mut value: Value = if payload.is_empty() {
        json!({})
    } else {
        serde_json::from_slice(payload).map_err(|error| {
            XaiWebsocketError::protocol(format!("invalid request JSON: {error}"))
        })?
    };
    let input: Value = if transcript_input.is_empty() {
        json!([])
    } else {
        serde_json::from_slice(transcript_input).map_err(|error| {
            XaiWebsocketError::protocol(format!("invalid transcript JSON: {error}"))
        })?
    };
    value["input"] = input;
    value
        .as_object_mut()
        .map(|object| object.remove("previous_response_id"));
    serde_json::to_vec(&value).map_err(|error| XaiWebsocketError::protocol(error.to_string()))
}

#[must_use]
pub fn xai_websocket_generate_false(payload: &[u8]) -> bool {
    serde_json::from_slice::<Value>(payload)
        .ok()
        .and_then(|value| value.get("generate").and_then(Value::as_bool))
        == Some(false)
}

#[must_use]
pub fn build_xai_websocket_warmup_completed_payload(created_payload: &[u8]) -> Vec<u8> {
    let created = serde_json::from_slice::<Value>(created_payload).unwrap_or(Value::Null);
    let mut response = created
        .get("response")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    response.insert("status".into(), Value::String("completed".into()));
    response.entry("output").or_insert_with(|| json!([]));
    response
        .entry("usage")
        .or_insert_with(|| json!({"input_tokens":0,"output_tokens":0,"total_tokens":0}));
    let mut completed = Map::new();
    completed.insert("type".into(), Value::String("response.completed".into()));
    if let Some(sequence) = created.get("sequence_number").and_then(Value::as_i64) {
        completed.insert("sequence_number".into(), Value::from(sequence + 1));
    }
    completed.insert("response".into(), Value::Object(response));
    serde_json::to_vec(&Value::Object(completed)).unwrap_or_default()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct XaiWebsocketError {
    pub status: u16,
    pub code: String,
    pub message: String,
    pub payload: Vec<u8>,
    pub headers: Headers,
    pub retry_after: Option<Duration>,
    pub request_scoped: bool,
    pub retryable: bool,
}

impl XaiWebsocketError {
    fn protocol(message: impl Into<String>) -> Self {
        Self::new(400, "websocket_protocol_error", message, false, false)
    }

    fn bad_gateway(message: impl Into<String>) -> Self {
        Self::new(502, "invalid_upstream_response", message, false, false)
    }

    fn replay_required() -> Self {
        Self::new(
            426,
            "upstream_http_replay_required",
            "upstream WebSocket replay is required",
            true,
            false,
        )
    }

    pub fn transport(message: impl Into<String>, retryable: bool) -> Self {
        Self::new(502, "websocket_transport_error", message, false, retryable)
    }

    fn new(
        status: u16,
        code: impl Into<String>,
        message: impl Into<String>,
        request_scoped: bool,
        retryable: bool,
    ) -> Self {
        let code = code.into();
        let message = message.into();
        let payload = serde_json::to_vec(&json!({
            "status": status,
            "error": {"code": code, "message": message}
        }))
        .unwrap_or_default();
        Self {
            status,
            code,
            message,
            payload,
            headers: Headers::new(),
            retry_after: None,
            request_scoped,
            retryable,
        }
    }

    #[must_use]
    pub fn should_retry_send(&self) -> bool {
        self.retryable && !self.request_scoped
    }
}

impl fmt::Display for XaiWebsocketError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&String::from_utf8_lossy(&self.payload))
    }
}

impl Error for XaiWebsocketError {}

impl StatusError for XaiWebsocketError {
    fn status_code(&self) -> u16 {
        self.status
    }
}

impl RequestScopedError for XaiWebsocketError {
    fn is_request_scoped(&self) -> bool {
        self.request_scoped
    }
}

#[must_use]
pub fn map_xai_websocket_close(code: u16, reason: &str) -> XaiWebsocketError {
    if code == XAI_CLOSE_MESSAGE_TOO_BIG {
        return XaiWebsocketError::new(
            413,
            "message_too_big",
            "upstream websocket message too big",
            true,
            false,
        );
    }
    XaiWebsocketError::new(
        502,
        "websocket_closed",
        format!("upstream websocket closed ({code}): {}", reason.trim()),
        false,
        code == XAI_CLOSE_NORMAL || code == 1001 || code == 1006,
    )
}

/// Preserves the write failure unless the connection concurrently reported a
/// request-scoped close (notably RFC 6455 code 1009). This is the typed form of
/// upstream's `mapXAIWebsocketWriteError`.
#[must_use]
pub fn map_xai_websocket_write_error(
    write_error: XaiWebsocketError,
    upstream_close: Option<(u16, &str)>,
) -> XaiWebsocketError {
    match upstream_close {
        Some((XAI_CLOSE_MESSAGE_TOO_BIG, reason)) => {
            map_xai_websocket_close(XAI_CLOSE_MESSAGE_TOO_BIG, reason)
        }
        _ => write_error,
    }
}

#[must_use]
pub fn should_retry_xai_websocket_send(error: &XaiWebsocketError) -> bool {
    error.should_retry_send()
}

#[must_use]
pub fn normalize_xai_websocket_completion(payload: &[u8]) -> Vec<u8> {
    let Ok(mut value) = serde_json::from_slice::<Value>(payload) else {
        return payload.to_vec();
    };
    if value.get("type").and_then(Value::as_str) != Some("response.done") {
        return payload.to_vec();
    }
    value["type"] = Value::String("response.completed".into());
    serde_json::to_vec(&value).unwrap_or_else(|_| payload.to_vec())
}

pub fn parse_xai_websocket_error(payload: &[u8]) -> Option<XaiWebsocketError> {
    let value: Value = serde_json::from_slice(payload).ok()?;
    let typed_error = value.get("type").and_then(Value::as_str) == Some("error");
    let supplied_error = value.pointer("/body/error").or_else(|| value.get("error"));
    if !typed_error && supplied_error.is_none() {
        return None;
    }
    let error_node = supplied_error
        .cloned()
        .unwrap_or_else(|| json!({"type":"server_error", "message":"xAI websocket error"}));
    let status = value
        .get("status")
        .or_else(|| value.get("status_code"))
        .and_then(value_as_status)
        .or_else(|| {
            error_node
                .get("code")
                .or_else(|| error_node.get("status"))
                .and_then(value_as_status)
        })
        .unwrap_or_else(|| {
            let message = error_node
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if message.contains("Request validation error") || message.contains("\"code\":\"400\"")
            {
                400
            } else {
                500
            }
        });
    let code = error_node
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| value.get("code").and_then(Value::as_str))
        .unwrap_or("upstream_error")
        .to_owned();
    let message = error_node
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error_node.as_str())
        .unwrap_or("xAI websocket error")
        .to_owned();
    let normalized = json!({"type":"error", "status":status, "error":error_node});
    let mut error = XaiWebsocketError {
        status,
        code,
        message,
        payload: serde_json::to_vec(&normalized).unwrap_or_else(|_| payload.to_vec()),
        headers: parse_error_headers(value.get("headers")),
        retry_after: None,
        request_scoped: false,
        retryable: status >= 500 || status == 429,
    };
    let haystack = String::from_utf8_lossy(&error.payload).to_ascii_lowercase();
    if status == 429
        && (haystack.contains("free-usage-exhausted") || haystack.contains("included free usage"))
    {
        error.retry_after = Some(XAI_FREE_USAGE_EXHAUSTED_COOLDOWN);
    }
    Some(error)
}

fn value_as_status(value: &Value) -> Option<u16> {
    value
        .as_u64()
        .and_then(|value| value.try_into().ok())
        .or_else(|| value.as_str()?.trim().parse().ok())
        .filter(|status| *status > 0)
}

fn parse_error_headers(value: Option<&Value>) -> Headers {
    value
        .and_then(Value::as_object)
        .map(|object| {
            object
                .iter()
                .filter_map(|(name, value)| {
                    let rendered = value
                        .as_str()
                        .map(str::to_owned)
                        .or_else(|| value.is_boolean().then(|| value.to_string()))
                        .or_else(|| value.is_number().then(|| value.to_string()))?;
                    (!name.trim().is_empty() && !rendered.trim().is_empty())
                        .then(|| (name.clone(), vec![rendered]))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn json_string(payload: &[u8], pointer: &str) -> String {
    serde_json::from_slice::<Value>(payload)
        .ok()
        .and_then(|value| {
            value
                .pointer(pointer)
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_default()
        .trim()
        .to_owned()
}

#[derive(Clone, Debug)]
pub struct XaiWebsocketExecutionRequest {
    pub session_id: String,
    pub url: String,
    pub headers: Headers,
    pub credential: XaiWebsocketCredential,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct XaiWebsocketExecutionResult {
    pub events: Vec<Vec<u8>>,
    pub reconnects: usize,
    pub target_changed: bool,
}

pub struct XaiWebsocketsExecutor {
    transport: Arc<dyn XaiWebsocketTransport>,
    clock: Arc<dyn XaiWebsocketClock>,
    sessions: Arc<XaiWebsocketSessionStore>,
    max_reconnects: usize,
}

impl fmt::Debug for XaiWebsocketsExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiWebsocketsExecutor")
            .field("max_reconnects", &self.max_reconnects)
            .finish_non_exhaustive()
    }
}

impl XaiWebsocketsExecutor {
    #[must_use]
    pub fn new(
        transport: Arc<dyn XaiWebsocketTransport>,
        clock: Arc<dyn XaiWebsocketClock>,
        sessions: Arc<XaiWebsocketSessionStore>,
    ) -> Self {
        Self {
            transport,
            clock,
            sessions,
            max_reconnects: 1,
        }
    }

    #[must_use]
    pub fn with_max_reconnects(mut self, max_reconnects: usize) -> Self {
        self.max_reconnects = max_reconnects;
        self
    }

    pub fn close_execution_session(&self, session_id: &str) -> bool {
        self.sessions.close_session(session_id)
    }

    /// Runs one bounded response turn. Connection ownership never escapes the
    /// call; a concrete async harness can invoke this on its blocking transport
    /// worker and forward `events` into its downstream stream.
    pub fn execute_stream(
        &self,
        request: XaiWebsocketExecutionRequest,
    ) -> Result<XaiWebsocketExecutionResult, XaiWebsocketError> {
        let now = self.clock.now_millis();
        let state = self
            .sessions
            .state(&request.session_id, now)
            .unwrap_or_else(|| Arc::new(XaiWebsocketIdState::default()));
        let target_changed = self.sessions.update_target(
            &request.session_id,
            &request.credential.auth_id,
            &request.url,
            now,
        );
        let prepared = build_xai_websocket_request_body(&request.payload);
        let mut mapper = XaiWebsocketRequestIdMapper::new(state.clone(), &prepared);
        let outbound = mapper.upstream_request_payload(&prepared);
        let generate_false = xai_websocket_generate_false(&outbound);
        let mut reconnects = 0;

        loop {
            let connect = XaiWebsocketConnectRequest {
                url: request.url.clone(),
                headers: request.headers.clone(),
                credential: request.credential.clone(),
                session_id: request.session_id.clone(),
            };
            let mut connection = self.transport.connect(connect)?;
            if let Err(error) = connection.write(XaiWebsocketFrame::Text(outbound.clone())) {
                let _ = connection.close("write_failed");
                if error.should_retry_send() && reconnects < self.max_reconnects {
                    reconnects += 1;
                    continue;
                }
                return Err(error);
            }

            let mut events = Vec::new();
            let terminal = loop {
                match connection.read() {
                    Ok(XaiWebsocketFrame::Text(payload)) => {
                        if let Some(error) = parse_xai_websocket_error(&payload) {
                            break Err(error);
                        }
                        let event_type = json_string(&payload, "/type");
                        let downstream = mapper.downstream_response_payload(&payload);
                        events.push(downstream.clone());
                        if generate_false && event_type == "response.created" {
                            let completed =
                                build_xai_websocket_warmup_completed_payload(&downstream);
                            state.record_transcript_turn(
                                &outbound,
                                &completed,
                                mapper.replayed_compacted_transcript,
                            );
                            events.push(completed);
                            break Ok(());
                        }
                        if event_type == "response.completed" || event_type == "response.done" {
                            state.record_transcript_turn(
                                &outbound,
                                &downstream,
                                mapper.replayed_compacted_transcript,
                            );
                            break Ok(());
                        }
                    }
                    Ok(XaiWebsocketFrame::Ping(payload)) => {
                        connection.write(XaiWebsocketFrame::Pong(payload))?;
                    }
                    Ok(XaiWebsocketFrame::Pong(_)) => {}
                    Ok(XaiWebsocketFrame::Binary(_)) => {
                        break Err(XaiWebsocketError::protocol(
                            "unexpected binary websocket message",
                        ));
                    }
                    Ok(XaiWebsocketFrame::Close { code, reason }) => {
                        break Err(map_xai_websocket_close(code, &reason));
                    }
                    Err(error) => break Err(error),
                }
            };
            let _ = connection.close(if terminal.is_ok() {
                "turn_complete"
            } else {
                "read_failed"
            });
            match terminal {
                Ok(()) => {
                    return Ok(XaiWebsocketExecutionResult {
                        events,
                        reconnects,
                        target_changed,
                    })
                }
                Err(error)
                    if error.should_retry_send()
                        && events.is_empty()
                        && reconnects < self.max_reconnects =>
                {
                    reconnects += 1;
                }
                Err(error) => return Err(error),
            }
        }
    }
}
