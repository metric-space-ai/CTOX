// ref: internal/client/codex/live/live.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use mime::Mime;
use serde_json::{Map, Value};

use crate::sdk::cliproxy::auth::Auth;
use crate::sdk::cliproxy::executor::Headers;

use super::{call_id_from_location, LiveSession, MediaRelay, MediaRoute, SessionStore};

pub const UPSTREAM_CALL_URL: &str =
    "https://chatgpt.com/backend-api/codex/realtime/calls?intent=quicksilver&architecture=avas";
pub const DEFAULT_LIVE_MODEL: &str = "gpt-live-1-codex";
pub const MAX_BODY_SIZE: usize = 16 << 20;
pub const LIVE_PROTOCOL_HEADERS: [&str; 6] = [
    "OpenAI-Alpha",
    "X-Session-Id",
    "Session-Id",
    "Thread-Id",
    "Originator",
    "X-Oai-Attestation",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LiveErrorKind {
    BadRequest,
    BodyTooLarge,
    Unauthorized,
    Conflict,
    NotFound,
    Unavailable,
    Upstream,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveError {
    pub kind: LiveErrorKind,
    pub message: String,
}

impl LiveError {
    pub fn new(kind: LiveErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    #[must_use]
    pub const fn status_code(&self) -> u16 {
        match self.kind {
            LiveErrorKind::BadRequest => 400,
            LiveErrorKind::Unauthorized => 401,
            LiveErrorKind::NotFound => 404,
            LiveErrorKind::Conflict => 409,
            LiveErrorKind::BodyTooLarge => 413,
            LiveErrorKind::Unavailable => 503,
            LiveErrorKind::Upstream => 502,
        }
    }
}

impl fmt::Display for LiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for LiveError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedCallRequest {
    pub body: Vec<u8>,
    pub content_type: String,
    pub model: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveHttpRequest {
    pub url: String,
    pub headers: Headers,
    pub body: Vec<u8>,
    pub model: String,
    pub auth_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveHttpResponse {
    pub status: u16,
    pub headers: Headers,
    pub body: Vec<u8>,
}

pub type LiveTransportFuture<'a> =
    Pin<Box<dyn Future<Output = Result<LiveHttpResponse, LiveError>> + Send + 'a>>;

/// Request-scoped transport authority. Implementations may use HTTP, a test
/// fixture, or a host gateway; the Codex module never installs a global client.
pub trait LiveTransport: Send + Sync {
    fn execute<'a>(&'a self, request: LiveHttpRequest) -> LiveTransportFuture<'a>;
}

/// Host-owned Home selection refresh. The live module receives a new owned
/// snapshot and never reaches into a global credential manager.
pub trait LiveCredentialRefresher: Send + Sync {
    fn report_unauthorized(&self, auth_id: &str, model: &str);

    fn refresh_after_unauthorized(
        &self,
        auth_id: &str,
        model: &str,
        current: Option<&Auth>,
    ) -> Result<Auth, LiveError>;
}

pub struct LiveClient {
    transport: Arc<dyn LiveTransport>,
    media: Option<Arc<dyn MediaRelay>>,
    sessions: Arc<SessionStore>,
    refresher: Option<Arc<dyn LiveCredentialRefresher>>,
    default_proxy_url: String,
}

impl LiveClient {
    pub fn new(
        transport: Arc<dyn LiveTransport>,
        media: Option<Arc<dyn MediaRelay>>,
        sessions: Arc<SessionStore>,
        default_proxy_url: impl Into<String>,
    ) -> Self {
        Self {
            transport,
            media,
            sessions,
            refresher: None,
            default_proxy_url: default_proxy_url.into().trim().to_owned(),
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

    /// Executes the complete bootstrap transaction. Auth choice is supplied by
    /// the caller and remains pinned in the returned sideband session.
    pub async fn bootstrap(
        &self,
        body: &[u8],
        content_type: &str,
        source_headers: &Headers,
        selected: &Auth,
        auth_index: &str,
        now_millis: u64,
    ) -> Result<LiveHttpResponse, LiveError> {
        let mut prepared = prepare_call_request(body, content_type)?;
        let mut media_session = None;
        if let Some(media) = &self.media {
            let offer = call_request_sdp(&prepared.body, &prepared.content_type)?;
            let route = MediaRoute {
                proxy_url: proxy_url_for_auth(&self.default_proxy_url, selected),
                credential: media_credential_name(Some(selected), auth_index),
                auth_index: auth_index.trim().to_owned(),
            };
            let (session, upstream_offer) = media.new_session(&offer, &route).await?;
            prepared.body =
                replace_call_request_sdp(&prepared.body, &prepared.content_type, &upstream_offer)?
                    .0;
            prepared.content_type = "application/json".to_owned();
            media_session = Some(session);
        }

        let request_for = |auth: &Auth| {
            let mut headers = protocol_headers(source_headers);
            header_set(&mut headers, "Content-Type", &prepared.content_type);
            set_account_header(&mut headers, auth);
            LiveHttpRequest {
                url: UPSTREAM_CALL_URL.to_owned(),
                headers,
                body: prepared.body.clone(),
                model: prepared.model.clone(),
                auth_id: auth.id.clone(),
            }
        };
        let mut effective_auth = selected.clone();
        let mut response = match self.transport.execute(request_for(&effective_auth)).await {
            Ok(response) => response,
            Err(error) => {
                if let Some(session) = media_session {
                    session.close("request_failed");
                }
                return Err(error);
            }
        };
        if response.status == 401 {
            if let Some(refresher) = &self.refresher {
                refresher.report_unauthorized(&effective_auth.id, &prepared.model);
                effective_auth = match refresher.refresh_after_unauthorized(
                    &effective_auth.id,
                    &prepared.model,
                    Some(&effective_auth),
                ) {
                    Ok(auth) => auth,
                    Err(error) => {
                        if let Some(session) = media_session.as_ref() {
                            session.close("refresh_failed");
                        }
                        return Err(error);
                    }
                };
                response = match self.transport.execute(request_for(&effective_auth)).await {
                    Ok(response) => response,
                    Err(error) => {
                        if let Some(session) = media_session.as_ref() {
                            session.close("retry_failed");
                        }
                        return Err(error);
                    }
                };
                if response.status == 401 {
                    refresher.report_unauthorized(&effective_auth.id, &prepared.model);
                }
            }
        }
        response.headers = call_response_headers(&response.headers);
        if response.body.len() > MAX_BODY_SIZE {
            if let Some(session) = media_session {
                session.close("response_too_large");
            }
            return Err(LiveError::new(
                LiveErrorKind::Upstream,
                "Codex live response body too large",
            ));
        }
        if !(200..300).contains(&response.status) {
            if let Some(session) = media_session {
                session.close("upstream_rejected");
            }
            return Ok(response);
        }

        let location = header_first(&response.headers, "Location").unwrap_or_default();
        let call_id = call_id_from_location(location);
        if media_session.is_some() && call_id.is_none() {
            if let Some(session) = media_session {
                session.close("missing_call_id");
            }
            return Err(LiveError::new(
                LiveErrorKind::Upstream,
                "Codex live response is missing a valid call ID",
            ));
        }
        if let Some(session) = media_session {
            let answer = call_response_sdp(
                &response.body,
                header_first(&response.headers, "Content-Type").unwrap_or_default(),
            )?;
            let downstream_answer = session.accept_upstream_answer(&answer).await?;
            response.body = downstream_answer.into_bytes();
            header_set(&mut response.headers, "Content-Type", "application/sdp");
            if let Some(call_id) = call_id.as_ref() {
                session.set_call_id(call_id.as_str());
            }
            let stored = LiveSession::new(
                effective_auth.id.clone(),
                prepared.model,
                Some(session),
                now_millis,
            );
            self.sessions.put(call_id.expect("validated above"), stored);
        } else if let Some(call_id) = call_id {
            self.sessions.put(
                call_id,
                LiveSession::new(effective_auth.id.clone(), prepared.model, None, now_millis),
            );
        }
        Ok(response)
    }
}

pub fn read_limited_body(body: &[u8]) -> Result<Vec<u8>, LiveError> {
    if body.len() > MAX_BODY_SIZE {
        return Err(LiveError::new(
            LiveErrorKind::BodyTooLarge,
            "Codex live request body too large",
        ));
    }
    Ok(body.to_vec())
}

pub fn prepare_call_request(
    body: &[u8],
    content_type: &str,
) -> Result<PreparedCallRequest, LiveError> {
    let body = read_limited_body(body)?;
    if let Ok(media_type) = content_type.parse::<Mime>() {
        if media_type.type_() == mime::MULTIPART && media_type.subtype() == mime::FORM_DATA {
            let boundary = media_type
                .get_param(mime::BOUNDARY)
                .map(|value| value.as_str())
                .unwrap_or_default();
            return multipart_call_request(&body, boundary);
        }
    }
    Ok(PreparedCallRequest {
        model: model_from_json(&body).unwrap_or_else(|| DEFAULT_LIVE_MODEL.to_owned()),
        body,
        content_type: if content_type.trim().is_empty() {
            "application/json".to_owned()
        } else {
            content_type.to_owned()
        },
    })
}

pub fn multipart_call_request(
    body: &[u8],
    boundary: &str,
) -> Result<PreparedCallRequest, LiveError> {
    if boundary.trim().is_empty() {
        return Err(bad_request("Codex live multipart boundary is missing"));
    }
    let fields = parse_multipart_fields(body, boundary)?;
    let sdp = fields
        .get("sdp")
        .ok_or_else(|| bad_request("Codex live multipart body requires an sdp field"))?;
    let session = match fields.get("session") {
        Some(raw) => Some(
            serde_json::from_slice::<Value>(raw)
                .map_err(|_| bad_request("Codex live session field must contain valid JSON"))?,
        ),
        None => None,
    };
    let model = session
        .as_ref()
        .and_then(model_from_value)
        .unwrap_or_else(|| DEFAULT_LIVE_MODEL.to_owned());
    Ok(PreparedCallRequest {
        body: encode_call_request(String::from_utf8_lossy(sdp).as_ref(), session.as_ref())?,
        content_type: "application/json".to_owned(),
        model,
    })
}

fn parse_multipart_fields(
    body: &[u8],
    boundary: &str,
) -> Result<BTreeMap<String, Vec<u8>>, LiveError> {
    let marker = format!("--{boundary}").into_bytes();
    let mut fields = BTreeMap::new();
    for part in split_bytes(body, &marker).into_iter().skip(1) {
        let part = trim_crlf(part);
        if part.is_empty() || part == b"--" || part.starts_with(b"--\r\n") {
            continue;
        }
        let Some(split) = find_bytes(part, b"\r\n\r\n") else {
            return Err(bad_request("failed to parse Codex live multipart body"));
        };
        let headers = String::from_utf8_lossy(&part[..split]);
        let Some(name) = disposition_name(&headers) else {
            continue;
        };
        fields.insert(name, trim_final_crlf(&part[split + 4..]).to_vec());
    }
    Ok(fields)
}

fn disposition_name(headers: &str) -> Option<String> {
    headers.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if !name.trim().eq_ignore_ascii_case("content-disposition") {
            return None;
        }
        value.split(';').find_map(|parameter| {
            let (key, value) = parameter.trim().split_once('=')?;
            key.eq_ignore_ascii_case("name")
                .then(|| value.trim().trim_matches('"').to_owned())
        })
    })
}

fn split_bytes<'a>(body: &'a [u8], needle: &[u8]) -> Vec<&'a [u8]> {
    let mut result = Vec::new();
    let mut start = 0;
    while let Some(offset) = find_bytes(&body[start..], needle) {
        result.push(&body[start..start + offset]);
        start += offset + needle.len();
    }
    result.push(&body[start..]);
    result
}

fn find_bytes(body: &[u8], needle: &[u8]) -> Option<usize> {
    (!needle.is_empty())
        .then(|| {
            body.windows(needle.len())
                .position(|window| window == needle)
        })
        .flatten()
}

fn trim_crlf(mut value: &[u8]) -> &[u8] {
    while value.starts_with(b"\r\n") {
        value = &value[2..];
    }
    value
}

fn trim_final_crlf(value: &[u8]) -> &[u8] {
    value.strip_suffix(b"\r\n").unwrap_or(value)
}

pub fn encode_call_request(sdp: &str, session: Option<&Value>) -> Result<Vec<u8>, LiveError> {
    let mut payload = Map::new();
    payload.insert("sdp".to_owned(), Value::String(sdp.to_owned()));
    if let Some(session) = session {
        payload.insert("session".to_owned(), session.clone());
    }
    serde_json::to_vec(&Value::Object(payload))
        .map_err(|error| bad_request(format!("failed to encode Codex live request: {error}")))
}

pub fn call_request_sdp(body: &[u8], content_type: &str) -> Result<String, LiveError> {
    let mime = content_type.parse::<Mime>().ok();
    let is_raw = mime.as_ref().is_some_and(|value| {
        (value.type_() == mime::APPLICATION && value.subtype().as_str() == "sdp")
            || (value.type_() == mime::TEXT && value.subtype() == mime::PLAIN)
    });
    if is_raw {
        let value = String::from_utf8_lossy(body).into_owned();
        return (!value.trim().is_empty())
            .then_some(value)
            .ok_or_else(|| bad_request("Codex live call request requires an SDP offer"));
    }
    if !mime
        .as_ref()
        .is_some_and(|value| value.type_() == mime::APPLICATION && value.subtype() == mime::JSON)
    {
        return Err(bad_request(
            "Codex live media relay requires an SDP or JSON call request",
        ));
    }
    let value: Value = serde_json::from_slice(body).map_err(|error| {
        bad_request(format!("failed to decode Codex live call request: {error}"))
    })?;
    value
        .get("sdp")
        .and_then(Value::as_str)
        .filter(|sdp| !sdp.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| bad_request("Codex live call request requires an SDP offer"))
}

pub fn replace_call_request_sdp(
    body: &[u8],
    content_type: &str,
    sdp: &str,
) -> Result<(Vec<u8>, String), LiveError> {
    let mime = content_type.parse::<Mime>().ok();
    let is_raw = mime.as_ref().is_some_and(|value| {
        (value.type_() == mime::APPLICATION && value.subtype().as_str() == "sdp")
            || (value.type_() == mime::TEXT && value.subtype() == mime::PLAIN)
    });
    if is_raw {
        return Ok((
            encode_call_request(sdp, None)?,
            "application/json".to_owned(),
        ));
    }
    if !mime
        .as_ref()
        .is_some_and(|value| value.type_() == mime::APPLICATION && value.subtype() == mime::JSON)
    {
        return Err(bad_request(
            "Codex live media relay requires an SDP or JSON call request",
        ));
    }
    let mut value: Value = serde_json::from_slice(body).map_err(|error| {
        bad_request(format!("failed to decode Codex live call request: {error}"))
    })?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| bad_request("Codex live call request must be a JSON object"))?;
    object.insert("sdp".to_owned(), Value::String(sdp.to_owned()));
    let encoded = serde_json::to_vec(&value).map_err(|error| {
        bad_request(format!("failed to encode Codex live call request: {error}"))
    })?;
    Ok((encoded, "application/json".to_owned()))
}

pub fn call_response_sdp(body: &[u8], content_type: &str) -> Result<String, LiveError> {
    let is_json = content_type
        .parse::<Mime>()
        .ok()
        .is_some_and(|value| value.type_() == mime::APPLICATION && value.subtype() == mime::JSON);
    if is_json {
        let value: Value = serde_json::from_slice(body).map_err(|error| {
            LiveError::new(
                LiveErrorKind::Upstream,
                format!("failed to decode Codex live response: {error}"),
            )
        })?;
        return value
            .get("sdp")
            .and_then(Value::as_str)
            .filter(|sdp| !sdp.trim().is_empty())
            .map(str::to_owned)
            .ok_or_else(|| {
                LiveError::new(
                    LiveErrorKind::Upstream,
                    "Codex live response requires an SDP answer",
                )
            });
    }
    let answer = String::from_utf8_lossy(body).into_owned();
    (!answer.trim().is_empty())
        .then_some(answer)
        .ok_or_else(|| {
            LiveError::new(
                LiveErrorKind::Upstream,
                "Codex live response requires an SDP answer",
            )
        })
}

pub fn model_from_json(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<Value>(body)
        .ok()
        .as_ref()
        .and_then(model_from_value)
}

fn model_from_value(value: &Value) -> Option<String> {
    value
        .pointer("/session/model")
        .and_then(Value::as_str)
        .or_else(|| value.get("model").and_then(Value::as_str))
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_owned)
}

pub fn protocol_headers(source: &Headers) -> Headers {
    let mut destination = Headers::new();
    for allowed in LIVE_PROTOCOL_HEADERS {
        if let Some((_, values)) = source
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(allowed))
        {
            destination.insert(allowed.to_owned(), values.clone());
        }
    }
    destination
}

pub fn set_account_header(headers: &mut Headers, selected: &Auth) {
    if let Some(account_id) = selected
        .metadata
        .get("account_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        header_set(headers, "Chatgpt-Account-Id", account_id);
    }
}

pub fn headers_for_logging(source: &Headers) -> Headers {
    let mut headers = source.clone();
    if header_first(&headers, "X-Oai-Attestation").is_some() {
        header_set(&mut headers, "X-Oai-Attestation", "[REDACTED]");
    }
    headers
}

pub fn call_response_headers(source: &Headers) -> Headers {
    let mut headers = Headers::new();
    for allowed in ["Content-Type", "Location"] {
        if let Some((_, values)) = source
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(allowed))
        {
            headers.insert(allowed.to_owned(), values.clone());
        }
    }
    headers
}

pub fn media_credential_name(selected: Option<&Auth>, auth_index: &str) -> String {
    if let Some(selected) = selected {
        if !selected.label.trim().is_empty() {
            return selected.label.trim().to_owned();
        }
        if let Some(name) = Path::new(selected.file_name.trim())
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::trim)
            .filter(|name| !name.is_empty() && *name != ".")
        {
            return name.to_owned();
        }
    }
    auth_index.trim().to_owned()
}

pub fn proxy_url_for_auth(default_proxy_url: &str, selected: &Auth) -> String {
    if selected.proxy_url.trim().is_empty() {
        default_proxy_url.trim().to_owned()
    } else {
        selected.proxy_url.trim().to_owned()
    }
}

pub(crate) fn header_first<'a>(headers: &'a Headers, name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .and_then(|(_, values)| values.iter().find(|value| !value.trim().is_empty()))
        .map(String::as_str)
}

pub(crate) fn header_set(headers: &mut Headers, name: &str, value: &str) {
    if let Some(existing) = headers
        .keys()
        .find(|key| key.eq_ignore_ascii_case(name))
        .cloned()
    {
        headers.remove(&existing);
    }
    headers.insert(name.to_owned(), vec![value.to_owned()]);
}

fn bad_request(message: impl Into<String>) -> LiveError {
    LiveError::new(LiveErrorKind::BadRequest, message)
}
