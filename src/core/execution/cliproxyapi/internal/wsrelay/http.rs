// ref: internal/wsrelay/http.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;

use chrono::{DateTime, SecondsFormat, Utc};
use serde_json::{Map, Value};
use tokio::sync::mpsc;

use super::manager::Manager;
use super::message::{
    Message, MESSAGE_TYPE_ERROR, MESSAGE_TYPE_HTTP_REQUEST, MESSAGE_TYPE_HTTP_RESPONSE,
    MESSAGE_TYPE_STREAM_CHUNK, MESSAGE_TYPE_STREAM_END, MESSAGE_TYPE_STREAM_START,
};

use super::session::{RelayCancellation, RelayError};

pub type HeaderMap = BTreeMap<String, Vec<String>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpRequest {
    pub method: String,
    pub url: String,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

#[derive(Debug)]
pub struct StreamEvent {
    pub kind: String,
    pub payload: Vec<u8>,
    pub status: u16,
    pub headers: HeaderMap,
    pub error: Option<RelayError>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RelayHandshake {
    pub method: String,
    pub path: String,
    pub headers: HeaderMap,
    /// Opaque identity supplied by the host after its own authentication.
    pub principal: Option<String>,
}

/// Host authority boundary. Implementations resolve a provider identity from
/// already-authenticated request metadata; relay code never reads secrets or
/// ambient process configuration.
pub trait RelayAuthority: Send + Sync {
    fn authorize(&self, request: &RelayHandshake) -> Result<String, RelayError>;
}

/// Minimal authority adapter for hosts that have already authenticated the
/// upgrade request and attached an opaque principal.
pub struct PrincipalAuthority;

impl RelayAuthority for PrincipalAuthority {
    fn authorize(&self, request: &RelayHandshake) -> Result<String, RelayError> {
        request
            .principal
            .as_deref()
            .map(str::trim)
            .filter(|principal| !principal.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| RelayError::Unauthorized("missing relay principal".into()))
    }
}

pub fn encode_request(request: &HttpRequest, sent_at: DateTime<Utc>) -> Map<String, Value> {
    let headers = request
        .headers
        .iter()
        .map(|(key, values)| {
            (
                key.clone(),
                Value::Array(values.iter().cloned().map(Value::String).collect()),
            )
        })
        .collect();
    Map::from_iter([
        ("method".into(), Value::String(request.method.clone())),
        ("url".into(), Value::String(request.url.clone())),
        ("headers".into(), Value::Object(headers)),
        (
            "body".into(),
            Value::String(String::from_utf8_lossy(&request.body).into_owned()),
        ),
        (
            "sent_at".into(),
            Value::String(sent_at.to_rfc3339_opts(SecondsFormat::AutoSi, true)),
        ),
    ])
}

pub fn decode_response(payload: Option<&Map<String, Value>>) -> HttpResponse {
    let Some(payload) = payload else {
        return HttpResponse {
            status: 502,
            headers: HeaderMap::new(),
            body: Vec::new(),
        };
    };
    let status = payload.get("status").and_then(status_value).unwrap_or(200);
    let headers = payload
        .get("headers")
        .and_then(Value::as_object)
        .map(decode_headers)
        .unwrap_or_default();
    let body = payload
        .get("body")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .as_bytes()
        .to_vec();
    HttpResponse {
        status,
        headers,
        body,
    }
}

pub fn decode_chunk(payload: Option<&Map<String, Value>>) -> Vec<u8> {
    let Some(payload) = payload else {
        return Vec::new();
    };
    payload
        .get("data")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .as_bytes()
        .to_vec()
}

pub fn decode_error(payload: Option<&Map<String, Value>>) -> RelayError {
    let Some(payload) = payload else {
        return RelayError::Upstream {
            message: "wsrelay: unknown error".into(),
            status: 0,
        };
    };
    let message = payload
        .get("error")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or("wsrelay: upstream error");
    let status = payload.get("status").and_then(status_value).unwrap_or(0);
    RelayError::Upstream {
        message: message.to_owned(),
        status,
    }
}

fn decode_headers(raw: &Map<String, Value>) -> HeaderMap {
    raw.iter()
        .filter_map(|(key, value)| {
            let values = match value {
                Value::Array(items) => items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect(),
                Value::String(value) => vec![value.clone()],
                _ => Vec::new(),
            };
            (!values.is_empty()).then(|| (key.clone(), values))
        })
        .collect()
}

fn status_value(value: &Value) -> Option<u16> {
    if let Some(value) = value.as_u64() {
        return u16::try_from(value).ok();
    }
    value
        .as_f64()
        .filter(|value| value.is_finite() && *value >= 0.0 && *value <= u16::MAX as f64)
        .map(|value| value as u16)
}

impl fmt::Display for StreamEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.kind)
    }
}

impl Manager {
    pub async fn non_stream(
        &self,
        cancellation: RelayCancellation,
        provider: &str,
        request: Option<&HttpRequest>,
    ) -> Result<HttpResponse, RelayError> {
        let request = request.ok_or_else(|| RelayError::InvalidRequest("request is nil".into()))?;
        let message = Message::with_payload(
            uuid::Uuid::new_v4().to_string(),
            MESSAGE_TYPE_HTTP_REQUEST,
            encode_request(request, self.inner.clock.utc_now()),
        );
        let mut responses = self.send(cancellation.clone(), provider, message).await?;
        let mut stream_response: Option<HttpResponse> = None;
        let mut stream_body = Vec::new();
        loop {
            let message = tokio::select! {
                () = cancellation.cancelled() => return Err(RelayError::Cancelled),
                message = responses.recv() => message,
            };
            let Some(message) = message else {
                if let Some(mut response) = stream_response {
                    response.body = stream_body;
                    return Ok(response);
                }
                return Err(RelayError::Closed);
            };
            match message.kind.as_str() {
                MESSAGE_TYPE_HTTP_RESPONSE => {
                    let mut response = decode_response(message.payload.as_ref());
                    if stream_response.is_some()
                        && !stream_body.is_empty()
                        && response.body.is_empty()
                    {
                        response.body = stream_body;
                    }
                    return Ok(response);
                }
                MESSAGE_TYPE_ERROR => return Err(decode_error(message.payload.as_ref())),
                MESSAGE_TYPE_STREAM_START => {
                    stream_response = Some(decode_response(message.payload.as_ref()));
                    stream_body.clear();
                }
                MESSAGE_TYPE_STREAM_CHUNK => {
                    if stream_response.is_none() {
                        stream_response = Some(HttpResponse {
                            status: 200,
                            headers: HeaderMap::new(),
                            body: Vec::new(),
                        });
                    }
                    stream_body.extend(decode_chunk(message.payload.as_ref()));
                }
                MESSAGE_TYPE_STREAM_END => {
                    let mut response = stream_response.unwrap_or(HttpResponse {
                        status: 200,
                        headers: HeaderMap::new(),
                        body: Vec::new(),
                    });
                    response.body = stream_body;
                    return Ok(response);
                }
                _ => {}
            }
        }
    }

    pub async fn stream(
        &self,
        cancellation: RelayCancellation,
        provider: &str,
        request: Option<&HttpRequest>,
    ) -> Result<mpsc::Receiver<StreamEvent>, RelayError> {
        let request = request.ok_or_else(|| RelayError::InvalidRequest("request is nil".into()))?;
        let message = Message::with_payload(
            uuid::Uuid::new_v4().to_string(),
            MESSAGE_TYPE_HTTP_REQUEST,
            encode_request(request, self.inner.clock.utc_now()),
        );
        let mut responses = self.send(cancellation.clone(), provider, message).await?;
        let (output, receiver) = mpsc::channel(self.inner.limits.response_capacity);
        tokio::spawn(async move {
            loop {
                let message = tokio::select! {
                    () = cancellation.cancelled() => return,
                    message = responses.recv() => message,
                };
                let Some(message) = message else {
                    let _ = output
                        .send(StreamEvent {
                            kind: MESSAGE_TYPE_ERROR.into(),
                            payload: Vec::new(),
                            status: 0,
                            headers: HeaderMap::new(),
                            error: Some(RelayError::Closed),
                        })
                        .await;
                    return;
                };
                let terminal = message.is_terminal();
                let event = match message.kind.as_str() {
                    MESSAGE_TYPE_STREAM_START => {
                        let response = decode_response(message.payload.as_ref());
                        Some(StreamEvent {
                            kind: MESSAGE_TYPE_STREAM_START.into(),
                            payload: Vec::new(),
                            status: response.status,
                            headers: response.headers,
                            error: None,
                        })
                    }
                    MESSAGE_TYPE_STREAM_CHUNK => Some(StreamEvent {
                        kind: MESSAGE_TYPE_STREAM_CHUNK.into(),
                        payload: decode_chunk(message.payload.as_ref()),
                        status: 0,
                        headers: HeaderMap::new(),
                        error: None,
                    }),
                    MESSAGE_TYPE_STREAM_END => Some(StreamEvent {
                        kind: MESSAGE_TYPE_STREAM_END.into(),
                        payload: Vec::new(),
                        status: 0,
                        headers: HeaderMap::new(),
                        error: None,
                    }),
                    MESSAGE_TYPE_ERROR => Some(StreamEvent {
                        kind: MESSAGE_TYPE_ERROR.into(),
                        payload: Vec::new(),
                        status: 0,
                        headers: HeaderMap::new(),
                        error: Some(decode_error(message.payload.as_ref())),
                    }),
                    MESSAGE_TYPE_HTTP_RESPONSE => {
                        let response = decode_response(message.payload.as_ref());
                        Some(StreamEvent {
                            kind: MESSAGE_TYPE_HTTP_RESPONSE.into(),
                            payload: response.body,
                            status: response.status,
                            headers: response.headers,
                            error: None,
                        })
                    }
                    _ => None,
                };
                if let Some(event) = event {
                    let sent = tokio::select! {
                        () = cancellation.cancelled() => false,
                        result = output.send(event) => result.is_ok(),
                    };
                    if !sent {
                        return;
                    }
                }
                if terminal {
                    return;
                }
            }
        });
        Ok(receiver)
    }
}
