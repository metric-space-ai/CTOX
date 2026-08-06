// ref: internal/wsrelay/message.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// JSON envelope exchanged with relay clients.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "payload_is_empty")]
    pub payload: Option<Map<String, Value>>,
}

fn payload_is_empty(payload: &Option<Map<String, Value>>) -> bool {
    payload.as_ref().is_none_or(Map::is_empty)
}

impl Message {
    pub fn new(id: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            payload: None,
        }
    }

    pub fn with_payload(
        id: impl Into<String>,
        kind: impl Into<String>,
        payload: Map<String, Value>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            payload: Some(payload),
        }
    }

    pub(crate) fn is_terminal(&self) -> bool {
        matches!(
            self.kind.as_str(),
            MESSAGE_TYPE_HTTP_RESPONSE | MESSAGE_TYPE_ERROR | MESSAGE_TYPE_STREAM_END
        )
    }
}

pub const MESSAGE_TYPE_HTTP_REQUEST: &str = "http_request";
pub const MESSAGE_TYPE_HTTP_RESPONSE: &str = "http_response";
pub const MESSAGE_TYPE_STREAM_START: &str = "stream_start";
pub const MESSAGE_TYPE_STREAM_CHUNK: &str = "stream_chunk";
pub const MESSAGE_TYPE_STREAM_END: &str = "stream_end";
pub const MESSAGE_TYPE_ERROR: &str = "error";
pub const MESSAGE_TYPE_PING: &str = "ping";
pub const MESSAGE_TYPE_PONG: &str = "pong";
