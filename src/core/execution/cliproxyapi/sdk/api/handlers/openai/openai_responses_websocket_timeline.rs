// ref: sdk/api/handlers/openai/openai_responses_websocket_timeline.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Default)]
pub struct WebsocketTimeline {
    enabled: bool,
    events: Vec<Vec<u8>>,
}

impl WebsocketTimeline {
    #[must_use]
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            events: Vec::new(),
        }
    }

    pub fn begin_request(&mut self) {
        if self.enabled && !self.events.is_empty() {
            self.events.push(b"---\n".to_vec());
        }
    }

    pub fn append(&mut self, event_type: &str, payload: &[u8], timestamp_ms: u128) {
        if self.enabled {
            self.events.push(format_websocket_timeline_event(
                event_type,
                payload,
                timestamp_ms,
            ));
        }
    }

    #[must_use]
    pub fn body(&self) -> Vec<u8> {
        self.events.concat()
    }
}

#[must_use]
pub fn websocket_payload_event_type(payload: &[u8]) -> String {
    serde_json::from_slice::<serde_json::Value>(payload)
        .ok()
        .and_then(|value| value.get("type")?.as_str().map(str::to_owned))
        .unwrap_or_else(|| "websocket.payload".to_owned())
}

#[must_use]
pub fn format_websocket_timeline_event(
    event_type: &str,
    payload: &[u8],
    timestamp_ms: u128,
) -> Vec<u8> {
    let timestamp_ms = if timestamp_ms == 0 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis())
    } else {
        timestamp_ms
    };
    format!(
        "[{timestamp_ms}] {} {}\n",
        event_type.trim(),
        String::from_utf8_lossy(payload)
    )
    .into_bytes()
}
