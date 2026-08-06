// ref: sdk/api/handlers/openai/openai_responses_websocket_toolcall_repair.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

use serde_json::Value;

#[derive(Debug)]
pub struct WebsocketToolOutputCache {
    max_per_session: usize,
    sessions: Mutex<BTreeMap<String, VecDeque<(String, Value)>>>,
}

impl WebsocketToolOutputCache {
    #[must_use]
    pub fn new(max_per_session: usize) -> Self {
        Self {
            max_per_session: max_per_session.max(1),
            sessions: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn record(&self, session: &str, call_id: &str, item: Value) {
        if session.trim().is_empty() || call_id.trim().is_empty() {
            return;
        }
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let items = sessions.entry(session.to_owned()).or_default();
        items.retain(|(id, _)| id != call_id);
        items.push_back((call_id.to_owned(), item));
        while items.len() > self.max_per_session {
            items.pop_front();
        }
    }

    #[must_use]
    pub fn get(&self, session: &str, call_id: &str) -> Option<Value> {
        self.sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(session)?
            .iter()
            .find(|(id, _)| id == call_id)
            .map(|(_, item)| item.clone())
    }

    pub fn delete_session(&self, session: &str) {
        self.sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(session);
    }
}

#[must_use]
pub fn repair_responses_websocket_tool_calls(
    cache: &WebsocketToolOutputCache,
    session: &str,
    payload: &[u8],
) -> Vec<u8> {
    let Ok(Value::Object(mut document)) = serde_json::from_slice(payload) else {
        return payload.to_vec();
    };
    let Some(Value::Array(output)) = document.get_mut("output") else {
        return payload.to_vec();
    };
    for item in output.iter_mut() {
        let call_id = item
            .get("call_id")
            .and_then(Value::as_str)
            .map(str::to_owned);
        if item.get("type").and_then(Value::as_str) == Some("function_call_output") {
            if let Some(cached) = call_id.as_deref().and_then(|id| cache.get(session, id)) {
                *item = cached;
            }
        } else if let Some(call_id) = call_id {
            cache.record(session, &call_id, item.clone());
        }
    }
    serde_json::to_vec(&document).unwrap_or_else(|_| payload.to_vec())
}
