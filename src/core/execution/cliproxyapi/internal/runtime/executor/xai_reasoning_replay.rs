// ref: internal/runtime/executor/xai_reasoning_replay.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct XaiReasoningReplayScope {
    pub provider: String,
    pub session_key: String,
    pub credential_scope: String,
}

impl XaiReasoningReplayScope {
    #[must_use]
    pub fn new(provider: &str, session_key: &str, credential: Option<&str>) -> Option<Self> {
        let provider = provider.trim();
        let session_key = session_key.trim();
        if provider.is_empty() || session_key.is_empty() {
            return None;
        }
        let credential_scope = credential
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(|value| format!("{:x}", Sha256::digest(value.as_bytes())))
            .unwrap_or_else(|| "trusted-session".into());
        Some(Self {
            provider: provider.into(),
            session_key: session_key.into(),
            credential_scope,
        })
    }
    #[must_use]
    pub fn key(&self) -> String {
        format!(
            "{}:{}:{}",
            self.provider, self.session_key, self.credential_scope
        )
    }
}

pub trait XaiReasoningReplayStore: Send + Sync {
    fn load(&self, key: &str) -> Result<Vec<Vec<u8>>, String>;
    fn store(&self, key: &str, items: &[Vec<u8>]) -> Result<(), String>;
    fn clear(&self, key: &str) -> Result<(), String>;
}

#[must_use]
pub fn apply_reasoning_replay(
    store: &dyn XaiReasoningReplayStore,
    scope: Option<&XaiReasoningReplayScope>,
    body: &[u8],
) -> Vec<u8> {
    let Some(scope) = scope else {
        return body.to_vec();
    };
    let Ok(items) = store.load(&scope.key()) else {
        return body.to_vec();
    };
    let filtered = filter_replay_items_for_input(body, &items);
    if filtered.is_empty() {
        return body.to_vec();
    }
    let Ok(mut root) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    let Some(input) = root.get_mut("input").and_then(Value::as_array_mut) else {
        return body.to_vec();
    };
    let mut replay: Vec<Value> = filtered
        .iter()
        .filter_map(|item| serde_json::from_slice(item).ok())
        .collect();
    replay.append(input);
    *input = replay;
    serde_json::to_vec(&root).unwrap_or_else(|_| body.to_vec())
}

#[must_use]
pub fn filter_replay_items_for_input(body: &[u8], items: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let Ok(root) = serde_json::from_slice::<Value>(body) else {
        return Vec::new();
    };
    let input = root
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let encrypted: BTreeSet<_> = input
        .iter()
        .filter_map(|item| item.get("encrypted_content").and_then(Value::as_str))
        .collect();
    let last_assistant = input
        .iter()
        .rev()
        .find(|item| item.get("role").and_then(Value::as_str) == Some("assistant"));
    items
        .iter()
        .filter(|raw| {
            let Ok(item) = serde_json::from_slice::<Value>(raw) else {
                return false;
            };
            if item.get("type").and_then(Value::as_str) == Some("reasoning") {
                return item
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .is_none_or(|content| !encrypted.contains(content));
            }
            if item.get("role").and_then(Value::as_str) == Some("assistant") {
                return last_assistant
                    .is_none_or(|current| current.get("content") != item.get("content"));
            }
            true
        })
        .cloned()
        .collect()
}

pub fn cache_reasoning_replay_from_completed(
    store: &dyn XaiReasoningReplayStore,
    scope: Option<&XaiReasoningReplayScope>,
    completed: &[u8],
) {
    let Some(scope) = scope else {
        return;
    };
    let items: Vec<Vec<u8>> = serde_json::from_slice::<Value>(completed)
        .ok()
        .and_then(|event| {
            event
                .pointer("/response/output")
                .and_then(Value::as_array)
                .cloned()
        })
        .unwrap_or_default()
        .into_iter()
        .filter(|item| {
            matches!(
                item.get("type").and_then(Value::as_str),
                Some("reasoning" | "function_call")
            ) || item.get("role").and_then(Value::as_str) == Some("assistant")
        })
        .filter_map(|item| serde_json::to_vec(&item).ok())
        .collect();
    if items.is_empty() {
        let _ = store.clear(&scope.key());
    } else {
        let _ = store.store(&scope.key(), &items);
    }
}

pub fn clear_reasoning_replay_after_compaction(
    store: &dyn XaiReasoningReplayStore,
    scope: Option<&XaiReasoningReplayScope>,
) {
    if let Some(scope) = scope {
        let _ = store.clear(&scope.key());
    }
}
