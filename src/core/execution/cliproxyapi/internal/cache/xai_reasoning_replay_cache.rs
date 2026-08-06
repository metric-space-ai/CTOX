// ref: internal/cache/xai_reasoning_replay_cache.go:15-397 @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: Home KV is supplied by the enclosing Rust gateway.
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{Map, Value};

use crate::internal::signature::is_valid_grok_encrypted_content;

use super::antigravity_reasoning_replay_cache::{
    AntigravityReasoningReplayCache, AntigravityReasoningReplayError,
    AntigravityReasoningReplaySnapshot,
};

pub const XAI_REASONING_REPLAY_TTL_MS: i64 = 60 * 60 * 1_000;
pub const XAI_REASONING_REPLAY_MAX_ENTRIES: usize = 10_240;
pub const XAI_REASONING_REPLAY_EVICT_BATCH_SIZE: usize = 128;

pub type XaiReasoningReplaySnapshot = AntigravityReasoningReplaySnapshot;
pub type XaiReasoningReplayError = AntigravityReasoningReplayError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XaiReasoningReplayStoreStatus {
    InvalidArgs,
    Stored,
    NoReplayableState,
    BackendError,
}

#[derive(Debug)]
pub struct XaiReasoningReplayCache {
    state: AntigravityReasoningReplayCache,
}

impl Default for XaiReasoningReplayCache {
    fn default() -> Self {
        Self::new()
    }
}

impl XaiReasoningReplayCache {
    pub fn new() -> Self {
        Self::with_limits(
            XAI_REASONING_REPLAY_MAX_ENTRIES,
            XAI_REASONING_REPLAY_EVICT_BATCH_SIZE,
        )
    }

    fn with_limits(max_entries: usize, evict_batch: usize) -> Self {
        Self {
            state: AntigravityReasoningReplayCache::with_namespace_limits(
                b"xai-reasoning-replay\0",
                max_entries,
                evict_batch,
            ),
        }
    }

    pub fn read(
        &self,
        model_name: &str,
        session_key: &str,
        now_ms: i64,
    ) -> Result<(Vec<Vec<u8>>, XaiReasoningReplaySnapshot, bool), XaiReasoningReplayError> {
        self.state.read(model_name, session_key, now_ms)
    }

    pub fn replace_if_unchanged(
        &self,
        model_name: &str,
        session_key: &str,
        snapshot: &XaiReasoningReplaySnapshot,
        items: &[Vec<u8>],
        now_ms: i64,
    ) -> Result<bool, XaiReasoningReplayError> {
        let normalized = normalize_items(items).ok_or(XaiReasoningReplayError::InvalidItems)?;
        self.state
            .replace_exact_if_unchanged(model_name, session_key, snapshot, normalized, now_ms)
    }

    pub fn store(
        &self,
        model_name: &str,
        session_key: &str,
        items: &[Vec<u8>],
        now_ms: i64,
    ) -> XaiReasoningReplayStoreStatus {
        if model_name.trim().is_empty() || session_key.trim().is_empty() {
            return XaiReasoningReplayStoreStatus::InvalidArgs;
        }
        let Some(normalized) = normalize_items(items) else {
            return XaiReasoningReplayStoreStatus::NoReplayableState;
        };
        for _ in 0..64 {
            let Ok((_, snapshot, _)) = self.state.read(model_name, session_key, now_ms) else {
                return XaiReasoningReplayStoreStatus::InvalidArgs;
            };
            match self.state.replace_exact_if_unchanged(
                model_name,
                session_key,
                &snapshot,
                normalized.clone(),
                now_ms,
            ) {
                Ok(true) => return XaiReasoningReplayStoreStatus::Stored,
                Ok(false) => continue,
                Err(_) => return XaiReasoningReplayStoreStatus::BackendError,
            }
        }
        XaiReasoningReplayStoreStatus::BackendError
    }

    pub fn delete_if_unchanged(
        &self,
        model_name: &str,
        session_key: &str,
        snapshot: &XaiReasoningReplaySnapshot,
        now_ms: i64,
    ) -> Result<bool, XaiReasoningReplayError> {
        self.state
            .delete_if_unchanged(model_name, session_key, snapshot, now_ms)
    }

    pub fn clear(&self) {
        self.state.clear();
    }

    #[cfg(test)]
    pub(crate) fn test_with_limits(max_entries: usize, evict_batch: usize) -> Self {
        Self::with_limits(max_entries, evict_batch)
    }

    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.state.entry_count()
    }
}

fn normalize_items(items: &[Vec<u8>]) -> Option<Vec<Vec<u8>>> {
    let mut normalized = Vec::with_capacity(items.len());
    let mut has_replay_anchor = false;
    for item in items {
        let Some((item, anchor)) = normalize_item(item) else {
            continue;
        };
        normalized.push(item);
        has_replay_anchor |= anchor;
    }
    has_replay_anchor.then_some(normalized)
}

fn normalize_item(raw: &[u8]) -> Option<(Vec<u8>, bool)> {
    let value = serde_json::from_slice::<Value>(raw).ok()?;
    match value.get("type")?.as_str()?.trim() {
        "reasoning" => normalize_reasoning(&value).map(|item| (item, true)),
        "message" => normalize_message(&value).map(|item| (item, false)),
        "function_call" => normalize_function_call(&value).map(|item| (item, true)),
        "custom_tool_call" => normalize_custom_tool_call(&value).map(|item| (item, true)),
        _ => None,
    }
}

fn normalize_reasoning(value: &Value) -> Option<Vec<u8>> {
    let encrypted = value.get("encrypted_content")?.as_str()?;
    if encrypted.trim() != encrypted || !is_valid_grok_encrypted_content(encrypted) {
        return None;
    }
    serde_json::to_vec(&serde_json::json!({
        "type": "reasoning",
        "summary": [],
        "content": null,
        "encrypted_content": encrypted,
    }))
    .ok()
}

fn normalize_message(value: &Value) -> Option<Vec<u8>> {
    if !value
        .get("role")?
        .as_str()?
        .trim()
        .eq_ignore_ascii_case("assistant")
    {
        return None;
    }
    let parts = value.get("content")?.as_array()?;
    if parts.is_empty() {
        return None;
    }
    let content = parts
        .iter()
        .filter_map(|part| match part.get("type")?.as_str()?.trim() {
            "output_text" => Some(serde_json::json!({
                "type": "output_text",
                "text": part.get("text")?.as_str()?,
            })),
            "refusal" => Some(serde_json::json!({
                "type": "refusal",
                "refusal": part.get("refusal")?.as_str()?,
            })),
            _ => None,
        })
        .collect::<Vec<_>>();
    if content.is_empty() {
        return None;
    }
    serde_json::to_vec(&serde_json::json!({
        "type": "message",
        "role": "assistant",
        "content": content,
    }))
    .ok()
}

fn normalize_function_call(value: &Value) -> Option<Vec<u8>> {
    let call_id = value.get("call_id")?.as_str()?.trim();
    let name = value.get("name")?.as_str()?.trim();
    let arguments = value.get("arguments")?.as_str()?;
    if call_id.is_empty() || name.is_empty() {
        return None;
    }
    serde_json::to_vec(&serde_json::json!({
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }))
    .ok()
}

fn normalize_custom_tool_call(value: &Value) -> Option<Vec<u8>> {
    let call_id = value.get("call_id")?.as_str()?.trim();
    let name = value.get("name")?.as_str()?.trim();
    let input = value.get("input")?;
    if call_id.is_empty() || name.is_empty() {
        return None;
    }
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|status| !status.is_empty())
        .unwrap_or("completed");
    let mut object = Map::new();
    object.insert(
        "type".to_owned(),
        Value::String("custom_tool_call".to_owned()),
    );
    object.insert("status".to_owned(), Value::String(status.to_owned()));
    object.insert("call_id".to_owned(), Value::String(call_id.to_owned()));
    object.insert("name".to_owned(), Value::String(name.to_owned()));
    object.insert("input".to_owned(), input.clone());
    serde_json::to_vec(&Value::Object(object)).ok()
}
