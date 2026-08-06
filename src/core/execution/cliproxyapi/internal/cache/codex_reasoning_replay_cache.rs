// ref: internal/cache/codex_reasoning_replay_cache.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use gjson::Kind;

use crate::internal::signature::is_valid_gpt_reasoning_signature;

use super::antigravity_reasoning_replay_cache::{
    AntigravityReasoningReplayCache, AntigravityReasoningReplayError,
    AntigravityReasoningReplaySnapshot,
};

pub const CODEX_REASONING_REPLAY_TURN_TYPE: &str = "cpa_codex_replay_turn";
pub const CODEX_REASONING_REPLAY_TTL_MS: i64 = 60 * 60 * 1_000;
pub const CODEX_REASONING_REPLAY_MAX_ENTRIES: usize = 10_240;
pub const CODEX_REASONING_REPLAY_MAX_TURNS_PER_ENTRY: usize = 256;
pub const CODEX_REASONING_REPLAY_MAX_BYTES_PER_ENTRY: usize = 16 << 20;
pub const CODEX_REASONING_REPLAY_EVICT_BATCH_SIZE: usize = 128;
const MAX_APPEND_ATTEMPTS: usize = 32;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CodexReasoningReplaySnapshot {
    inner: AntigravityReasoningReplaySnapshot,
}

/// Revisioned in-memory Codex replay state.
///
/// The state, hash, branch, eviction and tombstone machinery is shared with
/// the Antigravity replay cache. Codex owns only its item normalization and
/// turn-chain policy.
#[derive(Debug)]
pub struct CodexReasoningReplayCache {
    state: AntigravityReasoningReplayCache,
}

impl Default for CodexReasoningReplayCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexReasoningReplayCache {
    #[must_use]
    pub fn new() -> Self {
        Self::with_limits(
            CODEX_REASONING_REPLAY_MAX_ENTRIES,
            CODEX_REASONING_REPLAY_EVICT_BATCH_SIZE,
        )
    }

    pub(crate) fn with_limits(max_entries: usize, evict_batch: usize) -> Self {
        Self {
            state: AntigravityReasoningReplayCache::with_namespace_limits(
                b"codex-reasoning-replay\0",
                max_entries,
                evict_batch,
            ),
        }
    }

    pub fn read(
        &self,
        model: &str,
        session_key: &str,
        now_ms: i64,
    ) -> Result<(Vec<Vec<u8>>, CodexReasoningReplaySnapshot, bool), CodexReasoningReplayError> {
        self.state
            .read(model, session_key, now_ms)
            .map(|(items, snapshot, found)| {
                (
                    items,
                    CodexReasoningReplaySnapshot { inner: snapshot },
                    found,
                )
            })
            .map_err(CodexReasoningReplayError::from)
    }

    pub fn replace_if_unchanged(
        &self,
        model: &str,
        session_key: &str,
        snapshot: &CodexReasoningReplaySnapshot,
        items: &[Vec<u8>],
        now_ms: i64,
    ) -> Result<bool, CodexReasoningReplayError> {
        let normalized = normalize_items(items)?;
        self.state
            .replace_normalized_if_unchanged(
                model,
                session_key,
                &snapshot.inner,
                normalized,
                now_ms,
            )
            .map_err(CodexReasoningReplayError::from)
    }

    pub fn delete_if_unchanged(
        &self,
        model: &str,
        session_key: &str,
        snapshot: &CodexReasoningReplaySnapshot,
        now_ms: i64,
    ) -> Result<bool, CodexReasoningReplayError> {
        self.state
            .delete_if_unchanged(model, session_key, &snapshot.inner, now_ms)
            .map_err(CodexReasoningReplayError::from)
    }

    /// Appends a completed turn using bounded optimistic retries. Every retry
    /// reloads the latest branch, so concurrent successful turns accumulate
    /// without allowing a writer to cross a clear/tombstone boundary.
    pub fn append_turn(
        &self,
        model: &str,
        session_key: &str,
        turn: &[Vec<u8>],
        now_ms: i64,
    ) -> Result<bool, CodexReasoningReplayError> {
        let turn = normalize_items(turn)?;
        for _ in 0..MAX_APPEND_ATTEMPTS {
            let (existing, snapshot, _) = self.read(model, session_key, now_ms)?;
            let combined = append_turn(existing, &turn);
            if self.state.replace_normalized_if_unchanged(
                model,
                session_key,
                &snapshot.inner,
                combined,
                now_ms,
            )? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn clear(&self) {
        self.state.clear();
    }

    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.state.entry_count()
    }
}

fn normalize_items(items: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, CodexReasoningReplayError> {
    let normalized = items
        .iter()
        .filter_map(|item| normalize_item(item))
        .collect::<Vec<_>>();
    let normalized = trim_items(normalized);
    if normalized.is_empty() {
        Err(CodexReasoningReplayError::InvalidItems)
    } else {
        Ok(normalized)
    }
}

fn normalize_item(raw: &[u8]) -> Option<Vec<u8>> {
    let document = std::str::from_utf8(raw).ok()?;
    let root = gjson::parse(document);
    match root.get("type").str().trim() {
        CODEX_REASONING_REPLAY_TURN_TYPE => normalize_turn(&root),
        "reasoning" => normalize_reasoning(&root),
        "function_call" => normalize_function_call(&root),
        "custom_tool_call" => normalize_custom_tool_call(&root),
        _ => None,
    }
}

fn normalize_turn(root: &gjson::Value<'_>) -> Option<Vec<u8>> {
    let id = root.get("id");
    let id = id.str().trim();
    if id.is_empty() {
        return None;
    }
    let mut output = format!(
        "{{\"type\":\"{CODEX_REASONING_REPLAY_TURN_TYPE}\",\"id\":{}}}",
        json_string(id)
    );
    for key in ["assistant_fingerprint", "request_fingerprint"] {
        let value = root.get(key);
        let value = value.str().trim();
        if !value.is_empty() {
            insert_before_close(&mut output, key, &json_string(value));
        }
    }
    let call_ids = root.get("call_ids");
    if call_ids.kind() == Kind::Array {
        let values = call_ids
            .array()
            .iter()
            .map(gjson::Value::str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(json_string)
            .collect::<Vec<_>>();
        if !values.is_empty() {
            insert_before_close(&mut output, "call_ids", &format!("[{}]", values.join(",")));
        }
    }
    Some(output.into_bytes())
}

fn normalize_reasoning(root: &gjson::Value<'_>) -> Option<Vec<u8>> {
    let encrypted = root.get("encrypted_content");
    if encrypted.kind() != Kind::String {
        return None;
    }
    let encrypted = encrypted.str();
    if encrypted != encrypted.trim() || !is_valid_gpt_reasoning_signature(encrypted) {
        return None;
    }
    Some(
        format!(
            "{{\"type\":\"reasoning\",\"summary\":[],\"content\":null,\"encrypted_content\":{}}}",
            json_string(encrypted)
        )
        .into_bytes(),
    )
}

fn normalize_function_call(root: &gjson::Value<'_>) -> Option<Vec<u8>> {
    let call_id = root.get("call_id");
    let call_id = call_id.str().trim();
    let name = root.get("name");
    let name = name.str().trim();
    let arguments = root.get("arguments");
    if call_id.is_empty() || name.is_empty() || arguments.kind() != Kind::String {
        return None;
    }
    Some(
        format!(
            "{{\"type\":\"function_call\",\"call_id\":{},\"name\":{},\"arguments\":{}}}",
            json_string(call_id),
            json_string(name),
            json_string(arguments.str())
        )
        .into_bytes(),
    )
}

fn normalize_custom_tool_call(root: &gjson::Value<'_>) -> Option<Vec<u8>> {
    let call_id = root.get("call_id");
    let call_id = call_id.str().trim();
    let name = root.get("name");
    let name = name.str().trim();
    let input = root.get("input");
    if call_id.is_empty() || name.is_empty() || !input.exists() {
        return None;
    }
    let status = root.get("status");
    let status = status.str().trim();
    let status = if status.is_empty() {
        "completed"
    } else {
        status
    };
    let input = if input.kind() == Kind::String {
        json_string(input.str())
    } else {
        input.json().to_owned()
    };
    Some(
        format!(
            "{{\"type\":\"custom_tool_call\",\"status\":{},\"call_id\":{},\"name\":{},\"input\":{input}}}",
            json_string(status),
            json_string(call_id),
            json_string(name)
        )
        .into_bytes(),
    )
}

fn append_turn(mut existing: Vec<Vec<u8>>, turn: &[Vec<u8>]) -> Vec<Vec<u8>> {
    if existing
        .first()
        .is_some_and(|item| item_type(item) != CODEX_REASONING_REPLAY_TURN_TYPE)
    {
        existing.clear();
    }
    let turn_id = turn
        .first()
        .filter(|item| item_type(item) == CODEX_REASONING_REPLAY_TURN_TYPE)
        .and_then(|item| item_string(item, "id"));
    if turn_id.as_deref().is_some_and(|turn_id| {
        existing.iter().any(|item| {
            item_type(item) == CODEX_REASONING_REPLAY_TURN_TYPE
                && item_string(item, "id").as_deref() == Some(turn_id)
        })
    }) {
        return trim_items(existing);
    }
    existing.extend(turn.iter().cloned());
    trim_items(existing)
}

fn trim_items(mut items: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    loop {
        let mut turn_starts = vec![0_usize];
        let mut total_bytes = 0_usize;
        for (index, item) in items.iter().enumerate() {
            total_bytes = total_bytes.saturating_add(item.len());
            if index > 0 && item_type(item) == CODEX_REASONING_REPLAY_TURN_TYPE {
                turn_starts.push(index);
            }
        }
        if turn_starts.len() <= CODEX_REASONING_REPLAY_MAX_TURNS_PER_ENTRY
            && total_bytes <= CODEX_REASONING_REPLAY_MAX_BYTES_PER_ENTRY
        {
            return items;
        }
        if turn_starts.len() <= 1 {
            return Vec::new();
        }
        items.drain(..turn_starts[1]);
    }
}

fn item_type(item: &[u8]) -> String {
    std::str::from_utf8(item)
        .ok()
        .map(gjson::parse)
        .map(|root| root.get("type").str().trim().to_owned())
        .unwrap_or_default()
}

fn item_string(item: &[u8], field: &str) -> Option<String> {
    let document = std::str::from_utf8(item).ok()?;
    let root = gjson::parse(document);
    let value = root.get(field);
    let value = value.str().trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn json_string(value: &str) -> String {
    serde_json::to_string(value).expect("strings serialize")
}

fn insert_before_close(output: &mut String, key: &str, raw_value: &str) {
    output.pop();
    output.push(',');
    output.push_str(&json_string(key));
    output.push(':');
    output.push_str(raw_value);
    output.push('}');
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodexReasoningReplayError {
    InvalidKey,
    InvalidSnapshot,
    InvalidItems,
}

impl From<AntigravityReasoningReplayError> for CodexReasoningReplayError {
    fn from(error: AntigravityReasoningReplayError) -> Self {
        match error {
            AntigravityReasoningReplayError::InvalidKey => Self::InvalidKey,
            AntigravityReasoningReplayError::InvalidSnapshot => Self::InvalidSnapshot,
            AntigravityReasoningReplayError::InvalidItems
            | AntigravityReasoningReplayError::TooLarge => Self::InvalidItems,
        }
    }
}

impl std::fmt::Display for CodexReasoningReplayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Codex reasoning replay state is invalid")
    }
}

impl std::error::Error for CodexReasoningReplayError {}
