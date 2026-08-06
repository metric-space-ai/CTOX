// ref: internal/cache/kimi_thinking_replay_cache.go:18-426 @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: Home KV is supplied by the enclosing Rust gateway.
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::antigravity_reasoning_replay_cache::{
    AntigravityReasoningReplayCache, AntigravityReasoningReplayError,
    AntigravityReasoningReplaySnapshot,
};

pub const KIMI_THINKING_REPLAY_TTL_MS: i64 = 60 * 60 * 1_000;
pub const KIMI_THINKING_REPLAY_MAX_ENTRIES: usize = 10_240;
pub const KIMI_THINKING_REPLAY_EVICT_BATCH_SIZE: usize = 128;
pub const KIMI_THINKING_REPLAY_MAX_BYTES_PER_ENTRY: usize = 8 << 20;
pub const KIMI_THINKING_REPLAY_MAX_BLOCKS_PER_ENTRY: usize = 512;
pub const KIMI_THINKING_REPLAY_MAX_TOTAL_BYTES: usize = 256 << 20;

pub type KimiThinkingReplaySnapshot = AntigravityReasoningReplaySnapshot;
pub type KimiThinkingReplayError = AntigravityReasoningReplayError;

#[derive(Debug)]
pub struct KimiThinkingReplayCache {
    state: AntigravityReasoningReplayCache,
}

impl Default for KimiThinkingReplayCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KimiThinkingReplayCache {
    pub fn new() -> Self {
        Self::with_limits(
            KIMI_THINKING_REPLAY_MAX_ENTRIES,
            KIMI_THINKING_REPLAY_EVICT_BATCH_SIZE,
            KIMI_THINKING_REPLAY_MAX_TOTAL_BYTES,
        )
    }

    fn with_limits(max_entries: usize, evict_batch: usize, max_total_bytes: usize) -> Self {
        Self {
            state: AntigravityReasoningReplayCache::with_namespace_capacity(
                b"kimi-thinking-replay\0",
                max_entries,
                evict_batch,
                max_total_bytes,
            ),
        }
    }

    pub fn read(
        &self,
        model_family: &str,
        session_key: &str,
        now_ms: i64,
    ) -> Result<(Vec<u8>, KimiThinkingReplaySnapshot, bool), KimiThinkingReplayError> {
        let (items, snapshot, found) = self.state.read(model_family, session_key, now_ms)?;
        Ok((
            items.into_iter().next().unwrap_or_default(),
            snapshot,
            found,
        ))
    }

    pub fn replace_if_unchanged(
        &self,
        model_family: &str,
        session_key: &str,
        snapshot: &KimiThinkingReplaySnapshot,
        content: &[u8],
        now_ms: i64,
    ) -> Result<bool, KimiThinkingReplayError> {
        validate_content(content)?;
        self.state.replace_exact_if_unchanged(
            model_family,
            session_key,
            snapshot,
            vec![content.to_vec()],
            now_ms,
        )
    }

    pub fn store(
        &self,
        model_family: &str,
        session_key: &str,
        content: &[u8],
        now_ms: i64,
    ) -> Result<bool, KimiThinkingReplayError> {
        validate_content(content)?;
        for _ in 0..64 {
            let (_, snapshot, _) = self.state.read(model_family, session_key, now_ms)?;
            if self.state.replace_exact_if_unchanged(
                model_family,
                session_key,
                &snapshot,
                vec![content.to_vec()],
                now_ms,
            )? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn delete_if_unchanged(
        &self,
        model_family: &str,
        session_key: &str,
        snapshot: &KimiThinkingReplaySnapshot,
        now_ms: i64,
    ) -> Result<bool, KimiThinkingReplayError> {
        self.state
            .delete_if_unchanged(model_family, session_key, snapshot, now_ms)
    }

    pub fn clear(&self) {
        self.state.clear();
    }

    #[cfg(test)]
    pub(crate) fn test_with_limits(
        max_entries: usize,
        evict_batch: usize,
        max_total_bytes: usize,
    ) -> Self {
        Self::with_limits(max_entries, evict_batch, max_total_bytes)
    }

    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.state.entry_count()
    }

    #[cfg(test)]
    pub(crate) fn total_bytes(&self) -> usize {
        self.state.total_bytes()
    }
}

fn validate_content(content: &[u8]) -> Result<(), KimiThinkingReplayError> {
    if content.is_empty() {
        return Err(KimiThinkingReplayError::InvalidItems);
    }
    if content.len() > KIMI_THINKING_REPLAY_MAX_BYTES_PER_ENTRY {
        return Err(KimiThinkingReplayError::TooLarge);
    }
    let value = serde_json::from_slice::<Value>(content)
        .map_err(|_| KimiThinkingReplayError::InvalidItems)?;
    let blocks = value
        .as_array()
        .ok_or(KimiThinkingReplayError::InvalidItems)?;
    if blocks.is_empty() {
        return Err(KimiThinkingReplayError::InvalidItems);
    }
    if blocks.len() > KIMI_THINKING_REPLAY_MAX_BLOCKS_PER_ENTRY {
        return Err(KimiThinkingReplayError::TooLarge);
    }
    Ok(())
}
