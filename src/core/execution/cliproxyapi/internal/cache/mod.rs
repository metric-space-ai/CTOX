// Origin: CTOX
// License: AGPL-3.0-only

pub mod antigravity_reasoning_replay_cache;
pub mod bounded_lru;
pub mod codex_reasoning_replay_cache;
pub mod kimi_thinking_replay_cache;
pub mod signature_cache;
pub mod xai_reasoning_replay_cache;

#[cfg(test)]
mod bounded_lru_test;
#[cfg(test)]
mod codex_reasoning_replay_cache_test;
#[cfg(test)]
mod kimi_thinking_replay_cache_test;
#[cfg(test)]
mod signature_cache_test;
#[cfg(test)]
mod xai_reasoning_replay_cache_test;

pub use antigravity_reasoning_replay_cache::{
    AntigravityReasoningReplayCache, AntigravityReasoningReplayError,
    AntigravityReasoningReplaySnapshot, ANTIGRAVITY_REPLAY_MAX_BYTES_PER_ENTRY,
    ANTIGRAVITY_REPLAY_MAX_ITEMS_PER_ENTRY, ANTIGRAVITY_REPLAY_TTL_MS,
};
pub use bounded_lru::BoundedLru;
pub use codex_reasoning_replay_cache::{
    CodexReasoningReplayCache, CodexReasoningReplayError, CodexReasoningReplaySnapshot,
    CODEX_REASONING_REPLAY_EVICT_BATCH_SIZE, CODEX_REASONING_REPLAY_MAX_BYTES_PER_ENTRY,
    CODEX_REASONING_REPLAY_MAX_ENTRIES, CODEX_REASONING_REPLAY_MAX_TURNS_PER_ENTRY,
    CODEX_REASONING_REPLAY_TTL_MS, CODEX_REASONING_REPLAY_TURN_TYPE,
};
pub use kimi_thinking_replay_cache::{
    KimiThinkingReplayCache, KimiThinkingReplayError, KimiThinkingReplaySnapshot,
    KIMI_THINKING_REPLAY_EVICT_BATCH_SIZE, KIMI_THINKING_REPLAY_MAX_BLOCKS_PER_ENTRY,
    KIMI_THINKING_REPLAY_MAX_BYTES_PER_ENTRY, KIMI_THINKING_REPLAY_MAX_ENTRIES,
    KIMI_THINKING_REPLAY_MAX_TOTAL_BYTES, KIMI_THINKING_REPLAY_TTL_MS,
};
#[cfg(test)]
pub(crate) use signature_cache::signature_cache_test_guard;
pub use signature_cache::{
    cache_signature, cache_signature_best_effort, clear_signature_cache, delete_cached_signature,
    delete_cached_signature_required, get_cached_signature, get_cached_signature_required,
    get_model_group, has_valid_signature, set_signature_bypass_strict_mode,
    set_signature_cache_enabled, signature_bypass_strict_mode, signature_cache_enabled,
    signature_kv_key, SignatureCacheStoreError, SignatureKvStore,
};
pub use xai_reasoning_replay_cache::{
    XaiReasoningReplayCache, XaiReasoningReplayError, XaiReasoningReplaySnapshot,
    XaiReasoningReplayStoreStatus, XAI_REASONING_REPLAY_EVICT_BATCH_SIZE,
    XAI_REASONING_REPLAY_MAX_ENTRIES, XAI_REASONING_REPLAY_TTL_MS,
};
