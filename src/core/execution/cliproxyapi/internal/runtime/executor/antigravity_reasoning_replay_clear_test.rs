// ref: internal/runtime/executor/antigravity_reasoning_replay_clear_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::antigravity_reasoning_replay::{
    is_invalid_antigravity_signature_error, prepare_antigravity_reasoning_replay,
};
use crate::internal::cache::antigravity_reasoning_replay_cache::AntigravityReasoningReplayCache;
use std::sync::Arc;

#[test]
fn clears_exact_lane_only_on_invalid_signature_400() {
    let cache = Arc::new(AntigravityReasoningReplayCache::new());
    let payload = br#"{"request":{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}}"#;
    let (_, accumulator) =
        prepare_antigravity_reasoning_replay(cache.clone(), "gemini-3", "session", payload, 1)
            .unwrap();
    assert!(!accumulator
        .clear_on_invalid_signature(500, b"invalid thought signature", 2)
        .unwrap());
    assert!(!accumulator
        .clear_on_invalid_signature(400, b"other bad request", 2)
        .unwrap());
    assert!(accumulator
        .clear_on_invalid_signature(400, b"Thought signature is invalid", 2)
        .unwrap());
    let (_, _, found) = cache.read("gemini-3", "session", 3).unwrap();
    assert!(!found);
    assert!(is_invalid_antigravity_signature_error(
        b"thoughtSignature mismatch"
    ));
}
