// ref: internal/runtime/executor/codex_executor_reasoning_replay_cache_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_executor_reasoning::{CodexReasoningReplayCache, CodexReasoningReplayScope};

#[test]
fn completed_reasoning_is_scoped_bounded_and_replayed() {
    let cache = CodexReasoningReplayCache::new(1, 1);
    let scope = CodexReasoningReplayScope::new("codex", "a").unwrap();
    assert!(cache.commit_completed(
        scope.clone(),
        br#"{"output":[{"type":"reasoning","id":"r1","encrypted_content":"sig"}]}"#
    ));
    let body = cache.apply(&scope, br#"{"input":[{"role":"user","content":"next"}]}"#);
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["input"][0]["id"], "r1");
}
