// ref: internal/cache/kimi_thinking_replay_cache_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Barrier};
use std::thread;

use super::kimi_thinking_replay_cache::{
    KimiThinkingReplayCache, KIMI_THINKING_REPLAY_MAX_BYTES_PER_ENTRY, KIMI_THINKING_REPLAY_TTL_MS,
};
use super::AntigravityReasoningReplayError;

fn content(label: &str) -> Vec<u8> {
    format!(r#"[{{"type":"thinking","signature":"{label}"}}]"#).into_bytes()
}

#[test]
fn preserves_exact_content_and_conditional_delete_keeps_newer_state() {
    let cache = KimiThinkingReplayCache::new();
    assert!(cache.store("kimi-k2", "lane", &content("one"), 1).unwrap());
    let (original, stale, found) = cache.read("kimi-k2", "lane", 2).unwrap();
    assert!(found);
    assert_eq!(original, content("one"));
    assert!(cache.store("kimi-k2", "lane", &content("two"), 3).unwrap());
    assert!(!cache
        .delete_if_unchanged("kimi-k2", "lane", &stale, 4)
        .unwrap());
    assert_eq!(cache.read("kimi-k2", "lane", 5).unwrap().0, content("two"));
}

#[test]
fn tombstone_fences_a_concurrent_miss_and_aba() {
    let cache = KimiThinkingReplayCache::new();
    let (_, miss, found) = cache.read("kimi-k2", "lane", 1).unwrap();
    assert!(!found);
    let (_, current, _) = cache.read("kimi-k2", "lane", 2).unwrap();
    assert!(cache
        .delete_if_unchanged("kimi-k2", "lane", &current, 3)
        .unwrap());
    assert!(!cache
        .replace_if_unchanged("kimi-k2", "lane", &miss, &content("stale"), 4)
        .unwrap());
    assert!(cache.store("kimi-k2", "lane", &content("a"), 5).unwrap());
    let (_, a, _) = cache.read("kimi-k2", "lane", 6).unwrap();
    assert!(cache.store("kimi-k2", "lane", &content("b"), 7).unwrap());
    assert!(cache.store("kimi-k2", "lane", &content("a"), 8).unwrap());
    assert!(!cache
        .replace_if_unchanged("kimi-k2", "lane", &a, &content("stale"), 9)
        .unwrap());
}

#[test]
fn enforces_entry_count_and_aggregate_bytes() {
    let sample = content("123456789012345678901234567890");
    let cache = KimiThinkingReplayCache::test_with_limits(3, 1, sample.len() * 2);
    for index in 0..4 {
        assert!(cache
            .store("kimi-k2", &format!("lane-{index}"), &sample, index)
            .unwrap());
    }
    assert!(cache.entry_count() <= 2);
    assert!(cache.total_bytes() <= sample.len() * 2);
}

#[test]
fn rejects_invalid_empty_and_oversized_content() {
    let cache = KimiThinkingReplayCache::new();
    assert_eq!(
        cache.store("kimi-k2", "lane", b"{}", 1),
        Err(AntigravityReasoningReplayError::InvalidItems)
    );
    let oversized = vec![b' '; KIMI_THINKING_REPLAY_MAX_BYTES_PER_ENTRY + 1];
    assert_eq!(
        cache.store("kimi-k2", "lane", &oversized, 1),
        Err(AntigravityReasoningReplayError::TooLarge)
    );
}

#[test]
fn expiration_slides_on_read() {
    let cache = KimiThinkingReplayCache::new();
    cache.store("kimi-k2", "lane", &content("one"), 0).unwrap();
    assert!(
        cache
            .read("kimi-k2", "lane", KIMI_THINKING_REPLAY_TTL_MS)
            .unwrap()
            .2
    );
    assert!(
        cache
            .read("kimi-k2", "lane", KIMI_THINKING_REPLAY_TTL_MS * 2)
            .unwrap()
            .2
    );
    assert!(
        !cache
            .read("kimi-k2", "lane", KIMI_THINKING_REPLAY_TTL_MS * 3 + 1)
            .unwrap()
            .2
    );
}

#[test]
fn only_one_concurrent_writer_can_commit_one_snapshot() {
    let cache = Arc::new(KimiThinkingReplayCache::new());
    let (_, snapshot, _) = cache.read("kimi-k2", "lane", 1).unwrap();
    let barrier = Arc::new(Barrier::new(8));
    let joins = (0..8)
        .map(|index| {
            let cache = Arc::clone(&cache);
            let barrier = Arc::clone(&barrier);
            let snapshot = snapshot.clone();
            thread::spawn(move || {
                barrier.wait();
                cache
                    .replace_if_unchanged(
                        "kimi-k2",
                        "lane",
                        &snapshot,
                        &content(&index.to_string()),
                        2,
                    )
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        joins
            .into_iter()
            .map(|join| join.join().unwrap())
            .filter(|won| *won)
            .count(),
        1
    );
}
