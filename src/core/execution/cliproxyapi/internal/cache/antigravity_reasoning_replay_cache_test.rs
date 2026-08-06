// ref: internal/cache/antigravity_reasoning_replay_cache_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};

use super::*;

#[derive(Default)]
struct FakeStore {
    state: Mutex<FakeStoreState>,
}

#[derive(Default)]
struct FakeStoreState {
    values: HashMap<String, Vec<u8>>,
    expire_count: usize,
    cas_error: bool,
}

impl FakeStore {
    fn put(&self, key: String, value: Vec<u8>) {
        self.state.lock().unwrap().values.insert(key, value);
    }

    fn remove(&self, key: &str) {
        self.state.lock().unwrap().values.remove(key);
    }

    fn value(&self, key: &str) -> Vec<u8> {
        self.state.lock().unwrap().values[key].clone()
    }

    fn expire_count(&self) -> usize {
        self.state.lock().unwrap().expire_count
    }

    fn fail_cas(&self) {
        self.state.lock().unwrap().cas_error = true;
    }
}

impl AntigravityReasoningReplayStore for FakeStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, AntigravityReasoningReplayStoreError> {
        Ok(self.state.lock().unwrap().values.get(key).cloned())
    }

    fn set(
        &self,
        key: &str,
        value: &[u8],
        _ttl_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayStoreError> {
        self.state
            .lock()
            .unwrap()
            .values
            .insert(key.to_owned(), value.to_vec());
        Ok(true)
    }

    fn compare_and_swap(
        &self,
        key: &str,
        expected: Option<&[u8]>,
        value: &[u8],
        _ttl_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayStoreError> {
        let mut state = self.state.lock().unwrap();
        if state.cas_error {
            return Err(AntigravityReasoningReplayStoreError);
        }
        let matches = match (state.values.get(key), expected) {
            (None, None) => true,
            (Some(current), Some(expected)) => current.as_slice() == expected,
            _ => false,
        };
        if matches {
            state.values.insert(key.to_owned(), value.to_vec());
        }
        Ok(matches)
    }

    fn expire(
        &self,
        _key: &str,
        _ttl_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayStoreError> {
        self.state.lock().unwrap().expire_count += 1;
        Ok(true)
    }
}

fn item(signature: &str) -> Vec<u8> {
    format!(
        r#"{{"type":"thought_signature","contentIndex":1,"partIndex":0,"thoughtSignature":"{signature}"}}"#
    )
    .into_bytes()
}

fn store_cache() -> (Arc<FakeStore>, AntigravityReasoningReplayCache) {
    let store = Arc::new(FakeStore::default());
    let cache = AntigravityReasoningReplayCache::with_store(store.clone());
    (store, cache)
}

#[test]
fn conditional_mutation_rejects_stale_local_snapshot() {
    let cache = AntigravityReasoningReplayCache::new();
    cache
        .cache_items("gemini", "stale", &[item("old-local-signature-123456")], 1)
        .unwrap();
    let (_, stale, found) = cache.read("gemini", "stale", 2).unwrap();
    assert!(found);
    cache
        .cache_items("gemini", "stale", &[item("new-local-signature-123456")], 3)
        .unwrap();
    assert!(!cache
        .replace_if_unchanged(
            "gemini",
            "stale",
            &stale,
            &[item("stale-local-signature-123456")],
            4
        )
        .unwrap());
    assert!(!cache
        .delete_if_unchanged("gemini", "stale", &stale, 5)
        .unwrap());
}

#[test]
fn non_prefix_replace_rotates_local_branch() {
    let cache = AntigravityReasoningReplayCache::new();
    cache
        .cache_items("gemini", "branch", &[item("old-local-signature-123456")], 1)
        .unwrap();
    let (_, first, _) = cache.read("gemini", "branch", 2).unwrap();
    let (_, stale, _) = cache.read("gemini", "branch", 3).unwrap();
    let replacement = item("new-local-signature-123456");
    assert!(cache
        .replace_if_unchanged(
            "gemini",
            "branch",
            &first,
            std::slice::from_ref(&replacement),
            4
        )
        .unwrap());
    assert!(!cache
        .replace_if_unchanged(
            "gemini",
            "branch",
            &stale,
            &[replacement, item("latest-local-signature-123456")],
            5
        )
        .unwrap());
}

#[test]
fn descendant_replace_accepts_local_chain_and_rejects_reset_aba() {
    let cache = AntigravityReasoningReplayCache::new();
    let prefix = item("descendant-prefix-signature-123456");
    let middle = item("descendant-middle-signature-123456");
    cache
        .cache_items("gemini", "desc", std::slice::from_ref(&prefix), 1)
        .unwrap();
    let (_, stale, _) = cache.read("gemini", "desc", 2).unwrap();
    let (_, first, _) = cache.read("gemini", "desc", 3).unwrap();
    assert!(cache
        .replace_if_unchanged(
            "gemini",
            "desc",
            &first,
            &[prefix.clone(), middle.clone()],
            4
        )
        .unwrap());
    assert!(cache
        .replace_if_unchanged(
            "gemini",
            "desc",
            &stale,
            &[
                prefix.clone(),
                middle,
                item("descendant-latest-signature-123456")
            ],
            5
        )
        .unwrap());
    let (_, current, _) = cache.read("gemini", "desc", 6).unwrap();
    assert!(cache
        .delete_if_unchanged("gemini", "desc", &current, 7)
        .unwrap());
    let (_, reset, found) = cache.read("gemini", "desc", 8).unwrap();
    assert!(!found);
    assert!(cache
        .replace_if_unchanged("gemini", "desc", &reset, std::slice::from_ref(&prefix), 9)
        .unwrap());
    assert!(!cache
        .replace_if_unchanged(
            "gemini",
            "desc",
            &stale,
            &[prefix, item("reset-stale-signature-123456")],
            10
        )
        .unwrap());
}

#[test]
fn local_tombstone_fences_first_writer_across_eviction() {
    let cache = AntigravityReasoningReplayCache::with_limits(2, 1);
    let (_, stale, found) = cache.read("gemini", "absent", 1).unwrap();
    assert!(!found);
    let (_, clear, _) = cache.read("gemini", "absent", 2).unwrap();
    assert!(cache
        .delete_if_unchanged("gemini", "absent", &clear, 3)
        .unwrap());
    cache.evict_oldest(1);
    assert!(!cache
        .replace_if_unchanged(
            "gemini",
            "absent",
            &stale,
            &[item("stale-first-writer-123456")],
            4
        )
        .unwrap());
}

#[test]
fn unrelated_eviction_keeps_reserved_absent_snapshot_valid() {
    let cache = AntigravityReasoningReplayCache::with_limits(2, 1);
    cache
        .cache_items("gemini", "old", &[item("old-live-signature-123456")], 1)
        .unwrap();
    let (_, absent, found) = cache.read("gemini", "new", 2).unwrap();
    assert!(!found);
    cache.evict_oldest(1);
    assert!(cache
        .replace_if_unchanged(
            "gemini",
            "new",
            &absent,
            &[item("first-write-signature-123456")],
            3
        )
        .unwrap());
}

#[test]
fn local_entries_and_absence_reservations_remain_bounded() {
    let cache = AntigravityReasoningReplayCache::with_limits(4, 2);
    for index in 0..10 {
        cache
            .delete("gemini", &format!("tombstone-{index}"), index)
            .unwrap();
    }
    assert!(cache.entry_count() <= 4);
    for index in 0..10 {
        let (_, _, found) = cache
            .read("gemini", &format!("absent-{index}"), 20 + index)
            .unwrap();
        assert!(!found);
    }
    assert!(cache.entry_count() <= 4);
    let (_, _, newest_found) = cache.read("gemini", "absent-9", 40).unwrap();
    assert!(!newest_found);
}

#[test]
fn home_absent_snapshot_is_fenced() {
    let (store, cache) = store_cache();
    let (_, snapshot, found) = cache.read("gemini", "absent", 1).unwrap();
    assert!(!found);
    assert!(snapshot.found && !snapshot.raw.is_empty());
    store.remove(&home_store_key("gemini", "absent"));
    assert!(!cache
        .replace_if_unchanged(
            "gemini",
            "absent",
            &snapshot,
            &[item("stale-home-signature-123456")],
            2
        )
        .unwrap());
}

#[test]
fn conditional_mutation_rejects_stale_home_snapshot_and_value_aba() {
    let (_store, cache) = store_cache();
    let a = item("home-aba-signature-a-123456");
    let b = item("home-aba-signature-b-123456");
    cache
        .cache_items("gemini", "aba", std::slice::from_ref(&a), 1)
        .unwrap();
    let (_, stale, _) = cache.read("gemini", "aba", 2).unwrap();
    cache
        .cache_items("gemini", "aba", std::slice::from_ref(&b), 3)
        .unwrap();
    cache
        .cache_items("gemini", "aba", std::slice::from_ref(&a), 4)
        .unwrap();
    assert!(!cache
        .replace_if_unchanged("gemini", "aba", &stale, std::slice::from_ref(&b), 5)
        .unwrap());
    assert!(!cache
        .delete_if_unchanged("gemini", "aba", &stale, 6)
        .unwrap());
}

#[test]
fn non_prefix_rotates_home_branch_while_descendant_retry_succeeds() {
    let (_store, cache) = store_cache();
    let prefix = item("home-prefix-signature-123456");
    let middle = item("home-middle-signature-123456");
    cache
        .cache_items("gemini", "home-desc", std::slice::from_ref(&prefix), 1)
        .unwrap();
    let (_, stale, _) = cache.read("gemini", "home-desc", 2).unwrap();
    let (_, first, _) = cache.read("gemini", "home-desc", 3).unwrap();
    assert!(cache
        .replace_if_unchanged(
            "gemini",
            "home-desc",
            &first,
            &[prefix.clone(), middle.clone()],
            4
        )
        .unwrap());
    assert!(cache
        .replace_if_unchanged(
            "gemini",
            "home-desc",
            &stale,
            &[prefix, middle, item("home-latest-signature-123456")],
            5
        )
        .unwrap());

    cache
        .cache_items(
            "gemini",
            "home-reset",
            &[item("home-old-signature-123456")],
            6,
        )
        .unwrap();
    let (_, stale_reset, _) = cache.read("gemini", "home-reset", 7).unwrap();
    let (_, first_reset, _) = cache.read("gemini", "home-reset", 8).unwrap();
    let reset = item("home-new-signature-123456");
    assert!(cache
        .replace_if_unchanged(
            "gemini",
            "home-reset",
            &first_reset,
            std::slice::from_ref(&reset),
            9
        )
        .unwrap());
    assert!(!cache
        .replace_if_unchanged(
            "gemini",
            "home-reset",
            &stale_reset,
            &[reset, item("home-sibling-signature-123456")],
            10
        )
        .unwrap());
}

#[test]
fn home_cas_errors_are_reported() {
    let (store, cache) = store_cache();
    store.fail_cas();
    assert_eq!(
        cache.read("gemini", "cas-error", 1).unwrap_err(),
        AntigravityReasoningReplayError::InvalidSnapshot
    );
    let snapshot = AntigravityReasoningReplaySnapshot {
        loaded: true,
        ..Default::default()
    };
    assert_eq!(
        cache
            .replace_if_unchanged(
                "gemini",
                "cas-error",
                &snapshot,
                &[item("home-cas-signature-123456")],
                2
            )
            .unwrap_err(),
        AntigravityReasoningReplayError::InvalidSnapshot
    );
    assert_eq!(
        cache
            .delete_if_unchanged("gemini", "cas-error", &snapshot, 3)
            .unwrap_err(),
        AntigravityReasoningReplayError::InvalidSnapshot
    );
}

#[test]
fn home_cas_retry_rejects_oversized_serialized_value() {
    let (store, cache) = store_cache();
    let prefix = item("oversized-home-prefix-123456");
    cache
        .cache_items("gemini", "oversized", std::slice::from_ref(&prefix), 1)
        .unwrap();
    let (_, snapshot, _) = cache.read("gemini", "oversized", 2).unwrap();
    let oversized = vec![b' '; ANTIGRAVITY_REPLAY_MAX_SERIALIZED_BYTES + 1];
    store.put(home_store_key("gemini", "oversized"), oversized.clone());
    assert!(!cache
        .replace_if_unchanged(
            "gemini",
            "oversized",
            &snapshot,
            &[prefix, item("oversized-home-latest-123456")],
            3
        )
        .unwrap());
    assert_eq!(
        store.value(&home_store_key("gemini", "oversized")).len(),
        oversized.len()
    );
}

#[test]
fn home_writes_remain_legacy_array_readable() {
    let (store, cache) = store_cache();
    cache
        .cache_items(
            "gemini",
            "legacy",
            &[item("legacy-readable-signature-123456")],
            1,
        )
        .unwrap();
    let raw = store.value(&home_store_key("gemini", "legacy"));
    let encoded: Vec<String> = serde_json::from_slice(&raw).unwrap();
    let decoded = encoded
        .iter()
        .map(|item| BASE64_STANDARD.decode(item).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(decoded.len(), 2);
    assert_eq!(
        serde_json::from_slice::<Value>(&decoded[0]).unwrap()["type"],
        GENERATION_ITEM_TYPE
    );
    assert!(decoded[1]
        .windows(b"legacy-readable".len())
        .any(|window| window == b"legacy-readable"));
}

#[test]
fn home_read_normalizes_valid_legacy_value_and_rejects_mixed_chain() {
    let (store, cache) = store_cache();
    let key = home_store_key("gemini", "validation");
    let valid = br#"{"type":"function_call_part","name":"run","args":{"b":2,"a":1},"targetOccurrence":1,"thoughtSignature":"valid-home-signature-123456"}"#.to_vec();
    let legacy = serde_json::to_vec(&vec![BASE64_STANDARD.encode(&valid)]).unwrap();
    store.put(key.clone(), legacy);
    let (items, _, found) = cache.read("gemini", "validation", 1).unwrap();
    assert!(found && items.len() == 1);
    assert_eq!(store.expire_count(), 1);

    let mixed = serde_json::to_vec(&vec![
        BASE64_STANDARD.encode(&valid),
        BASE64_STANDARD.encode(br#"{"type":"unknown"}"#),
    ])
    .unwrap();
    store.put(key, mixed);
    let (_, _, found) = cache.read("gemini", "validation", 2).unwrap();
    assert!(!found);
    assert_eq!(store.expire_count(), 1);
}

#[test]
fn concurrent_siblings_publish_at_most_one_branch() {
    let cache = Arc::new(AntigravityReasoningReplayCache::new());
    let prefix = item("concurrent-prefix-signature-123456");
    cache
        .cache_items("gemini", "concurrent", std::slice::from_ref(&prefix), 1)
        .unwrap();
    let (_, snapshot, _) = cache.read("gemini", "concurrent", 2).unwrap();
    let barrier = Arc::new(Barrier::new(8));
    let handles = (0..8)
        .map(|index| {
            let cache = Arc::clone(&cache);
            let barrier = Arc::clone(&barrier);
            let snapshot = snapshot.clone();
            let prefix = prefix.clone();
            thread::spawn(move || {
                barrier.wait();
                cache
                    .replace_if_unchanged(
                        "gemini",
                        "concurrent",
                        &snapshot,
                        &[prefix, item(&format!("concurrent-sibling-{index}-123456"))],
                        3,
                    )
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    let successes = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .filter(|success| *success)
        .count();
    assert_eq!(successes, 1);
}
