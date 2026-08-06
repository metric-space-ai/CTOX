// ref: internal/cache/xai_reasoning_replay_cache_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Barrier};
use std::thread;

use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine as _};
use serde_json::Value;

use super::xai_reasoning_replay_cache::{
    XaiReasoningReplayCache, XaiReasoningReplayStoreStatus, XAI_REASONING_REPLAY_TTL_MS,
};

fn grok_encrypted_content() -> String {
    STANDARD_NO_PAD.encode((0_u16..256).map(|value| value as u8).collect::<Vec<_>>())
}

fn reasoning() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "type":"reasoning", "encrypted_content":grok_encrypted_content(), "leak":"drop"
    }))
    .unwrap()
}

#[test]
fn rejects_codex_shape_but_stores_valid_grok() {
    let cache = XaiReasoningReplayCache::new();
    let codex = br#"{"type":"reasoning","encrypted_content":"gAAAA-invalid-codex"}"#.to_vec();
    assert_eq!(
        cache.store("grok-4", "lane-a", &[codex], 1),
        XaiReasoningReplayStoreStatus::NoReplayableState
    );
    assert_eq!(
        cache.store("grok-4", "lane-b", &[reasoning()], 1),
        XaiReasoningReplayStoreStatus::Stored
    );
    let (items, _, found) = cache.read("grok-4", "lane-b", 2).unwrap();
    assert!(found);
    let value: Value = serde_json::from_slice(&items[0]).unwrap();
    assert_eq!(value["summary"], serde_json::json!([]));
    assert_eq!(value["content"], Value::Null);
    assert!(value.get("leak").is_none());
}

#[test]
fn retains_only_safe_assistant_message_parts_with_an_anchor() {
    let cache = XaiReasoningReplayCache::new();
    let message = br#"{"type":"message","role":" Assistant ","content":[{"type":"output_text","text":"ok","secret":1},{"type":"refusal","refusal":"no"},{"type":"input_text","text":"drop"}]}"#.to_vec();
    assert_eq!(
        cache.store("grok-4", "lane", &[reasoning(), message], 1),
        XaiReasoningReplayStoreStatus::Stored
    );
    let items = cache.read("grok-4", "lane", 2).unwrap().0;
    let message: Value = serde_json::from_slice(&items[1]).unwrap();
    assert_eq!(message["content"].as_array().unwrap().len(), 2);
    assert!(message["content"][0].get("secret").is_none());
}

#[test]
fn message_only_is_not_replayable() {
    let cache = XaiReasoningReplayCache::new();
    let message =
        br#"{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}"#
            .to_vec();
    assert_eq!(
        cache.store("grok-4", "lane", &[message], 1),
        XaiReasoningReplayStoreStatus::NoReplayableState
    );
}

#[test]
fn function_and_custom_tool_calls_are_replay_anchors() {
    let cache = XaiReasoningReplayCache::new();
    let function =
        br#"{"type":"function_call","call_id":" id ","name":" run ","arguments":"{}","secret":1}"#
            .to_vec();
    let custom =
        br#"{"type":"custom_tool_call","call_id":" c ","name":" shell ","input":{"x":1}}"#.to_vec();
    assert_eq!(
        cache.store("grok-4", "lane", &[function, custom], 1),
        XaiReasoningReplayStoreStatus::Stored
    );
    let items = cache.read("grok-4", "lane", 2).unwrap().0;
    let first: Value = serde_json::from_slice(&items[0]).unwrap();
    let second: Value = serde_json::from_slice(&items[1]).unwrap();
    assert_eq!(first["call_id"], "id");
    assert_eq!(second["status"], "completed");
    assert_eq!(second["input"]["x"], 1);
}

#[test]
fn stale_snapshot_cannot_replace_or_delete_across_tombstone() {
    let cache = XaiReasoningReplayCache::new();
    let (_, stale, _) = cache.read("grok-4", "lane", 1).unwrap();
    let (_, current, _) = cache.read("grok-4", "lane", 2).unwrap();
    assert!(cache
        .delete_if_unchanged("grok-4", "lane", &current, 3)
        .unwrap());
    assert!(!cache
        .replace_if_unchanged("grok-4", "lane", &stale, &[reasoning()], 4)
        .unwrap());
    assert!(!cache
        .delete_if_unchanged("grok-4", "lane", &stale, 5)
        .unwrap());
}

#[test]
fn ttl_capacity_and_clear_are_bounded() {
    let cache = XaiReasoningReplayCache::test_with_limits(2, 1);
    for index in 0..3 {
        assert_eq!(
            cache.store("grok-4", &format!("lane-{index}"), &[reasoning()], index),
            XaiReasoningReplayStoreStatus::Stored
        );
    }
    assert_eq!(cache.entry_count(), 2);
    assert!(
        !cache
            .read("grok-4", "lane-1", XAI_REASONING_REPLAY_TTL_MS + 5)
            .unwrap()
            .2
    );
    cache.clear();
    assert_eq!(cache.entry_count(), 0);
}

#[test]
fn one_snapshot_has_one_concurrent_winner() {
    let cache = Arc::new(XaiReasoningReplayCache::new());
    let (_, snapshot, _) = cache.read("grok-4", "lane", 1).unwrap();
    let barrier = Arc::new(Barrier::new(8));
    let joins = (0..8)
        .map(|_| {
            let cache = Arc::clone(&cache);
            let barrier = Arc::clone(&barrier);
            let snapshot = snapshot.clone();
            thread::spawn(move || {
                barrier.wait();
                cache
                    .replace_if_unchanged("grok-4", "lane", &snapshot, &[reasoning()], 2)
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
