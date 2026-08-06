// ref: internal/cache/codex_reasoning_replay_cache_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;
use std::thread;

use base64::{engine::general_purpose, Engine as _};

use super::codex_reasoning_replay_cache::{
    CodexReasoningReplayCache, CodexReasoningReplayError, CodexReasoningReplaySnapshot,
    CODEX_REASONING_REPLAY_MAX_BYTES_PER_ENTRY, CODEX_REASONING_REPLAY_MAX_TURNS_PER_ENTRY,
    CODEX_REASONING_REPLAY_TTL_MS, CODEX_REASONING_REPLAY_TURN_TYPE,
};

fn signature(seed: u8) -> String {
    let mut payload = vec![0_u8; 1 + 8 + 16 + 16 + 32];
    payload[0] = 0x80;
    for (index, byte) in payload.iter_mut().enumerate().skip(9) {
        *byte = seed.wrapping_add(index as u8);
    }
    general_purpose::URL_SAFE_NO_PAD.encode(payload)
}

fn reasoning(seed: u8) -> Vec<u8> {
    format!(
        "{{\"type\":\"reasoning\",\"summary\":[{{\"text\":\"drop\"}}],\"content\":[1],\"encrypted_content\":{},\"ignored\":true}}",
        serde_json::to_string(&signature(seed)).unwrap()
    )
    .into_bytes()
}

fn turn(id: &str) -> Vec<Vec<u8>> {
    vec![
        format!(
            "{{\"type\":\"{CODEX_REASONING_REPLAY_TURN_TYPE}\",\"id\":{},\"assistant_fingerprint\":\" answer \"}}",
            serde_json::to_string(id).unwrap()
        )
        .into_bytes(),
        reasoning(id.bytes().fold(0_u8, u8::wrapping_add)),
    ]
}

#[test]
fn rejects_invalid_keys_snapshots_and_reasoning_signatures() {
    let cache = CodexReasoningReplayCache::new();
    assert_eq!(
        cache.read("", "session", 1),
        Err(CodexReasoningReplayError::InvalidKey)
    );
    let invalid = vec![br#"{"type":"reasoning","encrypted_content":"bad"}"#.to_vec()];
    let (_, snapshot, _) = cache.read("gpt-5.4", "session", 1).unwrap();
    assert_eq!(
        cache.replace_if_unchanged("gpt-5.4", "session", &snapshot, &invalid, 2),
        Err(CodexReasoningReplayError::InvalidItems)
    );
    assert_eq!(
        cache.replace_if_unchanged(
            "gpt-5.4",
            "session",
            &CodexReasoningReplaySnapshot::default(),
            &turn("turn"),
            2,
        ),
        Err(CodexReasoningReplayError::InvalidSnapshot)
    );
}

#[test]
fn canonicalizes_all_replay_item_shapes_byte_exactly() {
    let cache = CodexReasoningReplayCache::new();
    let (_, snapshot, _) = cache.read("gpt-5.4", "session", 1).unwrap();
    let encrypted = signature(3);
    let items = vec![
        format!(
            "{{\"type\":\"{CODEX_REASONING_REPLAY_TURN_TYPE}\",\"id\":\" turn-1 \",\"assistant_fingerprint\":\" answer \",\"request_fingerprint\":\" request \",\"call_ids\":[\" call-1 \",\"\"]}}"
        ).into_bytes(),
        format!("{{\"type\":\"reasoning\",\"encrypted_content\":\"{encrypted}\",\"summary\":[1]}}").into_bytes(),
        br#"{"type":"function_call","call_id":" call-1 ","name":" tool ","arguments":"{\"x\":1}","status":"drop"}"#.to_vec(),
        br#"{"type":"custom_tool_call","status":" done ","call_id":" custom-1 ","name":" shell ","input":{"z":1.2300,"a":900719925474099312345},"ignored":true}"#.to_vec(),
    ];
    assert!(cache
        .replace_if_unchanged("gpt-5.4", "session", &snapshot, &items, 2)
        .unwrap());
    let (stored, _, found) = cache.read("gpt-5.4", "session", 3).unwrap();
    assert!(found);
    assert_eq!(
        stored[0],
        br#"{"type":"cpa_codex_replay_turn","id":"turn-1","assistant_fingerprint":"answer","request_fingerprint":"request","call_ids":["call-1"]}"#
    );
    assert_eq!(
        stored[1],
        format!("{{\"type\":\"reasoning\",\"summary\":[],\"content\":null,\"encrypted_content\":\"{encrypted}\"}}").as_bytes()
    );
    assert_eq!(
        stored[2],
        br#"{"type":"function_call","call_id":"call-1","name":"tool","arguments":"{\"x\":1}"}"#
    );
    assert_eq!(stored[3], br#"{"type":"custom_tool_call","status":"done","call_id":"custom-1","name":"shell","input":{"z":1.2300,"a":900719925474099312345}}"#);
}

#[test]
fn keys_are_scoped_by_model_and_session_and_ttl_is_sliding() {
    let cache = CodexReasoningReplayCache::new();
    let (_, snapshot, _) = cache.read(" gpt-5.4 ", " session-a ", 0).unwrap();
    assert!(cache
        .replace_if_unchanged("gpt-5.4", "session-a", &snapshot, &turn("one"), 1)
        .unwrap());
    assert!(!cache.read("gpt-5.5", "session-a", 2).unwrap().2);
    assert!(!cache.read("gpt-5.4", "session-b", 2).unwrap().2);
    assert!(
        cache
            .read("gpt-5.4", "session-a", CODEX_REASONING_REPLAY_TTL_MS)
            .unwrap()
            .2
    );
    assert!(
        cache
            .read("gpt-5.4", "session-a", CODEX_REASONING_REPLAY_TTL_MS * 2)
            .unwrap()
            .2
    );
    assert!(
        !cache
            .read(
                "gpt-5.4",
                "session-a",
                CODEX_REASONING_REPLAY_TTL_MS * 3 + 1
            )
            .unwrap()
            .2
    );
}

#[test]
fn stale_writer_cannot_cross_delete_or_clear_tombstones() {
    let cache = CodexReasoningReplayCache::new();
    let (_, stale, _) = cache.read("gpt-5.4", "session", 1).unwrap();
    let (_, delete, _) = cache.read("gpt-5.4", "session", 2).unwrap();
    assert!(cache
        .delete_if_unchanged("gpt-5.4", "session", &delete, 3)
        .unwrap());
    assert!(!cache
        .replace_if_unchanged("gpt-5.4", "session", &stale, &turn("stale"), 4)
        .unwrap());

    let (_, before_clear, _) = cache.read("gpt-5.4", "session", 5).unwrap();
    cache.clear();
    let _ = cache.read("gpt-5.4", "session", 6).unwrap();
    assert!(!cache
        .replace_if_unchanged("gpt-5.4", "session", &before_clear, &turn("resurrect"), 7,)
        .unwrap());
}

#[test]
fn descendant_extension_wins_and_stale_sibling_is_rejected() {
    let cache = CodexReasoningReplayCache::new();
    let (_, empty, _) = cache.read("gpt-5.4", "chain", 1).unwrap();
    let first = turn("first");
    assert!(cache
        .replace_if_unchanged("gpt-5.4", "chain", &empty, &first, 2)
        .unwrap());
    let (one, parent, _) = cache.read("gpt-5.4", "chain", 3).unwrap();
    let mut descendant = one.clone();
    descendant.extend(turn("second"));
    assert!(cache
        .replace_if_unchanged("gpt-5.4", "chain", &parent, &descendant, 4)
        .unwrap());
    let mut sibling = one;
    sibling.extend(turn("sibling"));
    assert!(!cache
        .replace_if_unchanged("gpt-5.4", "chain", &parent, &sibling, 5)
        .unwrap());
}

#[test]
fn append_accumulates_turns_and_deduplicates_turn_ids() {
    let cache = CodexReasoningReplayCache::new();
    assert!(cache
        .append_turn("gpt-5.4", "chain", &turn("one"), 1)
        .unwrap());
    assert!(cache
        .append_turn("gpt-5.4", "chain", &turn("two"), 2)
        .unwrap());
    assert!(cache
        .append_turn("gpt-5.4", "chain", &turn("two"), 3)
        .unwrap());
    let (items, _, found) = cache.read("gpt-5.4", "chain", 4).unwrap();
    assert!(found);
    assert_eq!(items.len(), 4);
    assert_eq!(
        gjson::get(std::str::from_utf8(&items[0]).unwrap(), "id").str(),
        "one"
    );
    assert_eq!(
        gjson::get(std::str::from_utf8(&items[2]).unwrap(), "id").str(),
        "two"
    );
}

#[test]
fn concurrent_appends_preserve_every_successful_turn() {
    let cache = Arc::new(CodexReasoningReplayCache::new());
    let workers = (0..16)
        .map(|index| {
            let cache = Arc::clone(&cache);
            thread::spawn(move || {
                cache
                    .append_turn(
                        "gpt-5.4",
                        "concurrent",
                        &turn(&format!("turn-{index}")),
                        index,
                    )
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    assert!(workers.into_iter().all(|worker| worker.join().unwrap()));
    let (items, _, found) = cache.read("gpt-5.4", "concurrent", 100).unwrap();
    assert!(found);
    assert_eq!(items.len(), 32);
}

#[test]
fn turn_and_byte_overflow_drop_whole_oldest_turns() {
    let cache = CodexReasoningReplayCache::new();
    for index in 0..=CODEX_REASONING_REPLAY_MAX_TURNS_PER_ENTRY {
        assert!(cache
            .append_turn(
                "gpt-5.4",
                "bounded",
                &turn(&format!("turn-{index}")),
                index as i64
            )
            .unwrap());
    }
    let (items, _, _) = cache.read("gpt-5.4", "bounded", 999).unwrap();
    assert_eq!(items.len(), CODEX_REASONING_REPLAY_MAX_TURNS_PER_ENTRY * 2);
    assert_eq!(
        gjson::get(std::str::from_utf8(&items[0]).unwrap(), "id").str(),
        "turn-1"
    );

    let huge = vec![format!(
        "{{\"type\":\"custom_tool_call\",\"call_id\":\"x\",\"name\":\"tool\",\"input\":\"{}\"}}",
        "x".repeat(CODEX_REASONING_REPLAY_MAX_BYTES_PER_ENTRY)
    )
    .into_bytes()];
    let (_, snapshot, _) = cache.read("gpt-5.4", "huge", 1).unwrap();
    assert_eq!(
        cache.replace_if_unchanged("gpt-5.4", "huge", &snapshot, &huge, 2),
        Err(CodexReasoningReplayError::InvalidItems)
    );
}

#[test]
fn batch_capacity_eviction_removes_oldest_unrelated_entries() {
    let cache = CodexReasoningReplayCache::with_limits(3, 2);
    for index in 0..4 {
        assert!(cache
            .append_turn("gpt-5.4", &format!("session-{index}"), &turn("turn"), index)
            .unwrap());
    }
    assert!(cache.entry_count() < 3);
    assert!(!cache.read("gpt-5.4", "session-0", 10).unwrap().2);
    let rendered = format!("{cache:?}");
    assert!(!rendered.contains("session-0"));
}
