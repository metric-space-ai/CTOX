// ref: internal/auth/claude/identity_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Candidate delta evidence: internal/auth/claude/identity_test.go
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{json, Value};

use super::identity::{
    ensure_device_id_pool, generate_device_id_pool, has_canonical_device_id_pool,
    normalize_device_id_pool, read_device_id_pool, read_metadata_string, select_device_id,
    store_device_id_pool, store_metadata_string, store_metadata_value, valid_device_id,
    ClaudeIdentityError, CLAUDE_DEVICE_IDS_METADATA_KEY, CLAUDE_DEVICE_POOL_SIZE,
};

#[test]
fn generate_device_id_pool_has_native_shape() {
    let device_ids = generate_device_id_pool().unwrap();
    assert_eq!(device_ids.len(), CLAUDE_DEVICE_POOL_SIZE);
    assert!(valid_device_id(&device_ids[0]));
}

#[test]
fn read_and_store_use_defensive_owned_values() {
    let mut metadata = BTreeMap::new();
    let mut input = vec!["device-a".to_owned(), "device-b".to_owned()];
    store_device_id_pool(&mut metadata, &input);
    input[0] = "mutated-input".to_owned();

    let mut read = read_device_id_pool(&metadata).unwrap();
    read[0] = Value::String("hijacked".to_owned());
    assert_eq!(metadata[CLAUDE_DEVICE_IDS_METADATA_KEY][0], "device-a");
}

#[test]
fn ensure_repairs_and_stabilizes_metadata() {
    let first = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let mut metadata = BTreeMap::from([(
        CLAUDE_DEVICE_IDS_METADATA_KEY.to_owned(),
        json!([first, first, "INVALID"]),
    )]);

    let (device_ids, changed) = ensure_device_id_pool(&mut metadata).unwrap();
    assert!(changed);
    assert_eq!(device_ids, vec![first.to_owned()]);
    assert!(has_canonical_device_id_pool(
        metadata.get(CLAUDE_DEVICE_IDS_METADATA_KEY)
    ));

    let (again, changed_again) = ensure_device_id_pool(&mut metadata).unwrap();
    assert!(!changed_again);
    assert_eq!(again, device_ids);
}

#[test]
fn ensure_canonicalizes_one_device_and_migrates_five_slots() {
    let canonical = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let mut metadata = BTreeMap::from([(
        CLAUDE_DEVICE_IDS_METADATA_KEY.to_owned(),
        json!(["  AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA  "]),
    )]);
    let (device_ids, changed) = ensure_device_id_pool(&mut metadata).unwrap();
    assert!(changed);
    assert_eq!(device_ids, vec![canonical.to_owned()]);

    let first = "0000000000000000000000000000000000000000000000000000000000000000";
    metadata.insert(
        CLAUDE_DEVICE_IDS_METADATA_KEY.to_owned(),
        json!([
            first,
            "1111111111111111111111111111111111111111111111111111111111111111",
            "2222222222222222222222222222222222222222222222222222222222222222",
            "3333333333333333333333333333333333333333333333333333333333333333",
            "4444444444444444444444444444444444444444444444444444444444444444"
        ]),
    );
    let (migrated, changed) = ensure_device_id_pool(&mut metadata).unwrap();
    assert!(changed);
    assert_eq!(migrated, vec![first.to_owned()]);
}

#[test]
fn normalization_matches_upstream_lenient_and_strict_paths() {
    let first = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let raw = json!([7, " invalid ", format!("  {}  ", first.to_uppercase())]);
    assert_eq!(normalize_device_id_pool(Some(&raw)), vec![first.to_owned()]);
    assert!(!has_canonical_device_id_pool(Some(&raw)));
    assert!(!valid_device_id(&first.to_uppercase()));
}

#[test]
fn concurrent_initialization_uses_the_credential_owner_lock() {
    let metadata = Arc::new(Mutex::new(BTreeMap::new()));
    let mut workers = Vec::new();
    for _ in 0..20 {
        let metadata = Arc::clone(&metadata);
        workers.push(thread::spawn(move || {
            let mut metadata = metadata
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            ensure_device_id_pool(&mut metadata).unwrap().0
        }));
    }
    let results: Vec<Vec<String>> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect();
    let metadata = metadata
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let stored = normalize_device_id_pool(metadata.get(CLAUDE_DEVICE_IDS_METADATA_KEY));
    assert_eq!(stored.len(), CLAUDE_DEVICE_POOL_SIZE);
    assert!(results.iter().all(|result| result == &stored));
}

#[test]
fn selection_uses_one_device_across_sessions_and_rejects_empty_inputs() {
    let first = "0000000000000000000000000000000000000000000000000000000000000000";
    let pool = vec![first.to_owned()];
    assert_eq!(select_device_id(&pool, "session-one").unwrap(), first);
    assert_eq!(select_device_id(&pool, "session-two").unwrap(), first);
    assert_eq!(
        select_device_id(&pool, " "),
        Err(ClaudeIdentityError::EmptySessionId)
    );
    assert_eq!(
        select_device_id(&[], "session"),
        Err(ClaudeIdentityError::DevicePoolSize { actual: 0 })
    );
}

#[test]
fn metadata_helpers_preserve_resolved_values_on_empty_updates() {
    let mut metadata = BTreeMap::new();
    assert!(store_metadata_string(
        &mut metadata,
        "account_uuid",
        "account-a"
    ));
    assert!(!store_metadata_string(&mut metadata, "account_uuid", "  "));
    assert_eq!(
        read_metadata_string(&metadata, "account_uuid"),
        Some("account-a")
    );
    store_metadata_value(&mut metadata, "roles", json!(["claude_cli"]));
    assert_eq!(metadata["roles"], json!(["claude_cli"]));
}
