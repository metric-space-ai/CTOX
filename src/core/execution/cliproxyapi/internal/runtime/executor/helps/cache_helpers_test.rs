// ref: internal/runtime/executor/helps/cache_helpers_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use super::cache_helpers::*;
use super::{ClaudeIdentityKvStore, ClaudeIdentityStoreError};

#[derive(Default)]
struct FakeStore {
    values: Mutex<HashMap<String, Vec<u8>>>,
    reject_write: bool,
}

impl ClaudeIdentityKvStore for FakeStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, ClaudeIdentityStoreError> {
        Ok(self.values.lock().unwrap().get(key).cloned())
    }
    fn set(
        &self,
        key: &str,
        value: &[u8],
        _ttl: Duration,
    ) -> Result<bool, ClaudeIdentityStoreError> {
        if self.reject_write {
            return Ok(false);
        }
        self.values
            .lock()
            .unwrap()
            .insert(key.to_owned(), value.to_vec());
        Ok(true)
    }
    fn set_nx(
        &self,
        _key: &str,
        _value: &[u8],
        _ttl: Duration,
    ) -> Result<bool, ClaudeIdentityStoreError> {
        Ok(false)
    }
    fn expire(&self, _key: &str, _ttl: Duration) -> Result<bool, ClaudeIdentityStoreError> {
        Ok(true)
    }
}

#[test]
fn required_store_write_failure_is_explicit() {
    let store = FakeStore {
        reject_write: true,
        ..Default::default()
    };
    let cache = CodexCache {
        id: "cache-id".to_owned(),
        expire: Some(SystemTime::now() + Duration::from_secs(3600)),
    };
    assert_eq!(
        CodexPromptCacheStore::new().set_required(Some(&store), "key", cache),
        Err(ClaudeIdentityStoreError::WriteSkipped)
    );
}

#[test]
fn local_and_authoritative_stores_round_trip_and_expire() {
    let cache_store = CodexPromptCacheStore::new();
    let cache = CodexCache {
        id: "cache-id".to_owned(),
        expire: Some(SystemTime::now() + Duration::from_secs(3600)),
    };
    cache_store.set("key", cache.clone());
    assert_eq!(cache_store.get("key"), Some(cache.clone()));
    let external = FakeStore::default();
    cache_store
        .set_required(Some(&external), "remote", cache.clone())
        .unwrap();
    let remote = cache_store
        .get_required(Some(&external), "remote")
        .unwrap()
        .unwrap();
    assert_eq!(remote.id, cache.id);
    assert!(remote.expire.is_some());
    cache_store.set(
        "expired",
        CodexCache {
            id: "old".to_owned(),
            expire: Some(SystemTime::UNIX_EPOCH),
        },
    );
    assert_eq!(cache_store.get("expired"), None);
}

#[test]
fn cache_key_hashes_both_untrusted_parts() {
    let key = codex_prompt_cache_key("model/name", "user@example.com");
    assert!(key.starts_with("cpa:codex:prompt-cache:"));
    assert!(!key.contains("model/name"));
    assert!(!key.contains("user@example.com"));
}
