// ref: internal/runtime/executor/helps/user_id_cache_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use super::cloak_utils::{generate_fake_user_id, is_valid_user_id};
use super::user_id_cache::*;

#[derive(Default)]
struct FakeStore {
    inner: Mutex<FakeStoreInner>,
}

#[derive(Default)]
struct FakeStoreInner {
    values: HashMap<String, Vec<u8>>,
    get_error: bool,
    set_error: bool,
    expire_error: bool,
    set_no_persist: bool,
    get_count: usize,
    set_count: usize,
    expire_count: usize,
    last_set_ttl: Option<Duration>,
    last_expire_ttl: Option<Duration>,
}

impl ClaudeIdentityKvStore for FakeStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, ClaudeIdentityStoreError> {
        let mut inner = self.inner.lock().unwrap();
        inner.get_count += 1;
        if inner.get_error {
            return Err(ClaudeIdentityStoreError::Backend("get failed".to_owned()));
        }
        Ok(inner.values.get(key).cloned())
    }

    fn set(
        &self,
        key: &str,
        value: &[u8],
        ttl: Duration,
    ) -> Result<bool, ClaudeIdentityStoreError> {
        let mut inner = self.inner.lock().unwrap();
        inner.set_count += 1;
        inner.last_set_ttl = Some(ttl);
        inner.values.insert(key.to_owned(), value.to_vec());
        Ok(true)
    }

    fn set_nx(
        &self,
        key: &str,
        value: &[u8],
        ttl: Duration,
    ) -> Result<bool, ClaudeIdentityStoreError> {
        let mut inner = self.inner.lock().unwrap();
        inner.set_count += 1;
        inner.last_set_ttl = Some(ttl);
        if inner.set_error {
            return Err(ClaudeIdentityStoreError::Backend("set failed".to_owned()));
        }
        if inner.values.contains_key(key) {
            return Ok(false);
        }
        if !inner.set_no_persist {
            inner.values.insert(key.to_owned(), value.to_vec());
        }
        Ok(true)
    }

    fn expire(&self, _key: &str, ttl: Duration) -> Result<bool, ClaudeIdentityStoreError> {
        let mut inner = self.inner.lock().unwrap();
        inner.expire_count += 1;
        inner.last_expire_ttl = Some(ttl);
        if inner.expire_error {
            return Err(ClaudeIdentityStoreError::Backend(
                "expire failed".to_owned(),
            ));
        }
        Ok(true)
    }
}

#[test]
fn reuses_within_ttl() {
    let cache = UserIdCache::new();
    let first = cache.cached_user_id("api-key-1");
    let second = cache.cached_user_id("api-key-1");
    assert!(!first.is_empty());
    assert_eq!(first, second);
}

#[test]
fn expires_after_ttl() {
    let cache = UserIdCache::new();
    let first = cache.cached_user_id("api-key-expired");
    cache.expire_for_test("api-key-expired");
    let second = cache.cached_user_id("api-key-expired");
    assert_ne!(first, second);
    assert!(!second.is_empty());
}

#[test]
fn is_scoped_by_api_key() {
    let cache = UserIdCache::new();
    assert_ne!(
        cache.cached_user_id("api-key-1"),
        cache.cached_user_id("api-key-2")
    );
}

#[test]
fn renews_ttl_on_hit() {
    let cache = UserIdCache::new();
    let id = cache.cached_user_id("api-key-renew");
    assert_eq!(cache.cached_user_id("api-key-renew"), id);
    assert!(cache.remaining_ttl_for_test("api-key-renew").unwrap() > Duration::from_secs(30 * 60));
}

#[test]
fn required_store_reuses_across_local_cache_instances() {
    let store = FakeStore::default();
    let first = UserIdCache::new()
        .cached_user_id_required(Some(&store), "api-key-1")
        .unwrap();
    let second = UserIdCache::new()
        .cached_user_id_required(Some(&store), "api-key-1")
        .unwrap();
    assert_eq!(first, second);
    assert!(is_valid_user_id(&first));
    let inner = store.inner.lock().unwrap();
    assert_eq!(inner.set_count, 2);
    assert_eq!(inner.expire_count, 1);
    assert_eq!(inner.last_expire_ttl, Some(USER_ID_TTL));
    assert_eq!(inner.last_set_ttl, Some(USER_ID_TTL));
}

#[test]
fn empty_api_key_does_not_use_store() {
    let store = FakeStore::default();
    let value = UserIdCache::new()
        .cached_user_id_required(Some(&store), "")
        .unwrap();
    assert!(is_valid_user_id(&value));
    let inner = store.inner.lock().unwrap();
    assert_eq!(
        (inner.get_count, inner.set_count, inner.expire_count),
        (0, 0, 0)
    );
}

#[test]
fn required_store_get_failure_is_propagated() {
    let store = FakeStore::default();
    store.inner.lock().unwrap().get_error = true;
    assert!(UserIdCache::new()
        .cached_user_id_required(Some(&store), "api-key-1")
        .is_err());
}

#[test]
fn required_store_set_failure_is_propagated() {
    let store = FakeStore::default();
    store.inner.lock().unwrap().set_error = true;
    assert!(UserIdCache::new()
        .cached_user_id_required(Some(&store), "api-key-1")
        .is_err());
}

#[test]
fn required_store_expire_failure_is_propagated() {
    let store = FakeStore::default();
    let key = claude_user_id_kv_key("api-key-1");
    {
        let mut inner = store.inner.lock().unwrap();
        inner
            .values
            .insert(key, generate_fake_user_id().into_bytes());
        inner.expire_error = true;
    }
    assert!(UserIdCache::new()
        .cached_user_id_required(Some(&store), "api-key-1")
        .is_err());
}

#[test]
fn required_store_requires_read_after_set() {
    let store = FakeStore::default();
    store.inner.lock().unwrap().set_no_persist = true;
    assert!(UserIdCache::new()
        .cached_user_id_required(Some(&store), "api-key-1")
        .is_err());
}

#[test]
fn user_id_embeds_the_same_cached_session_id() {
    let cache = UserIdCache::new();
    let session = cache.cached_session_id(None, "api-key-shared-session");
    let user_id = cache.cached_user_id("api-key-shared-session");
    let value: serde_json::Value = serde_json::from_str(&user_id).unwrap();
    assert_eq!(value["session_id"], session);
}
