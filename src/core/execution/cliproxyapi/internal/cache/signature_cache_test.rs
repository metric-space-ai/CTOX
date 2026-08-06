// ref: internal/cache/signature_cache_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::signature_cache::*;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

const MODEL: &str = "claude-sonnet-4-5";
const SIGNATURE: &str = "abc123validSignature1234567890123456789012345678901234567890";
const GEMINI_BYPASS: &str = "skip_thought_signature_validator";

#[test]
fn stores_overwrites_groups_and_unicode_text() {
    let _guard = signature_cache_test_guard();
    clear_signature_cache("");
    assert!(cache_signature(MODEL, "한글 🎉", SIGNATURE));
    assert_eq!(get_cached_signature(MODEL, "한글 🎉"), SIGNATURE);
    let replacement = "replacementSignature123456789012345678901234567890123456789";
    assert!(cache_signature(MODEL, "한글 🎉", replacement));
    assert_eq!(get_cached_signature(MODEL, "한글 🎉"), replacement);
    assert_eq!(
        get_cached_signature("gemini-3-pro", "한글 🎉"),
        GEMINI_BYPASS
    );
}

#[test]
fn rejects_short_or_empty_inputs_and_supports_exact_delete() {
    let _guard = signature_cache_test_guard();
    clear_signature_cache("");
    assert!(!cache_signature(MODEL, "", SIGNATURE));
    assert!(!cache_signature(MODEL, "text", "short"));
    assert_eq!(get_cached_signature(MODEL, "text"), "");
    assert!(cache_signature(MODEL, "text", SIGNATURE));
    delete_cached_signature(MODEL, "text");
    assert_eq!(get_cached_signature(MODEL, "text"), "");
}

#[test]
fn clears_exact_model_group_and_all_groups() {
    let _guard = signature_cache_test_guard();
    clear_signature_cache("");
    assert!(cache_signature(MODEL, "one", SIGNATURE));
    assert!(cache_signature("claude-opus", "two", SIGNATURE));
    assert!(cache_signature("gpt-5", "one", SIGNATURE));
    clear_signature_cache("claude-haiku");
    assert_eq!(get_cached_signature(MODEL, "one"), "");
    assert_eq!(get_cached_signature("claude-opus", "two"), "");
    assert_eq!(get_cached_signature("gpt-5", "one"), SIGNATURE);
    clear_signature_cache("");
    assert_eq!(get_cached_signature("gpt-5", "one"), "");
}

#[test]
fn validates_lengths_gemini_sentinel_and_mode_switches() {
    let _guard = signature_cache_test_guard();
    assert!(has_valid_signature(MODEL, &"x".repeat(50)));
    assert!(!has_valid_signature(MODEL, &"x".repeat(49)));
    assert!(has_valid_signature("gemini-3-pro", GEMINI_BYPASS));
    assert!(!has_valid_signature(MODEL, GEMINI_BYPASS));

    let cache_previous = set_signature_cache_enabled(false);
    assert!(!signature_cache_enabled());
    set_signature_cache_enabled(cache_previous);
    let strict_previous = set_signature_bypass_strict_mode(true);
    assert!(signature_bypass_strict_mode());
    set_signature_bypass_strict_mode(strict_previous);
}

#[derive(Default)]
struct FakeKvStore {
    values: Mutex<HashMap<String, Vec<u8>>>,
    fail_get: bool,
    fail_set: bool,
    fail_expire: bool,
    calls: Mutex<Vec<(String, Duration)>>,
}

impl SignatureKvStore for FakeKvStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, SignatureCacheStoreError> {
        if self.fail_get {
            return Err(SignatureCacheStoreError::Read);
        }
        Ok(self.values.lock().unwrap().get(key).cloned())
    }

    fn set(
        &self,
        key: &str,
        value: &[u8],
        ttl: Duration,
    ) -> Result<bool, SignatureCacheStoreError> {
        if self.fail_set {
            return Err(SignatureCacheStoreError::Write);
        }
        self.values
            .lock()
            .unwrap()
            .insert(key.to_owned(), value.to_vec());
        self.calls.lock().unwrap().push(("set".to_owned(), ttl));
        Ok(true)
    }

    fn delete(&self, key: &str) -> Result<bool, SignatureCacheStoreError> {
        Ok(self.values.lock().unwrap().remove(key).is_some())
    }

    fn expire(&self, _key: &str, ttl: Duration) -> Result<bool, SignatureCacheStoreError> {
        if self.fail_expire {
            return Err(SignatureCacheStoreError::Expire);
        }
        self.calls.lock().unwrap().push(("expire".to_owned(), ttl));
        Ok(true)
    }
}

#[test]
fn durable_required_read_refreshes_sliding_ttl_and_uses_hashed_key() {
    let store = FakeKvStore::default();
    let key = signature_kv_key(MODEL, "thinking text");
    assert!(key.starts_with("cpa:signature:claude:"));
    assert!(!key.contains("thinking text"));
    store
        .values
        .lock()
        .unwrap()
        .insert(key, SIGNATURE.as_bytes().to_vec());
    assert_eq!(
        get_cached_signature_required(Some(&store), MODEL, "thinking text").unwrap(),
        SIGNATURE
    );
    assert_eq!(
        store.calls.lock().unwrap().as_slice(),
        &[("expire".to_owned(), SIGNATURE_CACHE_TTL)]
    );
}

#[test]
fn durable_miss_and_failure_never_fall_back_to_local_cache() {
    let _guard = signature_cache_test_guard();
    clear_signature_cache("");
    cache_signature(MODEL, "thinking text", SIGNATURE);
    let store = FakeKvStore::default();
    assert_eq!(
        get_cached_signature_required(Some(&store), MODEL, "thinking text").unwrap(),
        ""
    );
    let failing = FakeKvStore {
        fail_get: true,
        ..FakeKvStore::default()
    };
    assert_eq!(
        get_cached_signature_required(Some(&failing), MODEL, "thinking text"),
        Err(SignatureCacheStoreError::Read)
    );
}

#[test]
fn durable_best_effort_write_delete_and_errors_are_explicit() {
    let store = FakeKvStore::default();
    assert!(cache_signature_best_effort(
        Some(&store),
        MODEL,
        "thinking text",
        SIGNATURE
    ));
    assert_eq!(
        get_cached_signature_required(Some(&store), MODEL, "thinking text").unwrap(),
        SIGNATURE
    );
    delete_cached_signature_required(Some(&store), MODEL, "thinking text").unwrap();
    assert_eq!(
        get_cached_signature_required(Some(&store), MODEL, "thinking text").unwrap(),
        ""
    );
    let failing = FakeKvStore {
        fail_set: true,
        ..FakeKvStore::default()
    };
    assert!(!cache_signature_best_effort(
        Some(&failing),
        MODEL,
        "thinking text",
        SIGNATURE
    ));
}

#[test]
fn durable_expire_failure_and_gemini_empty_sentinel_propagate_correctly() {
    let store = FakeKvStore {
        fail_expire: true,
        ..FakeKvStore::default()
    };
    store.values.lock().unwrap().insert(
        signature_kv_key(MODEL, "thinking text"),
        SIGNATURE.as_bytes().to_vec(),
    );
    assert_eq!(
        get_cached_signature_required(Some(&store), MODEL, "thinking text"),
        Err(SignatureCacheStoreError::Expire)
    );
    assert_eq!(
        get_cached_signature_required(Some(&store), "gemini-3-pro", "").unwrap(),
        GEMINI_BYPASS
    );
}
