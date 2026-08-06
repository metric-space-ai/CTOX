// ref: internal/cache/signature_cache.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::internal::home::hash_key_part;

pub const SIGNATURE_CACHE_TTL: Duration = Duration::from_secs(3 * 60 * 60);
pub const SIGNATURE_TEXT_HASH_LEN: usize = 16;
pub const MIN_VALID_SIGNATURE_LEN: usize = 50;

const GEMINI_BYPASS: &str = "skip_thought_signature_validator";

#[derive(Clone, Debug)]
struct SignatureEntry {
    signature: String,
    touched_at: Instant,
}

type GroupCache = HashMap<String, SignatureEntry>;

static CACHE: OnceLock<Mutex<HashMap<String, GroupCache>>> = OnceLock::new();
static SIGNATURE_CACHE_ENABLED: AtomicBool = AtomicBool::new(true);
static SIGNATURE_BYPASS_STRICT_MODE: AtomicBool = AtomicBool::new(false);

fn cache() -> &'static Mutex<HashMap<String, GroupCache>> {
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
pub(crate) fn signature_cache_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn text_hash(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    digest
        .iter()
        .take(SIGNATURE_TEXT_HASH_LEN / 2)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn cache_signature(model_name: &str, text: &str, signature: &str) -> bool {
    if text.is_empty() || signature.len() < MIN_VALID_SIGNATURE_LEN {
        return false;
    }
    let mut groups = cache().lock().unwrap_or_else(|error| error.into_inner());
    groups
        .entry(get_model_group(model_name))
        .or_default()
        .insert(
            text_hash(text),
            SignatureEntry {
                signature: signature.to_owned(),
                touched_at: Instant::now(),
            },
        );
    true
}

/// Returns a cached signature and refreshes its sliding three-hour TTL.
/// Gemini misses use the provider-documented synthetic-history sentinel.
pub fn get_cached_signature(model_name: &str, text: &str) -> String {
    let group = get_model_group(model_name);
    if text.is_empty() {
        return gemini_miss(&group);
    }
    let key = text_hash(text);
    let now = Instant::now();
    let mut groups = cache().lock().unwrap_or_else(|error| error.into_inner());
    let Some(entries) = groups.get_mut(&group) else {
        return gemini_miss(&group);
    };
    let expired = entries
        .get(&key)
        .is_some_and(|entry| now.duration_since(entry.touched_at) > SIGNATURE_CACHE_TTL);
    if expired {
        entries.remove(&key);
    }
    let signature = entries.get_mut(&key).map(|entry| {
        entry.touched_at = now;
        entry.signature.clone()
    });
    if entries.is_empty() {
        groups.remove(&group);
    }
    signature.unwrap_or_else(|| gemini_miss(&group))
}

/// Explicit durable key-value boundary. CTOX injects its own revisioned store;
/// the portable port never discovers one from ambient process state.
pub trait SignatureKvStore: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, SignatureCacheStoreError>;
    fn set(&self, key: &str, value: &[u8], ttl: Duration)
        -> Result<bool, SignatureCacheStoreError>;
    fn delete(&self, key: &str) -> Result<bool, SignatureCacheStoreError>;
    fn expire(&self, key: &str, ttl: Duration) -> Result<bool, SignatureCacheStoreError>;
}

/// Required request-time read. Passing a durable store makes it authoritative:
/// a miss or error never falls through to the process-local cache.
pub fn get_cached_signature_required(
    store: Option<&dyn SignatureKvStore>,
    model_name: &str,
    text: &str,
) -> Result<String, SignatureCacheStoreError> {
    let group = get_model_group(model_name);
    if text.is_empty() {
        return Ok(gemini_miss(&group));
    }
    let Some(store) = store else {
        return Ok(get_cached_signature(model_name, text));
    };
    let key = signature_kv_key(model_name, text);
    let Some(raw) = store.get(&key)? else {
        return Ok(gemini_miss(&group));
    };
    let signature = String::from_utf8(raw).map_err(|_| SignatureCacheStoreError::InvalidValue)?;
    store.expire(&key, SIGNATURE_CACHE_TTL)?;
    Ok(signature)
}

/// Completed response paths publish durable signatures best-effort, exactly as
/// upstream does. An injected store remains authoritative on failure.
pub fn cache_signature_best_effort(
    store: Option<&dyn SignatureKvStore>,
    model_name: &str,
    text: &str,
    signature: &str,
) -> bool {
    if text.is_empty() || signature.len() < MIN_VALID_SIGNATURE_LEN {
        return false;
    }
    match store {
        Some(store) => store
            .set(
                &signature_kv_key(model_name, text),
                signature.as_bytes(),
                SIGNATURE_CACHE_TTL,
            )
            .unwrap_or(false),
        None => cache_signature(model_name, text, signature),
    }
}

pub fn delete_cached_signature_required(
    store: Option<&dyn SignatureKvStore>,
    model_name: &str,
    text: &str,
) -> Result<(), SignatureCacheStoreError> {
    if text.is_empty() {
        return Ok(());
    }
    if let Some(store) = store {
        store.delete(&signature_kv_key(model_name, text))?;
    } else {
        delete_cached_signature(model_name, text);
    }
    Ok(())
}

pub fn signature_kv_key(model_name: &str, text: &str) -> String {
    format!(
        "cpa:signature:{}:{}",
        get_model_group(model_name),
        hash_key_part(text)
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignatureCacheStoreError {
    Unavailable,
    Read,
    Write,
    Delete,
    Expire,
    InvalidValue,
}

impl std::fmt::Display for SignatureCacheStoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("signature cache store unavailable")
    }
}

impl std::error::Error for SignatureCacheStoreError {}

pub fn delete_cached_signature(model_name: &str, text: &str) {
    if text.is_empty() {
        return;
    }
    let group = get_model_group(model_name);
    let key = text_hash(text);
    let mut groups = cache().lock().unwrap_or_else(|error| error.into_inner());
    if let Some(entries) = groups.get_mut(&group) {
        entries.remove(&key);
        if entries.is_empty() {
            groups.remove(&group);
        }
    }
}

pub fn clear_signature_cache(model_name: &str) {
    let mut groups = cache().lock().unwrap_or_else(|error| error.into_inner());
    if model_name.is_empty() {
        groups.clear();
    } else {
        groups.remove(&get_model_group(model_name));
    }
}

pub fn has_valid_signature(model_name: &str, signature: &str) -> bool {
    (!signature.is_empty() && signature.len() >= MIN_VALID_SIGNATURE_LEN)
        || (signature == GEMINI_BYPASS && get_model_group(model_name) == "gemini")
}

pub fn get_model_group(model_name: &str) -> String {
    if model_name.contains("gpt") {
        "gpt".to_owned()
    } else if model_name.contains("claude") {
        "claude".to_owned()
    } else if model_name.contains("gemini") {
        "gemini".to_owned()
    } else {
        model_name.to_owned()
    }
}

pub fn set_signature_cache_enabled(enabled: bool) -> bool {
    SIGNATURE_CACHE_ENABLED.swap(enabled, Ordering::SeqCst)
}

pub fn signature_cache_enabled() -> bool {
    SIGNATURE_CACHE_ENABLED.load(Ordering::SeqCst)
}

pub fn set_signature_bypass_strict_mode(strict: bool) -> bool {
    SIGNATURE_BYPASS_STRICT_MODE.swap(strict, Ordering::SeqCst)
}

pub fn signature_bypass_strict_mode() -> bool {
    SIGNATURE_BYPASS_STRICT_MODE.load(Ordering::SeqCst)
}

fn gemini_miss(group: &str) -> String {
    if group == "gemini" {
        GEMINI_BYPASS.to_owned()
    } else {
        String::new()
    }
}
