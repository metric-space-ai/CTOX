// ref: internal/cache/antigravity_reasoning_replay_cache.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub const ANTIGRAVITY_REPLAY_TTL_MS: i64 = 60 * 60 * 1_000;
pub const ANTIGRAVITY_REPLAY_MAX_ITEMS_PER_ENTRY: usize = 4_096;
pub const ANTIGRAVITY_REPLAY_MAX_BYTES_PER_ENTRY: usize = 16 << 20;
pub const ANTIGRAVITY_REPLAY_MAX_SERIALIZED_BYTES: usize = 24 << 20;
const DEFAULT_MAX_ENTRIES: usize = 10_240;
const DEFAULT_EVICT_BATCH: usize = 128;
const MIN_THOUGHT_SIGNATURE_LEN: usize = 16;
const HOME_KEY_PREFIX: &str = "cpa:antigravity:reasoning-replay:";
const GENERATION_ITEM_TYPE: &str = "cpa_antigravity_replay_generation";
const ABSENT_FENCE_ATTEMPTS: usize = 4;

/// Explicit, instance-owned Home-KV authority for reasoning replay state.
///
/// The Go implementation discovers a process-global Home client. CTOX instead
/// injects the authority into one cache instance so credentials, connectivity,
/// and ownership cannot leak across runtimes.
pub trait AntigravityReasoningReplayStore: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, AntigravityReasoningReplayStoreError>;

    fn set(
        &self,
        key: &str,
        value: &[u8],
        ttl_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayStoreError>;

    fn compare_and_swap(
        &self,
        key: &str,
        expected: Option<&[u8]>,
        value: &[u8],
        ttl_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayStoreError>;

    fn expire(&self, key: &str, ttl_ms: i64) -> Result<bool, AntigravityReasoningReplayStoreError>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AntigravityReasoningReplayStoreError;

impl std::fmt::Display for AntigravityReasoningReplayStoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Antigravity reasoning replay store operation failed")
    }
}

impl std::error::Error for AntigravityReasoningReplayStoreError {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AntigravityReasoningReplaySnapshot {
    raw: Vec<u8>,
    items: Vec<Vec<u8>>,
    loaded: bool,
    found: bool,
    revision: u64,
    branch: String,
}

#[derive(Clone, Debug)]
struct Entry {
    items: Vec<Vec<u8>>,
    touched_at_ms: i64,
    revision: u64,
    branch: String,
    deleted: bool,
}

#[derive(Debug, Default)]
struct State {
    entries: HashMap<[u8; 32], Entry>,
    next_revision: u64,
    total_bytes: usize,
}

pub struct AntigravityReasoningReplayCache {
    state: Mutex<State>,
    namespace: &'static [u8],
    max_entries: usize,
    evict_batch: usize,
    max_total_bytes: usize,
    store: Option<Arc<dyn AntigravityReasoningReplayStore>>,
}

impl std::fmt::Debug for AntigravityReasoningReplayCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AntigravityReasoningReplayCache")
            .field("namespace", &self.namespace)
            .field("max_entries", &self.max_entries)
            .field("evict_batch", &self.evict_batch)
            .field("max_total_bytes", &self.max_total_bytes)
            .field("store", &self.store.as_ref().map(|_| "injected"))
            .finish_non_exhaustive()
    }
}

impl Default for AntigravityReasoningReplayCache {
    fn default() -> Self {
        Self::new()
    }
}

impl AntigravityReasoningReplayCache {
    pub fn new() -> Self {
        Self::with_limits(DEFAULT_MAX_ENTRIES, DEFAULT_EVICT_BATCH)
    }

    pub fn with_store(store: Arc<dyn AntigravityReasoningReplayStore>) -> Self {
        Self::with_options(
            b"antigravity-reasoning-replay\0",
            DEFAULT_MAX_ENTRIES,
            DEFAULT_EVICT_BATCH,
            usize::MAX,
            Some(store),
        )
    }

    fn with_limits(max_entries: usize, evict_batch: usize) -> Self {
        Self::with_namespace_limits(b"antigravity-reasoning-replay\0", max_entries, evict_batch)
    }

    pub(crate) fn with_namespace_limits(
        namespace: &'static [u8],
        max_entries: usize,
        evict_batch: usize,
    ) -> Self {
        Self::with_namespace_capacity(namespace, max_entries, evict_batch, usize::MAX)
    }

    pub(crate) fn with_namespace_capacity(
        namespace: &'static [u8],
        max_entries: usize,
        evict_batch: usize,
        max_total_bytes: usize,
    ) -> Self {
        Self::with_options(namespace, max_entries, evict_batch, max_total_bytes, None)
    }

    fn with_options(
        namespace: &'static [u8],
        max_entries: usize,
        evict_batch: usize,
        max_total_bytes: usize,
        store: Option<Arc<dyn AntigravityReasoningReplayStore>>,
    ) -> Self {
        Self {
            state: Mutex::new(State::default()),
            namespace,
            max_entries: max_entries.max(1),
            evict_batch: evict_batch.max(1),
            max_total_bytes,
            store,
        }
    }

    pub fn read(
        &self,
        model: &str,
        session_key: &str,
        now_ms: i64,
    ) -> Result<
        (Vec<Vec<u8>>, AntigravityReasoningReplaySnapshot, bool),
        AntigravityReasoningReplayError,
    > {
        let key = cache_key(self.namespace, model, session_key)
            .ok_or(AntigravityReasoningReplayError::InvalidKey)?;
        if let Some(store) = &self.store {
            return self.read_store(store.as_ref(), model, session_key);
        }
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let expired = state.entries.get(&key).is_some_and(|entry| {
            now_ms.saturating_sub(entry.touched_at_ms) > ANTIGRAVITY_REPLAY_TTL_MS
        });
        if expired {
            remove_entry(&mut state, &key);
        }
        if !state.entries.contains_key(&key) {
            reserve_tombstone(&mut state, key, now_ms);
            self.evict_if_needed(&mut state, Some(key));
        }
        let entry = state.entries.get_mut(&key).expect("read reserves its key");
        entry.touched_at_ms = now_ms;
        let snapshot = AntigravityReasoningReplaySnapshot {
            raw: Vec::new(),
            items: entry.items.clone(),
            loaded: true,
            found: true,
            revision: entry.revision,
            branch: entry.branch.clone(),
        };
        let found = !entry.deleted && !entry.items.is_empty();
        Ok((entry.items.clone(), snapshot, found))
    }

    fn read_store(
        &self,
        store: &dyn AntigravityReasoningReplayStore,
        model: &str,
        session_key: &str,
    ) -> Result<
        (Vec<Vec<u8>>, AntigravityReasoningReplaySnapshot, bool),
        AntigravityReasoningReplayError,
    > {
        let key = home_store_key(model, session_key);
        let mut raw = None;
        for _ in 0..ABSENT_FENCE_ATTEMPTS {
            if let Some(current) = store.get(&key).map_err(store_error)? {
                raw = Some(current);
                break;
            }
            let reservation = marshal_tombstone()?;
            if store
                .compare_and_swap(&key, None, &reservation, ANTIGRAVITY_REPLAY_TTL_MS)
                .map_err(store_error)?
            {
                raw = Some(reservation);
                break;
            }
        }
        let Some(raw) = raw else {
            return Err(AntigravityReasoningReplayError::InvalidSnapshot);
        };
        let mut snapshot = AntigravityReasoningReplaySnapshot {
            raw: raw.clone(),
            loaded: true,
            found: true,
            ..AntigravityReasoningReplaySnapshot::default()
        };
        if raw.len() > ANTIGRAVITY_REPLAY_MAX_SERIALIZED_BYTES {
            return Ok((Vec::new(), snapshot, false));
        }
        let Some(home) = decode_home_value(&raw) else {
            return Ok((Vec::new(), snapshot, false));
        };
        snapshot.branch = home.branch;
        if home.deleted || home.items.is_empty() {
            return Ok((Vec::new(), snapshot, false));
        }
        if home.items.len() > ANTIGRAVITY_REPLAY_MAX_ITEMS_PER_ENTRY {
            return Ok((Vec::new(), snapshot, false));
        }
        let Ok(normalized) = normalize_items(&home.items) else {
            return Ok((Vec::new(), snapshot, false));
        };
        if normalized.len() != home.items.len() {
            return Ok((Vec::new(), snapshot, false));
        }
        snapshot.items = normalized.clone();
        store
            .expire(&key, ANTIGRAVITY_REPLAY_TTL_MS)
            .map_err(store_error)?;
        Ok((normalized, snapshot, true))
    }

    pub fn cache_items(
        &self,
        model: &str,
        session_key: &str,
        items: &[Vec<u8>],
        now_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayError> {
        cache_key(self.namespace, model, session_key)
            .ok_or(AntigravityReasoningReplayError::InvalidKey)?;
        let normalized = normalize_items(items)?;
        self.cache_normalized(model, session_key, normalized, now_ms)
    }

    fn cache_normalized(
        &self,
        model: &str,
        session_key: &str,
        normalized: Vec<Vec<u8>>,
        now_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayError> {
        if let Some(store) = &self.store {
            let raw = marshal_home_value(&normalized, None, false)?;
            return store
                .set(
                    &home_store_key(model, session_key),
                    &raw,
                    ANTIGRAVITY_REPLAY_TTL_MS,
                )
                .map_err(store_error);
        }
        let key = cache_key(self.namespace, model, session_key)
            .ok_or(AntigravityReasoningReplayError::InvalidKey)?;
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        replace_entry(&mut state, key, normalized, now_ms, new_generation(), false);
        self.evict_if_needed(&mut state, Some(key));
        Ok(true)
    }

    pub fn replace_if_unchanged(
        &self,
        model: &str,
        session_key: &str,
        snapshot: &AntigravityReasoningReplaySnapshot,
        items: &[Vec<u8>],
        now_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayError> {
        if cache_key(self.namespace, model, session_key).is_none() {
            return Err(AntigravityReasoningReplayError::InvalidKey);
        }
        let normalized = normalize_items(items)?;
        if !snapshot.loaded {
            return self.cache_normalized(model, session_key, normalized, now_ms);
        }
        self.replace_normalized_if_unchanged(model, session_key, snapshot, normalized, now_ms)
    }

    pub(crate) fn replace_normalized_if_unchanged(
        &self,
        model: &str,
        session_key: &str,
        snapshot: &AntigravityReasoningReplaySnapshot,
        normalized: Vec<Vec<u8>>,
        now_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayError> {
        let key = cache_key(self.namespace, model, session_key)
            .ok_or(AntigravityReasoningReplayError::InvalidKey)?;
        if !snapshot.loaded {
            return Err(AntigravityReasoningReplayError::InvalidSnapshot);
        }
        if let Some(store) = &self.store {
            return self.replace_store(store.as_ref(), model, session_key, snapshot, normalized);
        }
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let Some(current) = state.entries.get(&key) else {
            return Ok(false);
        };
        let exact = snapshot.found && current.revision == snapshot.revision;
        let descendant = !current.deleted
            && !snapshot.branch.is_empty()
            && current.branch == snapshot.branch
            && is_prefix(&current.items, &normalized);
        if !exact && !descendant {
            return Ok(false);
        }
        let keep_branch =
            !snapshot.branch.is_empty() && (descendant || is_prefix(&snapshot.items, &normalized));
        let branch = if keep_branch {
            snapshot.branch.clone()
        } else {
            new_generation()
        };
        replace_entry(&mut state, key, normalized, now_ms, branch, false);
        self.evict_if_needed(&mut state, Some(key));
        Ok(true)
    }

    fn replace_store(
        &self,
        store: &dyn AntigravityReasoningReplayStore,
        model: &str,
        session_key: &str,
        snapshot: &AntigravityReasoningReplaySnapshot,
        normalized: Vec<Vec<u8>>,
    ) -> Result<bool, AntigravityReasoningReplayError> {
        let key = home_store_key(model, session_key);
        let mut expected = snapshot.raw.clone();
        let mut expected_found = snapshot.found;
        let branch = if snapshot.branch.is_empty() || !is_prefix(&snapshot.items, &normalized) {
            new_generation()
        } else {
            snapshot.branch.clone()
        };
        for _ in 0..ABSENT_FENCE_ATTEMPTS {
            let raw = marshal_home_value(&normalized, Some(&branch), false)?;
            if store
                .compare_and_swap(
                    &key,
                    expected_found.then_some(expected.as_slice()),
                    &raw,
                    ANTIGRAVITY_REPLAY_TTL_MS,
                )
                .map_err(store_error)?
            {
                return Ok(true);
            }
            let Some(current_raw) = store.get(&key).map_err(store_error)? else {
                return Ok(false);
            };
            if current_raw.len() > ANTIGRAVITY_REPLAY_MAX_SERIALIZED_BYTES {
                return Ok(false);
            }
            let Some(current) = decode_home_value(&current_raw) else {
                return Ok(false);
            };
            if current.deleted || snapshot.branch.is_empty() || current.branch != snapshot.branch {
                return Ok(false);
            }
            let Ok(normalized_current) = normalize_items(&current.items) else {
                return Ok(false);
            };
            if normalized_current.len() != current.items.len()
                || !is_prefix(&normalized_current, &normalized)
            {
                return Ok(false);
            }
            expected = current_raw;
            expected_found = true;
        }
        Ok(false)
    }

    pub(crate) fn replace_exact_if_unchanged(
        &self,
        model: &str,
        session_key: &str,
        snapshot: &AntigravityReasoningReplaySnapshot,
        items: Vec<Vec<u8>>,
        now_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayError> {
        let key = cache_key(self.namespace, model, session_key)
            .ok_or(AntigravityReasoningReplayError::InvalidKey)?;
        if !snapshot.loaded {
            return Err(AntigravityReasoningReplayError::InvalidSnapshot);
        }
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let Some(current) = state.entries.get(&key) else {
            return Ok(false);
        };
        if !snapshot.found || current.revision != snapshot.revision {
            return Ok(false);
        }
        replace_entry(&mut state, key, items, now_ms, new_generation(), false);
        self.evict_if_needed(&mut state, Some(key));
        Ok(true)
    }

    pub fn delete_if_unchanged(
        &self,
        model: &str,
        session_key: &str,
        snapshot: &AntigravityReasoningReplaySnapshot,
        now_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayError> {
        let key = cache_key(self.namespace, model, session_key)
            .ok_or(AntigravityReasoningReplayError::InvalidKey)?;
        if !snapshot.loaded {
            self.delete(model, session_key, now_ms)?;
            return Ok(true);
        }
        if let Some(store) = &self.store {
            let tombstone = marshal_tombstone()?;
            return store
                .compare_and_swap(
                    &home_store_key(model, session_key),
                    snapshot.found.then_some(snapshot.raw.as_slice()),
                    &tombstone,
                    ANTIGRAVITY_REPLAY_TTL_MS,
                )
                .map_err(store_error);
        }
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let Some(current) = state.entries.get(&key) else {
            return Ok(false);
        };
        if !snapshot.found || current.revision != snapshot.revision {
            return Ok(false);
        }
        reserve_tombstone(&mut state, key, now_ms);
        self.evict_if_needed(&mut state, Some(key));
        Ok(true)
    }

    pub fn delete(
        &self,
        model: &str,
        session_key: &str,
        now_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayError> {
        let key = cache_key(self.namespace, model, session_key)
            .ok_or(AntigravityReasoningReplayError::InvalidKey)?;
        if let Some(store) = &self.store {
            let tombstone = marshal_tombstone()?;
            return store
                .set(
                    &home_store_key(model, session_key),
                    &tombstone,
                    ANTIGRAVITY_REPLAY_TTL_MS,
                )
                .map_err(store_error);
        }
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        reserve_tombstone(&mut state, key, now_ms);
        self.evict_if_needed(&mut state, Some(key));
        Ok(true)
    }

    pub(crate) fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.entries.clear();
        state.total_bytes = 0;
    }

    #[cfg(test)]
    fn evict_oldest(&self, count: usize) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        evict_oldest(&mut state, count, None);
    }

    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entries
            .len()
    }

    #[cfg(test)]
    pub(crate) fn total_bytes(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .total_bytes
    }

    fn evict_if_needed(&self, state: &mut State, protected: Option<[u8; 32]>) {
        while state.entries.len() > self.max_entries || state.total_bytes > self.max_total_bytes {
            let mut candidates = state
                .entries
                .iter()
                .filter(|(key, _)| protected.as_ref() != Some(*key))
                .map(|(key, entry)| (*key, entry.touched_at_ms))
                .collect::<Vec<_>>();
            candidates.sort_by_key(|(_, touched)| *touched);
            if candidates.is_empty() {
                break;
            }
            for (key, _) in candidates.into_iter().take(self.evict_batch) {
                remove_entry(state, &key);
            }
        }
    }
}

fn reserve_tombstone(state: &mut State, key: [u8; 32], now_ms: i64) {
    replace_entry(state, key, Vec::new(), now_ms, new_generation(), true);
}

fn entry_bytes(items: &[Vec<u8>]) -> usize {
    items.iter().map(Vec::len).sum()
}

fn remove_entry(state: &mut State, key: &[u8; 32]) {
    if let Some(entry) = state.entries.remove(key) {
        state.total_bytes = state.total_bytes.saturating_sub(entry_bytes(&entry.items));
    }
}

fn replace_entry(
    state: &mut State,
    key: [u8; 32],
    items: Vec<Vec<u8>>,
    now_ms: i64,
    branch: String,
    deleted: bool,
) {
    remove_entry(state, &key);
    state.next_revision += 1;
    state.total_bytes = state.total_bytes.saturating_add(entry_bytes(&items));
    state.entries.insert(
        key,
        Entry {
            items,
            touched_at_ms: now_ms,
            revision: state.next_revision,
            branch,
            deleted,
        },
    );
}

fn cache_key(namespace: &[u8], model: &str, session_key: &str) -> Option<[u8; 32]> {
    let model = model.trim();
    let session_key = session_key.trim();
    if model.is_empty() || session_key.is_empty() {
        return None;
    }
    let mut digest = Sha256::new();
    digest.update(namespace);
    digest.update(model.as_bytes());
    digest.update(b"\0");
    digest.update(session_key.as_bytes());
    Some(digest.finalize().into())
}

fn home_store_key(model: &str, session_key: &str) -> String {
    format!(
        "{HOME_KEY_PREFIX}{}:{}",
        hash_key_part(model.trim()),
        hash_key_part(session_key.trim())
    )
}

fn hash_key_part(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

fn new_generation() -> String {
    Uuid::new_v4().simple().to_string()
}

struct HomeValue {
    items: Vec<Vec<u8>>,
    deleted: bool,
    branch: String,
}

fn marshal_home_value(
    items: &[Vec<u8>],
    branch: Option<&str>,
    deleted: bool,
) -> Result<Vec<u8>, AntigravityReasoningReplayError> {
    let branch = branch
        .filter(|value| !value.is_empty())
        .map_or_else(new_generation, str::to_owned);
    let mut marker = Map::new();
    marker.insert(
        "type".to_owned(),
        Value::String(GENERATION_ITEM_TYPE.to_owned()),
    );
    marker.insert("generation".to_owned(), Value::String(new_generation()));
    marker.insert("branch".to_owned(), Value::String(branch));
    if deleted {
        marker.insert("deleted".to_owned(), Value::Bool(true));
    }
    let marker = serde_json::to_vec(&Value::Object(marker))
        .map_err(|_| AntigravityReasoningReplayError::InvalidItems)?;
    let encoded = std::iter::once(marker.as_slice())
        .chain(items.iter().map(Vec::as_slice))
        .map(|item| BASE64_STANDARD.encode(item))
        .collect::<Vec<_>>();
    let raw =
        serde_json::to_vec(&encoded).map_err(|_| AntigravityReasoningReplayError::InvalidItems)?;
    if raw.len() > ANTIGRAVITY_REPLAY_MAX_SERIALIZED_BYTES {
        return Err(AntigravityReasoningReplayError::TooLarge);
    }
    Ok(raw)
}

fn marshal_tombstone() -> Result<Vec<u8>, AntigravityReasoningReplayError> {
    marshal_home_value(&[], None, true)
}

fn decode_home_value(raw: &[u8]) -> Option<HomeValue> {
    let encoded = serde_json::from_slice::<Vec<String>>(raw).ok()?;
    let mut items = encoded
        .iter()
        .map(|item| BASE64_STANDARD.decode(item).ok())
        .collect::<Option<Vec<_>>>()?;
    if items.is_empty() {
        return Some(HomeValue {
            items,
            deleted: false,
            branch: String::new(),
        });
    }
    let marker = serde_json::from_slice::<Value>(&items[0]).ok();
    let is_marker = marker
        .as_ref()
        .and_then(|value| value.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|kind| kind.trim() == GENERATION_ITEM_TYPE);
    if !is_marker {
        return Some(HomeValue {
            items,
            deleted: false,
            branch: String::new(),
        });
    }
    let marker = marker.expect("marker was validated");
    let deleted = marker
        .get("deleted")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let branch = marker
        .get("branch")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("")
        .to_owned();
    items.remove(0);
    Some(HomeValue {
        items,
        deleted,
        branch,
    })
}

fn store_error(_: AntigravityReasoningReplayStoreError) -> AntigravityReasoningReplayError {
    AntigravityReasoningReplayError::InvalidSnapshot
}

#[cfg(test)]
fn evict_oldest(state: &mut State, count: usize, protected: Option<[u8; 32]>) {
    let mut candidates = state
        .entries
        .iter()
        .filter(|(key, _)| protected.as_ref() != Some(*key))
        .map(|(key, entry)| (*key, entry.touched_at_ms))
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(_, touched)| *touched);
    for (key, _) in candidates.into_iter().take(count) {
        remove_entry(state, &key);
    }
}

fn normalize_items(items: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, AntigravityReasoningReplayError> {
    if items.len() > ANTIGRAVITY_REPLAY_MAX_ITEMS_PER_ENTRY {
        return Err(AntigravityReasoningReplayError::TooLarge);
    }
    let mut normalized = Vec::new();
    let mut bytes = 0_usize;
    for item in items {
        let Some(item) = normalize_item(item) else {
            continue;
        };
        bytes = bytes.saturating_add(item.len());
        if bytes > ANTIGRAVITY_REPLAY_MAX_BYTES_PER_ENTRY {
            return Err(AntigravityReasoningReplayError::TooLarge);
        }
        normalized.push(item);
    }
    if normalized.is_empty() {
        return Err(AntigravityReasoningReplayError::InvalidItems);
    }
    Ok(normalized)
}

fn normalize_item(raw: &[u8]) -> Option<Vec<u8>> {
    let root = serde_json::from_slice::<Value>(raw).ok()?;
    match root.get("type").and_then(Value::as_str)?.trim() {
        "thought_signature" => normalize_thought(&root),
        "function_call_part" => normalize_function(&root),
        _ => None,
    }
}

fn normalize_thought(root: &Value) -> Option<Vec<u8>> {
    let signature = root
        .get("thoughtSignature")
        .or_else(|| root.get("thought_signature"))
        .and_then(Value::as_str)?
        .trim();
    if signature == "skip_thought_signature_validator"
        || signature.len() < MIN_THOUGHT_SIGNATURE_LEN
    {
        return None;
    }
    let mut out = Map::new();
    out.insert(
        "type".to_owned(),
        Value::String("thought_signature".to_owned()),
    );
    out.insert(
        "thoughtSignature".to_owned(),
        Value::String(signature.to_owned()),
    );
    copy_integer(root, &mut out, "contentIndex", false);
    copy_integer(root, &mut out, "partIndex", false);
    if let Some(kind @ ("text" | "thought")) = root.get("targetKind").and_then(Value::as_str) {
        out.insert("targetKind".to_owned(), Value::String(kind.to_owned()));
    }
    copy_trimmed_string(root, &mut out, "targetHash");
    copy_integer(root, &mut out, "targetOccurrence", true);
    copy_trimmed_string(root, &mut out, "contextHash");
    serde_json::to_vec(&Value::Object(out)).ok()
}

fn normalize_function(root: &Value) -> Option<Vec<u8>> {
    let nested = root.get("functionCall").unwrap_or(&Value::Null);
    let call_id = root
        .get("call_id")
        .or_else(|| root.get("id"))
        .or_else(|| nested.get("id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    let name = root
        .get("name")
        .or_else(|| nested.get("name"))
        .and_then(Value::as_str)?
        .trim();
    let args = root.get("args").or_else(|| nested.get("args"))?;
    if name.is_empty() {
        return None;
    }
    let mut out = Map::new();
    out.insert(
        "type".to_owned(),
        Value::String("function_call_part".to_owned()),
    );
    if !call_id.is_empty() {
        out.insert("call_id".to_owned(), Value::String(call_id.to_owned()));
    }
    out.insert("name".to_owned(), Value::String(name.to_owned()));
    out.insert("args".to_owned(), args.clone());
    if let Some(signature) = root
        .get("thoughtSignature")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "skip_thought_signature_validator")
    {
        out.insert(
            "thoughtSignature".to_owned(),
            Value::String(signature.to_owned()),
        );
    }
    copy_integer(root, &mut out, "contentIndex", false);
    copy_integer(root, &mut out, "partIndex", false);
    copy_integer(root, &mut out, "targetOccurrence", true);
    copy_trimmed_string(root, &mut out, "contextHash");
    serde_json::to_vec(&Value::Object(out)).ok()
}

fn copy_integer(root: &Value, out: &mut Map<String, Value>, key: &str, non_negative: bool) {
    if let Some(value) = root
        .get(key)
        .and_then(Value::as_i64)
        .filter(|value| !non_negative || *value >= 0)
    {
        out.insert(key.to_owned(), Value::from(value));
    }
}

fn copy_trimmed_string(root: &Value, out: &mut Map<String, Value>, key: &str) {
    if let Some(value) = root
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        out.insert(key.to_owned(), Value::String(value.to_owned()));
    }
}

fn is_prefix(prefix: &[Vec<u8>], items: &[Vec<u8>]) -> bool {
    prefix.len() <= items.len() && prefix.iter().zip(items).all(|(left, right)| left == right)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AntigravityReasoningReplayError {
    InvalidKey,
    InvalidSnapshot,
    InvalidItems,
    TooLarge,
}

impl std::fmt::Display for AntigravityReasoningReplayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Antigravity reasoning replay state is invalid")
    }
}

impl std::error::Error for AntigravityReasoningReplayError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(signature: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "type":"thought_signature", "thoughtSignature":signature,
            "targetKind":"text", "targetHash":"hash", "ignored":"secret"
        }))
        .unwrap()
    }

    #[test]
    fn stale_first_writer_cannot_cross_a_tombstone() {
        let cache = AntigravityReasoningReplayCache::new();
        let (_, stale, found) = cache.read("gemini-3", "session-secret", 1).unwrap();
        assert!(!found);
        let (_, clear, _) = cache.read("gemini-3", "session-secret", 2).unwrap();
        assert!(cache
            .delete_if_unchanged("gemini-3", "session-secret", &clear, 3)
            .unwrap());
        assert!(!cache
            .replace_if_unchanged(
                "gemini-3",
                "session-secret",
                &stale,
                &[item("stale-signature-123456")],
                4
            )
            .unwrap());
    }

    #[test]
    fn descendant_extension_wins_but_stale_sibling_does_not() {
        let cache = AntigravityReasoningReplayCache::new();
        let (_, empty, _) = cache.read("gemini-3", "lane", 1).unwrap();
        let first = item("first-signature-123456");
        assert!(cache
            .replace_if_unchanged("gemini-3", "lane", &empty, std::slice::from_ref(&first), 2)
            .unwrap());
        let (one, parent, found) = cache.read("gemini-3", "lane", 3).unwrap();
        assert!(found);
        let second = item("second-signature-123456");
        assert!(cache
            .replace_if_unchanged(
                "gemini-3",
                "lane",
                &parent,
                &[one[0].clone(), second.clone()],
                4
            )
            .unwrap());
        assert!(!cache
            .replace_if_unchanged(
                "gemini-3",
                "lane",
                &parent,
                &[one[0].clone(), item("sibling-signature-123456")],
                5
            )
            .unwrap());
        let (items, _, _) = cache.read("gemini-3", "lane", 6).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[1], normalize_item(&second).unwrap());
    }

    #[test]
    fn normalization_drops_unknown_fields_and_rejects_oversized_or_short_items() {
        let cache = AntigravityReasoningReplayCache::new();
        let (_, snapshot, _) = cache.read("gemini-3", "lane", 1).unwrap();
        assert_eq!(
            cache.replace_if_unchanged("gemini-3", "lane", &snapshot, &[item("short")], 2),
            Err(AntigravityReasoningReplayError::InvalidItems)
        );
        let valid = item("long-enough-signature-123456");
        assert!(cache
            .replace_if_unchanged("gemini-3", "lane", &snapshot, &[valid], 3)
            .unwrap());
        let (stored, _, _) = cache.read("gemini-3", "lane", 4).unwrap();
        let value: Value = serde_json::from_slice(&stored[0]).unwrap();
        assert!(value.get("ignored").is_none());
    }

    #[test]
    fn oldest_unrelated_entry_is_evicted_without_exposing_session_keys() {
        let cache = AntigravityReasoningReplayCache::with_limits(2, 1);
        let (_, a, _) = cache.read("gemini-3", "session-a-secret", 1).unwrap();
        cache
            .replace_if_unchanged(
                "gemini-3",
                "session-a-secret",
                &a,
                &[item("session-a-signature-123456")],
                2,
            )
            .unwrap();
        let _ = cache.read("gemini-3", "session-b-secret", 3).unwrap();
        let _ = cache.read("gemini-3", "session-c-secret", 4).unwrap();
        let rendered = format!("{cache:?}");
        assert!(!rendered.contains("session-a-secret"));
        let (_, _, found_a) = cache.read("gemini-3", "session-a-secret", 5).unwrap();
        assert!(!found_a);
    }
}

#[cfg(test)]
#[path = "antigravity_reasoning_replay_cache_test.rs"]
mod upstream_tests;
