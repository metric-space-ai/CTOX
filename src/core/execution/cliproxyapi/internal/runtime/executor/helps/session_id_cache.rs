// ref: internal/runtime/executor/helps/session_id_cache.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::internal::home::hash_key_part;

use super::{ClaudeIdentityKvStore, ClaudeIdentityStoreError};

pub const SESSION_ID_TTL: Duration = Duration::from_secs(60 * 60);
pub const DEFAULT_SESSION_ID_CACHE_CAPACITY: usize = 4_096;

pub trait SessionIdClock: Send + Sync {
    fn now(&self) -> Instant;
}

#[derive(Debug)]
struct SystemSessionIdClock;

impl SessionIdClock for SystemSessionIdClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

#[derive(Clone, Debug)]
struct SessionIdCacheEntry {
    value: String,
    expire: Instant,
    touched: u64,
}

#[derive(Debug, Default)]
struct SessionIdCacheState {
    entries: HashMap<String, SessionIdCacheEntry>,
    generation: u64,
}

/// Instance-owned replacement for upstream's process-global session map.
///
/// A supplied KV store is authoritative (Home/host mode); without one the
/// cache is local to this instance. Expired entries are removed
/// opportunistically, avoiding an immortal background cleanup task. The
/// explicit capacity bounds attacker-controlled API-key cardinality.
pub struct SessionIdCache {
    state: Mutex<SessionIdCacheState>,
    clock: Arc<dyn SessionIdClock>,
    capacity: usize,
}

impl SessionIdCache {
    pub fn new() -> Self {
        Self::with_clock_and_capacity(
            Arc::new(SystemSessionIdClock),
            DEFAULT_SESSION_ID_CACHE_CAPACITY,
        )
    }

    pub fn with_clock_and_capacity(clock: Arc<dyn SessionIdClock>, capacity: usize) -> Self {
        Self {
            state: Mutex::new(SessionIdCacheState::default()),
            clock,
            capacity: capacity.max(1),
        }
    }

    /// Best-effort equivalent of upstream `CachedSessionID`.
    pub fn cached_session_id(
        &self,
        store: Option<&dyn ClaudeIdentityKvStore>,
        api_key: &str,
    ) -> String {
        self.cached_session_id_required(store, api_key)
            .unwrap_or_else(|_| Uuid::new_v4().to_string())
    }

    /// Returns a stable UUID per API key and renews its one-hour TTL.
    pub fn cached_session_id_required(
        &self,
        store: Option<&dyn ClaudeIdentityKvStore>,
        api_key: &str,
    ) -> Result<String, SessionIdCacheError> {
        if api_key.is_empty() {
            return Ok(Uuid::new_v4().to_string());
        }
        if let Some(store) = store {
            return cached_session_id_from_store(store, api_key);
        }

        let key = session_id_cache_key(api_key);
        let now = self.clock.now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.entries.retain(|_, entry| entry.expire > now);
        state.generation = state.generation.wrapping_add(1);
        let generation = state.generation;
        if let Some(entry) = state.entries.get_mut(&key) {
            if !entry.value.is_empty() && entry.expire > now {
                entry.expire = now + SESSION_ID_TTL;
                entry.touched = generation;
                return Ok(entry.value.clone());
            }
        }

        while state.entries.len() >= self.capacity {
            let Some(oldest) = state
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.touched)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            state.entries.remove(&oldest);
        }
        let value = Uuid::new_v4().to_string();
        state.entries.insert(
            key,
            SessionIdCacheEntry {
                value: value.clone(),
                expire: now + SESSION_ID_TTL,
                touched: generation,
            },
        );
        Ok(value)
    }

    pub fn purge_expired(&self) {
        let now = self.clock.now();
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entries
            .retain(|_, entry| entry.expire > now);
    }

    #[cfg(test)]
    pub(crate) fn len_for_test(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entries
            .len()
    }

    #[cfg(test)]
    pub(crate) fn poison_for_test(&self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = self.state.lock().unwrap();
            panic!("poison session cache");
        }));
    }
}

impl Default for SessionIdCache {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SessionIdCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionIdCache")
            .field("capacity", &self.capacity)
            .field(
                "entries",
                &self.state.lock().map(|state| state.entries.len()).ok(),
            )
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionIdCacheError {
    Store(ClaudeIdentityStoreError),
    InvalidUtf8,
    MissingAfterSet,
}

impl fmt::Display for SessionIdCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(formatter, "session ID store failed: {error}"),
            Self::InvalidUtf8 => formatter.write_str("session ID store value is not UTF-8"),
            Self::MissingAfterSet => formatter.write_str("home kv session id missing after set"),
        }
    }
}

impl std::error::Error for SessionIdCacheError {}

impl From<ClaudeIdentityStoreError> for SessionIdCacheError {
    fn from(value: ClaudeIdentityStoreError) -> Self {
        Self::Store(value)
    }
}

fn cached_session_id_from_store(
    store: &dyn ClaudeIdentityKvStore,
    api_key: &str,
) -> Result<String, SessionIdCacheError> {
    let key = claude_session_id_kv_key(api_key);
    if let Some(raw) = store.get(&key)? {
        let value = String::from_utf8(raw).map_err(|_| SessionIdCacheError::InvalidUtf8)?;
        let value = value.trim();
        if !value.is_empty() {
            store.expire(&key, SESSION_ID_TTL)?;
            return Ok(value.to_owned());
        }
    }

    let new_id = Uuid::new_v4().to_string();
    let _ = store.set_nx(&key, new_id.as_bytes(), SESSION_ID_TTL)?;
    let Some(raw) = store.get(&key)? else {
        return Err(SessionIdCacheError::MissingAfterSet);
    };
    let value = String::from_utf8(raw).map_err(|_| SessionIdCacheError::InvalidUtf8)?;
    let value = value.trim();
    if value.is_empty() {
        Err(SessionIdCacheError::MissingAfterSet)
    } else {
        Ok(value.to_owned())
    }
}

pub fn session_id_cache_key(api_key: &str) -> String {
    format!("{:x}", Sha256::digest(api_key.as_bytes()))
}

pub fn claude_session_id_kv_key(api_key: &str) -> String {
    format!("cpa:claude:session-id:{}", hash_key_part(api_key))
}
