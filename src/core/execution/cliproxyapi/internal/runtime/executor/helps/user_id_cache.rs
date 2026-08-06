// ref: internal/runtime/executor/helps/user_id_cache.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::internal::home::hash_key_part;

use super::cloak_utils::{
    generate_fake_user_id, generate_fake_user_id_with_session_id, is_valid_user_id,
};
use super::session_id_cache::SessionIdCache;

pub const USER_ID_TTL: Duration = Duration::from_secs(60 * 60);

#[derive(Clone, Debug)]
struct UserIdCacheEntry {
    value: String,
    expire: Instant,
}

#[derive(Default, Debug)]
pub struct UserIdCache {
    entries: Mutex<HashMap<String, UserIdCacheEntry>>,
    session_ids: SessionIdCache,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClaudeIdentityStoreError {
    Backend(String),
    InvalidUtf8,
    MissingAfterSet,
    WriteSkipped,
    InvalidJson,
    Session(String),
}

impl fmt::Display for ClaudeIdentityStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(message) => formatter.write_str(message),
            Self::InvalidUtf8 => formatter.write_str("identity store value is not UTF-8"),
            Self::MissingAfterSet => formatter.write_str("home kv identity missing after set"),
            Self::WriteSkipped => formatter.write_str("home kv identity write skipped"),
            Self::InvalidJson => formatter.write_str("home kv identity value is invalid JSON"),
            Self::Session(message) => write!(formatter, "session identity failed: {message}"),
        }
    }
}

impl std::error::Error for ClaudeIdentityStoreError {}

/// Explicit durable-state boundary shared by the Claude identity helpers.
/// Supplying a store makes it authoritative; omitting it selects instance-local
/// state owned by the caller.
pub trait ClaudeIdentityKvStore: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, ClaudeIdentityStoreError>;
    fn set(&self, key: &str, value: &[u8], ttl: Duration)
        -> Result<bool, ClaudeIdentityStoreError>;
    fn set_nx(
        &self,
        key: &str,
        value: &[u8],
        ttl: Duration,
    ) -> Result<bool, ClaudeIdentityStoreError>;
    fn expire(&self, key: &str, ttl: Duration) -> Result<bool, ClaudeIdentityStoreError>;
}

impl UserIdCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cached_user_id(&self, api_key: &str) -> String {
        self.cached_user_id_required(None, api_key)
            .unwrap_or_else(|_| generate_fake_user_id())
    }

    pub fn cached_session_id(
        &self,
        store: Option<&dyn ClaudeIdentityKvStore>,
        api_key: &str,
    ) -> String {
        self.session_ids.cached_session_id(store, api_key)
    }

    pub fn cached_user_id_required(
        &self,
        store: Option<&dyn ClaudeIdentityKvStore>,
        api_key: &str,
    ) -> Result<String, ClaudeIdentityStoreError> {
        let new_user_id = || {
            self.session_ids
                .cached_session_id_required(store, api_key)
                .map(|session_id| generate_fake_user_id_with_session_id(&session_id))
                .map_err(|error| ClaudeIdentityStoreError::Session(error.to_string()))
        };
        if api_key.is_empty() {
            return new_user_id();
        }
        if let Some(store) = store {
            return cached_user_id_from_store(store, api_key, new_user_id);
        }

        let key = user_id_cache_key(api_key);
        let now = Instant::now();
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        entries.retain(|_, entry| entry.expire > now);
        if let Some(entry) = entries.get_mut(&key) {
            if is_valid_user_id(&entry.value) {
                entry.expire = now + USER_ID_TTL;
                return Ok(entry.value.clone());
            }
        }
        let value = new_user_id()?;
        entries.insert(
            key,
            UserIdCacheEntry {
                value: value.clone(),
                expire: now + USER_ID_TTL,
            },
        );
        Ok(value)
    }

    pub fn purge_expired(&self) {
        let now = Instant::now();
        self.entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .retain(|_, entry| entry.expire > now);
    }

    #[cfg(test)]
    pub(crate) fn expire_for_test(&self, api_key: &str) {
        if let Some(entry) = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get_mut(&user_id_cache_key(api_key))
        {
            entry.expire = Instant::now() - Duration::from_secs(1);
        }
    }

    #[cfg(test)]
    pub(crate) fn remaining_ttl_for_test(&self, api_key: &str) -> Option<Duration> {
        self.entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&user_id_cache_key(api_key))
            .and_then(|entry| entry.expire.checked_duration_since(Instant::now()))
    }
}

fn cached_user_id_from_store(
    store: &dyn ClaudeIdentityKvStore,
    api_key: &str,
    new_user_id: impl FnOnce() -> Result<String, ClaudeIdentityStoreError>,
) -> Result<String, ClaudeIdentityStoreError> {
    let key = claude_user_id_kv_key(api_key);
    if let Some(raw) = store.get(&key)? {
        let value = String::from_utf8(raw).map_err(|_| ClaudeIdentityStoreError::InvalidUtf8)?;
        let value = value.trim();
        if is_valid_user_id(value) {
            store.expire(&key, USER_ID_TTL)?;
            return Ok(value.to_owned());
        }
    }
    let new_id = new_user_id()?;
    store.set_nx(&key, new_id.as_bytes(), USER_ID_TTL)?;
    let Some(raw) = store.get(&key)? else {
        return Err(ClaudeIdentityStoreError::MissingAfterSet);
    };
    let value = String::from_utf8(raw).map_err(|_| ClaudeIdentityStoreError::InvalidUtf8)?;
    let value = value.trim();
    if is_valid_user_id(value) {
        Ok(value.to_owned())
    } else {
        Err(ClaudeIdentityStoreError::MissingAfterSet)
    }
}

pub fn user_id_cache_key(api_key: &str) -> String {
    format!("{:x}", Sha256::digest(api_key.as_bytes()))
}

pub fn claude_user_id_kv_key(api_key: &str) -> String {
    format!("cpa:claude:user-id:{}", hash_key_part(api_key))
}
