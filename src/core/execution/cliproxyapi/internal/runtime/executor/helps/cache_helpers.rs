// ref: internal/runtime/executor/helps/cache_helpers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::{ClaudeIdentityKvStore, ClaudeIdentityStoreError};
use crate::internal::home::hash_key_part;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CodexCache {
    pub id: String,
    pub expire: Option<SystemTime>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredCodexCache {
    id: String,
    expire_unix_ms: Option<u64>,
}

#[derive(Default, Debug)]
pub struct CodexPromptCacheStore {
    entries: Mutex<HashMap<String, CodexCache>>,
}

impl CodexPromptCacheStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &str) -> Option<CodexCache> {
        self.get_required(None, key).ok().flatten()
    }

    pub fn get_required(
        &self,
        store: Option<&dyn ClaudeIdentityKvStore>,
        key: &str,
    ) -> Result<Option<CodexCache>, ClaudeIdentityStoreError> {
        let now = SystemTime::now();
        if let Some(store) = store {
            let Some(raw) = store.get(key)? else {
                return Ok(None);
            };
            let stored: StoredCodexCache =
                serde_json::from_slice(&raw).map_err(|_| ClaudeIdentityStoreError::InvalidJson)?;
            let cache = stored.into_cache();
            return Ok((!cache.is_expired_at(now)).then_some(cache));
        }
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        entries.retain(|_, cache| !cache.is_expired_at(now));
        Ok(entries.get(key).cloned())
    }

    pub fn set(&self, key: impl Into<String>, cache: CodexCache) {
        let _ = self.set_best_effort(None, key, cache);
    }

    pub fn set_required(
        &self,
        store: Option<&dyn ClaudeIdentityKvStore>,
        key: impl Into<String>,
        cache: CodexCache,
    ) -> Result<(), ClaudeIdentityStoreError> {
        let key = key.into();
        let Some(ttl) = cache.remaining_ttl(SystemTime::now()) else {
            return Ok(());
        };
        if let Some(store) = store {
            let raw = serde_json::to_vec(&StoredCodexCache::from_cache(&cache))
                .map_err(|_| ClaudeIdentityStoreError::InvalidJson)?;
            if !store.set(&key, &raw, ttl)? {
                return Err(ClaudeIdentityStoreError::WriteSkipped);
            }
            return Ok(());
        }
        self.entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(key, cache);
        Ok(())
    }

    pub fn set_best_effort(
        &self,
        store: Option<&dyn ClaudeIdentityKvStore>,
        key: impl Into<String>,
        cache: CodexCache,
    ) -> bool {
        self.set_required(store, key, cache).is_ok()
    }

    pub fn purge_expired(&self) {
        let now = SystemTime::now();
        self.entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .retain(|_, cache| !cache.is_expired_at(now));
    }
}

impl CodexCache {
    fn is_expired_at(&self, now: SystemTime) -> bool {
        self.expire.is_some_and(|expire| expire < now)
    }

    fn remaining_ttl(&self, now: SystemTime) -> Option<Duration> {
        match self.expire {
            Some(expire) => expire.duration_since(now).ok().filter(|ttl| !ttl.is_zero()),
            None => Some(Duration::from_secs(60 * 60)),
        }
    }
}

impl StoredCodexCache {
    fn from_cache(cache: &CodexCache) -> Self {
        Self {
            id: cache.id.clone(),
            expire_unix_ms: cache.expire.and_then(|expire| {
                expire
                    .duration_since(UNIX_EPOCH)
                    .ok()
                    .and_then(|duration| u64::try_from(duration.as_millis()).ok())
            }),
        }
    }

    fn into_cache(self) -> CodexCache {
        CodexCache {
            id: self.id,
            expire: self
                .expire_unix_ms
                .map(|millis| UNIX_EPOCH + Duration::from_millis(millis)),
        }
    }
}

pub fn codex_prompt_cache_key(model_name: &str, user_scope: &str) -> String {
    format!(
        "cpa:codex:prompt-cache:{}:{}",
        hash_key_part(model_name),
        hash_key_part(user_scope)
    )
}
