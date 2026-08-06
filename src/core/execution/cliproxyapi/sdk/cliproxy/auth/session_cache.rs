// ref: sdk/cliproxy/auth/session_cache.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const MAX_STABLE_SESSION_ALIASES: usize = 64;
const DEFAULT_TTL: Duration = Duration::from_secs(30 * 60);

pub trait SessionClock: Send + Sync {
    fn now_ms(&self) -> i64;
}

#[derive(Debug)]
pub struct SystemSessionClock;
impl SessionClock for SystemSessionClock {
    fn now_ms(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(i64::MAX as u128) as i64
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SessionEntry {
    auth_id: String,
    expires_at_ms: i64,
    aliases: Vec<String>,
}

/// TTL session bindings with an injected clock. Cleanup is deterministic and
/// occurs on cache operations; the cache owns no background process authority.
pub struct SessionCache {
    entries: Mutex<BTreeMap<String, SessionEntry>>,
    ttl_ms: i64,
    clock: Arc<dyn SessionClock>,
}

impl SessionCache {
    #[must_use]
    pub fn new(ttl: Duration, clock: Arc<dyn SessionClock>) -> Self {
        let ttl = if ttl.is_zero() { DEFAULT_TTL } else { ttl };
        Self {
            entries: Mutex::new(BTreeMap::new()),
            ttl_ms: ttl.as_millis().min(i64::MAX as u128) as i64,
            clock,
        }
    }

    #[must_use]
    pub fn get(&self, session_id: &str) -> Option<String> {
        let session_id = session_id.trim();
        if session_id.is_empty() {
            return None;
        }
        let now = self.clock.now_ms();
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = entries.get(session_id)?.clone();
        if now < entry.expires_at_ms {
            return Some(entry.auth_id);
        }
        remove_group(&mut entries, &entry);
        None
    }

    #[must_use]
    pub fn get_and_refresh(&self, session_id: &str) -> Option<String> {
        let session_id = session_id.trim();
        if session_id.is_empty() {
            return None;
        }
        let now = self.clock.now_ms();
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = entries.get(session_id)?.clone();
        if now >= entry.expires_at_ms {
            remove_group(&mut entries, &entry);
            return None;
        }
        let aliases = compact_session_aliases(merge_aliases(
            [session_id],
            entry.aliases.iter().map(String::as_str),
        ));
        replace_groups(
            &mut entries,
            &entry.auth_id,
            now.saturating_add(self.ttl_ms),
            aliases,
            std::slice::from_ref(&entry),
        );
        Some(entry.auth_id)
    }

    pub fn set(&self, session_id: &str, auth_id: &str) {
        self.set_aliases(auth_id, [session_id]);
    }

    pub fn set_aliases<'a>(&self, auth_id: &str, session_ids: impl IntoIterator<Item = &'a str>) {
        let auth_id = auth_id.trim();
        if auth_id.is_empty() {
            return;
        }
        let now = self.clock.now_ms();
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut aliases = merge_aliases([], session_ids);
        let mut previous = Vec::new();
        for alias in aliases.clone() {
            if let Some(entry) = entries.get(&alias).cloned() {
                if now >= entry.expires_at_ms {
                    remove_group(&mut entries, &entry);
                } else {
                    aliases = merge_aliases(
                        aliases.iter().map(String::as_str),
                        entry.aliases.iter().map(String::as_str),
                    );
                    previous.push(entry);
                }
            }
        }
        aliases = compact_session_aliases(aliases);
        if !aliases.is_empty() {
            replace_groups(
                &mut entries,
                auth_id,
                now.saturating_add(self.ttl_ms),
                aliases,
                &previous,
            );
        }
    }

    pub fn invalidate(&self, session_id: &str) {
        if session_id.is_empty() {
            return;
        }
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(entry) = entries.remove(session_id) else {
            return;
        };
        for alias in entry
            .aliases
            .iter()
            .filter(|alias| alias.as_str() != session_id)
        {
            if let Some(current) = entries.get_mut(alias) {
                if current.auth_id == entry.auth_id {
                    current.aliases.retain(|candidate| candidate != session_id);
                }
            }
        }
    }

    pub fn invalidate_auth(&self, auth_id: &str) {
        if auth_id.is_empty() {
            return;
        }
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|_, entry| entry.auth_id != auth_id);
    }

    pub fn cleanup(&self) {
        let now = self.clock.now_ms();
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|_, entry| now < entry.expires_at_ms);
    }

    pub fn stop(&self) {}
}

impl std::fmt::Debug for SessionCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionCache")
            .field("ttl_ms", &self.ttl_ms)
            .finish_non_exhaustive()
    }
}

fn replace_groups(
    entries: &mut BTreeMap<String, SessionEntry>,
    auth_id: &str,
    expires_at_ms: i64,
    aliases: Vec<String>,
    previous: &[SessionEntry],
) {
    for entry in previous {
        remove_group(entries, entry);
    }
    let entry = SessionEntry {
        auth_id: auth_id.to_owned(),
        expires_at_ms,
        aliases: aliases.clone(),
    };
    for alias in aliases {
        entries.insert(alias, entry.clone());
    }
}

fn remove_group(entries: &mut BTreeMap<String, SessionEntry>, entry: &SessionEntry) {
    for alias in &entry.aliases {
        if entries.get(alias).is_some_and(|current| current == entry) {
            entries.remove(alias);
        }
    }
}

fn merge_aliases<'a>(
    existing: impl IntoIterator<Item = &'a str>,
    candidates: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    let mut seen = BTreeSet::new();
    existing
        .into_iter()
        .chain(candidates)
        .filter_map(|alias| {
            let alias = alias.trim();
            (!alias.is_empty() && seen.insert(alias.to_owned())).then(|| alias.to_owned())
        })
        .collect()
}

#[must_use]
pub fn compact_session_aliases(aliases: Vec<String>) -> Vec<String> {
    compact_aliases_with(aliases, is_local_prompt_cache_session_alias)
}

#[must_use]
pub fn compact_home_session_aliases(aliases: Vec<String>) -> Vec<String> {
    compact_aliases_with(aliases, |alias| alias.starts_with("pck:"))
}

fn compact_aliases_with(aliases: Vec<String>, prompt: impl Fn(&str) -> bool) -> Vec<String> {
    let mut has_prompt = false;
    let mut stable = 0;
    aliases
        .into_iter()
        .filter(|alias| {
            if prompt(alias) {
                if has_prompt {
                    false
                } else {
                    has_prompt = true;
                    true
                }
            } else if stable >= MAX_STABLE_SESSION_ALIASES {
                false
            } else {
                stable += 1;
                true
            }
        })
        .collect()
}

fn is_local_prompt_cache_session_alias(alias: &str) -> bool {
    alias.starts_with("pck:")
        || alias
            .split_once("::")
            .is_some_and(|(_, suffix)| suffix.starts_with("pck:"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, Ordering};
    struct Clock(AtomicI64);
    impl SessionClock for Clock {
        fn now_ms(&self) -> i64 {
            self.0.load(Ordering::SeqCst)
        }
    }

    #[test]
    fn aliases_refresh_expire_and_invalidate_as_one_group() {
        let clock = Arc::new(Clock(AtomicI64::new(100)));
        let cache = SessionCache::new(Duration::from_millis(10), clock.clone());
        cache.set_aliases("auth", ["stable", "pck:first", "pck:second"]);
        assert_eq!(cache.get("stable").as_deref(), Some("auth"));
        assert!(cache.get("pck:second").is_none());
        clock.0.store(105, Ordering::SeqCst);
        assert_eq!(cache.get_and_refresh("pck:first").as_deref(), Some("auth"));
        clock.0.store(111, Ordering::SeqCst);
        assert_eq!(cache.get("stable").as_deref(), Some("auth"));
        cache.invalidate("stable");
        assert!(cache.get("stable").is_none());
        assert_eq!(cache.get("pck:first").as_deref(), Some("auth"));
        cache.invalidate_auth("auth");
        assert!(cache.get("pck:first").is_none());
    }
}
