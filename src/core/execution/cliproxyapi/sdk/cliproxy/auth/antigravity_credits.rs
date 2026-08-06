// ref: sdk/cliproxy/auth/antigravity_credits.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::Auth;

const CREDIT_HINT_TTL: Duration = Duration::from_secs(30 * 60);
const CREDIT_HINT_PREFIX: &str = "cpa:antigravity:credits-hint:";

/// Request-scoped execution option replacing Go context values with a typed
/// field that can cross the Rust harness boundary explicitly.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AntigravityCreditsRequest {
    pub enabled: bool,
}

impl AntigravityCreditsRequest {
    #[must_use]
    pub const fn requested() -> Self {
        Self { enabled: true }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct AntigravityCreditsHint {
    pub known: bool,
    pub available: bool,
    pub credit_amount: f64,
    pub min_credit_amount: f64,
    pub paid_tier_id: String,
    pub updated_at: Option<DateTime<Utc>>,
}

pub trait AntigravityCreditsClock: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
}

pub trait AntigravityCreditsStore: Send + Sync {
    fn set(
        &self,
        key: &str,
        hint: &AntigravityCreditsHint,
        ttl: Duration,
    ) -> Result<(), AntigravityCreditsStoreError>;

    fn get(
        &self,
        key: &str,
    ) -> Result<Option<AntigravityCreditsHint>, AntigravityCreditsStoreError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AntigravityCreditsStoreError {
    Read,
    Write,
}

impl fmt::Display for AntigravityCreditsStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Read => "could not read antigravity credits hint",
            Self::Write => "could not write antigravity credits hint",
        })
    }
}

impl std::error::Error for AntigravityCreditsStoreError {}

/// Store-backed credits hints. The SDK owns neither an ambient singleton nor
/// a fallback memory cache; CTOX chooses the durable implementation.
pub struct AntigravityCreditsHints {
    store: Arc<dyn AntigravityCreditsStore>,
    clock: Arc<dyn AntigravityCreditsClock>,
}

impl AntigravityCreditsHints {
    #[must_use]
    pub fn new(
        store: Arc<dyn AntigravityCreditsStore>,
        clock: Arc<dyn AntigravityCreditsClock>,
    ) -> Self {
        Self { store, clock }
    }

    pub fn set(
        &self,
        auth_id: &str,
        mut hint: AntigravityCreditsHint,
    ) -> Result<bool, AntigravityCreditsStoreError> {
        let Some(key) = hint_key(auth_id) else {
            return Ok(false);
        };
        if hint.updated_at.is_none() {
            hint.updated_at = Some(self.clock.now());
        }
        self.store.set(&key, &hint, CREDIT_HINT_TTL)?;
        Ok(true)
    }

    pub fn get(
        &self,
        auth_id: &str,
    ) -> Result<Option<AntigravityCreditsHint>, AntigravityCreditsStoreError> {
        let Some(key) = hint_key(auth_id) else {
            return Ok(None);
        };
        self.store.get(&key)
    }

    pub fn has_known(&self, auth_id: &str) -> Result<bool, AntigravityCreditsStoreError> {
        Ok(self.get(auth_id)?.is_some_and(|hint| hint.known))
    }

    pub fn available_for_model(
        &self,
        auth: &Auth,
        model: &str,
    ) -> Result<bool, AntigravityCreditsStoreError> {
        if !auth.provider.trim().eq_ignore_ascii_case("antigravity")
            || !model.trim().to_ascii_lowercase().contains("claude")
        {
            return Ok(false);
        }
        Ok(self
            .get(&auth.id)?
            .is_some_and(|hint| hint.known && hint.available))
    }

    /// Orders eligible Antigravity credentials for a Claude credits fallback.
    /// Known available balances are attempted first, unknown balances remain a
    /// conservative fallback, and known unavailable balances are excluded.
    pub fn rank_candidates(
        &self,
        auths: &[Auth],
        model: &str,
    ) -> Result<Vec<Auth>, AntigravityCreditsStoreError> {
        if !model.trim().to_ascii_lowercase().contains("claude") {
            return Ok(Vec::new());
        }
        let mut known = Vec::new();
        let mut unknown = Vec::new();
        for auth in auths {
            if !auth.provider.trim().eq_ignore_ascii_case("antigravity")
                || auth.disabled
                || auth.unavailable
            {
                continue;
            }
            match self.get(&auth.id)? {
                Some(hint) if hint.known && hint.available => known.push(auth.clone()),
                Some(hint) if hint.known => {}
                _ => unknown.push(auth.clone()),
            }
        }
        known.extend(unknown);
        Ok(known)
    }
}

impl fmt::Debug for AntigravityCreditsHints {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AntigravityCreditsHints")
            .finish_non_exhaustive()
    }
}

fn hint_key(auth_id: &str) -> Option<String> {
    let auth_id = auth_id.trim();
    (!auth_id.is_empty()).then(|| format!("{CREDIT_HINT_PREFIX}{auth_id}"))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use chrono::TimeZone;

    use super::*;

    #[derive(Default)]
    struct Store(Mutex<BTreeMap<String, AntigravityCreditsHint>>);

    impl AntigravityCreditsStore for Store {
        fn set(
            &self,
            key: &str,
            hint: &AntigravityCreditsHint,
            _ttl: Duration,
        ) -> Result<(), AntigravityCreditsStoreError> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(key.to_owned(), hint.clone());
            Ok(())
        }

        fn get(
            &self,
            key: &str,
        ) -> Result<Option<AntigravityCreditsHint>, AntigravityCreditsStoreError> {
            Ok(self
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(key)
                .cloned())
        }
    }

    struct Clock;
    impl AntigravityCreditsClock for Clock {
        fn now(&self) -> DateTime<Utc> {
            Utc.timestamp_opt(1_700_000_000, 0).single().unwrap()
        }
    }

    #[test]
    fn persists_and_routes_known_claude_credits_only() {
        let hints = AntigravityCreditsHints::new(Arc::new(Store::default()), Arc::new(Clock));
        assert_eq!(
            hints.set(
                "ag-1",
                AntigravityCreditsHint {
                    known: true,
                    available: true,
                    ..AntigravityCreditsHint::default()
                }
            ),
            Ok(true)
        );
        let mut auth = Auth::default();
        auth.id = "ag-1".to_owned();
        auth.provider = "antigravity".to_owned();
        assert_eq!(hints.available_for_model(&auth, "claude-sonnet"), Ok(true));
        assert_eq!(hints.available_for_model(&auth, "gemini-flash"), Ok(false));
        assert!(hints.get("ag-1").unwrap().unwrap().updated_at.is_some());
    }
}
