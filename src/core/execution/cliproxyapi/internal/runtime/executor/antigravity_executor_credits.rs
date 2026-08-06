// ref: internal/runtime/executor/antigravity_executor_credits.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Credits, 429 classification and short-cooldown state for Antigravity.
//! Upstream package globals and Home-KV discovery are replaced by one injected
//! authority; this keeps multiple CTOX gateway hosts isolated.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::helps::parse_retry_delay;

pub const ANTIGRAVITY_CREDIT_TYPE: &str = "GOOGLE_ONE_AI";
pub const ANTIGRAVITY_INSTANT_RETRY_THRESHOLD: Duration = Duration::from_secs(2);
pub const ANTIGRAVITY_SHORT_COOLDOWN_THRESHOLD: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Antigravity429Category {
    #[default]
    Unknown,
    RateLimited,
    QuotaExhausted,
    SoftRateLimit,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Antigravity429DecisionKind {
    #[default]
    SoftRetry,
    InstantRetrySameAuth,
    ShortCooldownSwitchAuth,
    FullQuotaExhausted,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Antigravity429Decision {
    pub kind: Antigravity429DecisionKind,
    pub retry_after: Option<Duration>,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AntigravityCreditsBalance {
    pub credit_amount: f64,
    pub min_credit_amount: f64,
    pub paid_tier_id: String,
    pub known: bool,
}

impl AntigravityCreditsBalance {
    #[must_use]
    pub fn available(&self) -> bool {
        self.known && self.credit_amount >= self.min_credit_amount
    }
}

pub trait AntigravityCreditsStore: Send + Sync {
    fn load_balance(&self, auth_id: &str) -> Result<Option<AntigravityCreditsBalance>, String>;
    fn store_balance(
        &self,
        auth_id: &str,
        balance: &AntigravityCreditsBalance,
    ) -> Result<(), String>;
    fn load_cooldown_until_ms(&self, auth_id: &str, model: &str) -> Result<Option<i64>, String>;
    fn store_cooldown_until_ms(
        &self,
        auth_id: &str,
        model: &str,
        until_ms: i64,
    ) -> Result<(), String>;
    /// Durable SET-NX used to throttle balance refresh across processes.
    fn acquire_refresh_lease(
        &self,
        auth_id: &str,
        now_ms: i64,
        ttl: Duration,
    ) -> Result<bool, String>;
}

#[derive(Default)]
pub struct MemoryAntigravityCreditsStore {
    state: Mutex<MemoryCreditsState>,
}

#[derive(Default)]
struct MemoryCreditsState {
    balances: HashMap<String, AntigravityCreditsBalance>,
    cooldowns: HashMap<(String, String), i64>,
    leases: HashMap<String, i64>,
}

impl AntigravityCreditsStore for MemoryAntigravityCreditsStore {
    fn load_balance(&self, auth_id: &str) -> Result<Option<AntigravityCreditsBalance>, String> {
        Ok(self.state.lock().unwrap().balances.get(auth_id).cloned())
    }
    fn store_balance(
        &self,
        auth_id: &str,
        balance: &AntigravityCreditsBalance,
    ) -> Result<(), String> {
        self.state
            .lock()
            .unwrap()
            .balances
            .insert(auth_id.into(), balance.clone());
        Ok(())
    }
    fn load_cooldown_until_ms(&self, auth_id: &str, model: &str) -> Result<Option<i64>, String> {
        Ok(self
            .state
            .lock()
            .unwrap()
            .cooldowns
            .get(&(auth_id.into(), model.into()))
            .copied())
    }
    fn store_cooldown_until_ms(
        &self,
        auth_id: &str,
        model: &str,
        until_ms: i64,
    ) -> Result<(), String> {
        self.state
            .lock()
            .unwrap()
            .cooldowns
            .insert((auth_id.into(), model.into()), until_ms);
        Ok(())
    }
    fn acquire_refresh_lease(
        &self,
        auth_id: &str,
        now_ms: i64,
        ttl: Duration,
    ) -> Result<bool, String> {
        let mut state = self.state.lock().unwrap();
        if state
            .leases
            .get(auth_id)
            .is_some_and(|until| *until > now_ms)
        {
            return Ok(false);
        }
        let ttl = i64::try_from(ttl.as_millis()).unwrap_or(i64::MAX);
        state
            .leases
            .insert(auth_id.into(), now_ms.saturating_add(ttl));
        Ok(true)
    }
}

pub struct AntigravityCreditsController {
    store: Arc<dyn AntigravityCreditsStore>,
    refresh_lease_ttl: Duration,
}

impl AntigravityCreditsController {
    #[must_use]
    pub fn new(store: Arc<dyn AntigravityCreditsStore>, refresh_lease_ttl: Duration) -> Self {
        Self {
            store,
            refresh_lease_ttl,
        }
    }

    pub fn has_credits(&self, auth_id: &str) -> Result<bool, AntigravityCreditsError> {
        if auth_id.trim().is_empty() {
            return Ok(false);
        }
        Ok(self
            .store
            .load_balance(auth_id)
            .map_err(AntigravityCreditsError::Store)?
            .is_none_or(|balance| balance.available()))
    }

    pub fn store_balance(
        &self,
        auth_id: &str,
        balance: &AntigravityCreditsBalance,
    ) -> Result<(), AntigravityCreditsError> {
        self.store
            .store_balance(auth_id, balance)
            .map_err(AntigravityCreditsError::Store)
    }

    pub fn cooldown_remaining(
        &self,
        auth_id: &str,
        model: &str,
        now_ms: i64,
    ) -> Result<Option<Duration>, AntigravityCreditsError> {
        let until = self
            .store
            .load_cooldown_until_ms(auth_id, model)
            .map_err(AntigravityCreditsError::Store)?;
        Ok(until
            .filter(|until| *until > now_ms)
            .map(|until| Duration::from_millis(u64::try_from(until - now_ms).unwrap_or(u64::MAX))))
    }

    pub fn mark_short_cooldown(
        &self,
        auth_id: &str,
        model: &str,
        now_ms: i64,
        duration: Duration,
    ) -> Result<(), AntigravityCreditsError> {
        let millis = i64::try_from(duration.as_millis()).unwrap_or(i64::MAX);
        self.store
            .store_cooldown_until_ms(auth_id, model, now_ms.saturating_add(millis))
            .map_err(AntigravityCreditsError::Store)
    }

    pub fn acquire_refresh_lease(
        &self,
        auth_id: &str,
        now_ms: i64,
    ) -> Result<bool, AntigravityCreditsError> {
        self.store
            .acquire_refresh_lease(auth_id, now_ms, self.refresh_lease_ttl)
            .map_err(AntigravityCreditsError::Store)
    }
}

impl fmt::Debug for AntigravityCreditsController {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AntigravityCreditsController")
            .field("store", &"injected")
            .field("refresh_lease_ttl", &self.refresh_lease_ttl)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AntigravityCreditsError {
    Store(String),
    InvalidJson,
}
impl fmt::Display for AntigravityCreditsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(formatter, "Antigravity credits store failed: {error}"),
            Self::InvalidJson => formatter.write_str("Antigravity credits payload is invalid"),
        }
    }
}
impl std::error::Error for AntigravityCreditsError {}

#[must_use]
pub fn inject_enabled_credit_types(payload: &[u8]) -> Option<Vec<u8>> {
    let mut root = serde_json::from_slice::<Value>(payload).ok()?;
    root.as_object_mut()?.insert(
        "enabledCreditTypes".into(),
        Value::Array(vec![Value::String(ANTIGRAVITY_CREDIT_TYPE.into())]),
    );
    serde_json::to_vec(&root).ok()
}

#[must_use]
pub fn classify_antigravity_429(body: &[u8]) -> Antigravity429Category {
    match decide_antigravity_429(body).kind {
        Antigravity429DecisionKind::InstantRetrySameAuth
        | Antigravity429DecisionKind::ShortCooldownSwitchAuth => {
            Antigravity429Category::RateLimited
        }
        Antigravity429DecisionKind::FullQuotaExhausted => Antigravity429Category::QuotaExhausted,
        Antigravity429DecisionKind::SoftRetry => Antigravity429Category::SoftRateLimit,
    }
}

#[must_use]
pub fn decide_antigravity_429(body: &[u8]) -> Antigravity429Decision {
    let retry_after = parse_retry_delay(body)
        .ok()
        .and_then(|delay| delay.to_std().ok());
    let root = serde_json::from_slice::<Value>(body).unwrap_or(Value::Null);
    if root
        .pointer("/error/status")
        .and_then(Value::as_str)
        .is_none_or(|status| !status.eq_ignore_ascii_case("RESOURCE_EXHAUSTED"))
    {
        return Antigravity429Decision {
            retry_after,
            ..Antigravity429Decision::default()
        };
    }
    let reason = root
        .pointer("/error/details")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|detail| {
            detail.get("@type").and_then(Value::as_str)
                == Some("type.googleapis.com/google.rpc.ErrorInfo")
        })
        .and_then(|detail| detail.get("reason"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned();
    let kind = if reason.eq_ignore_ascii_case("QUOTA_EXHAUSTED") || contains_quota_exhausted(body) {
        Antigravity429DecisionKind::FullQuotaExhausted
    } else if reason.eq_ignore_ascii_case("RATE_LIMIT_EXCEEDED") {
        retry_after.map_or(Antigravity429DecisionKind::SoftRetry, |delay| {
            if delay < ANTIGRAVITY_INSTANT_RETRY_THRESHOLD {
                Antigravity429DecisionKind::InstantRetrySameAuth
            } else if delay < ANTIGRAVITY_SHORT_COOLDOWN_THRESHOLD {
                Antigravity429DecisionKind::ShortCooldownSwitchAuth
            } else {
                Antigravity429DecisionKind::FullQuotaExhausted
            }
        })
    } else {
        Antigravity429DecisionKind::SoftRetry
    };
    Antigravity429Decision {
        kind,
        retry_after,
        reason,
    }
}

#[must_use]
pub fn antigravity_should_retry_no_capacity(status: u16, body: &[u8]) -> bool {
    status == 503
        && String::from_utf8_lossy(body)
            .to_ascii_lowercase()
            .contains("no capacity")
}

#[must_use]
pub fn antigravity_should_retry_transient_429(status: u16, body: &[u8]) -> bool {
    status == 429 && decide_antigravity_429(body).kind == Antigravity429DecisionKind::SoftRetry
}

#[must_use]
pub fn parse_meta_float(
    metadata: &std::collections::BTreeMap<String, Value>,
    key: &str,
) -> Option<f64> {
    metadata.get(key).and_then(|value| {
        value
            .as_f64()
            .or_else(|| value.as_str().and_then(|value| value.trim().parse().ok()))
    })
}

fn contains_quota_exhausted(body: &[u8]) -> bool {
    let body = String::from_utf8_lossy(body).to_ascii_lowercase();
    body.contains("quota_exhausted") || body.contains("quota exhausted")
}
