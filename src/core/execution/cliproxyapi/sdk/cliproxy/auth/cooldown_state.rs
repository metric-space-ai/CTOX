// ref: sdk/cliproxy/auth/cooldown_state.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

use serde::{Deserialize, Serialize};

/// Persisted provider-neutral cooldown state for one account or account/model pair.
///
/// CTOX stores epoch milliseconds instead of Go's `time.Time` JSON encoding so
/// the portable crate has no wall-clock or timezone dependency at its storage
/// boundary. Credential material and auth-file paths are deliberately absent.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CooldownStateRecord {
    pub provider: String,
    pub auth_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_retry_after_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reason: String,
    #[serde(default)]
    pub quota: CooldownQuotaState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<CooldownErrorState>,
    pub updated_at_ms: i64,
}

impl CooldownStateRecord {
    pub fn validate(&self) -> Result<(), CooldownStoreError> {
        if self.provider.trim().is_empty() || self.auth_id.trim().is_empty() {
            return Err(CooldownStoreError::InvalidRecord);
        }
        if self
            .model
            .as_deref()
            .is_some_and(|model| model.trim().is_empty())
        {
            return Err(CooldownStoreError::InvalidRecord);
        }
        Ok(())
    }

    /// Mirrors upstream `availabilityBlock`: an expired timestamp unblocks the
    /// account, and an unavailable/error state without a recovery timestamp is
    /// not persisted as an endless cooldown. Explicitly disabled state remains
    /// blocked.
    pub fn is_available_at(&self, now_ms: i64) -> bool {
        if self.status.eq_ignore_ascii_case("disabled") {
            return false;
        }
        match self.blocking_until_ms() {
            Some(retry_after) => retry_after <= now_ms,
            None => true,
        }
    }

    pub fn blocking_until_ms(&self) -> Option<i64> {
        [self.next_retry_after_ms, self.quota.next_recover_at_ms]
            .into_iter()
            .flatten()
            .max()
    }
}

impl fmt::Debug for CooldownStateRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CooldownStateRecord")
            .field("provider", &self.provider)
            .field("auth_id", &self.auth_id)
            .field("model", &self.model)
            .field("status", &self.status)
            .field("next_retry_after_ms", &self.next_retry_after_ms)
            .field("reason", &self.reason)
            .field("quota", &self.quota)
            .field("last_error", &self.last_error)
            .field("updated_at_ms", &self.updated_at_ms)
            .finish()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CooldownQuotaState {
    #[serde(default)]
    pub exceeded: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_recover_at_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub backoff_level: u32,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CooldownErrorState {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub code: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub message: String,
    #[serde(default)]
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
}

impl fmt::Debug for CooldownErrorState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CooldownErrorState")
            .field("code", &self.code)
            .field("message", &"[redacted]")
            .field("retryable", &self.retryable)
            .field("http_status", &self.http_status)
            .finish()
    }
}

/// Persistence boundary implemented by the CTOX host with its SQLite payload store.
pub trait CooldownStateStore: Send + Sync {
    fn load(&self) -> Result<Vec<CooldownStateRecord>, CooldownStoreError>;
    fn save(&self, records: &[CooldownStateRecord]) -> Result<(), CooldownStoreError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CooldownStoreError {
    Read,
    Write,
    InvalidRecord,
}

impl fmt::Display for CooldownStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Read => "cooldown state read failed",
            Self::Write => "cooldown state write failed",
            Self::InvalidRecord => "invalid cooldown state record",
        })
    }
}

impl std::error::Error for CooldownStoreError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> CooldownStateRecord {
        CooldownStateRecord {
            provider: "claude".to_owned(),
            auth_id: "account-a".to_owned(),
            model: Some("claude-sonnet-4-5".to_owned()),
            status: "cooling".to_owned(),
            next_retry_after_ms: Some(2_000),
            reason: "rate_limit".to_owned(),
            quota: CooldownQuotaState {
                exceeded: true,
                reason: "quota".to_owned(),
                next_recover_at_ms: Some(3_000),
                backoff_level: 2,
            },
            last_error: Some(CooldownErrorState {
                code: "rate_limit_error".to_owned(),
                message: "provider detail must not enter Debug".to_owned(),
                retryable: true,
                http_status: Some(429),
            }),
            updated_at_ms: 1_000,
        }
    }

    #[test]
    fn serialized_state_round_trips_without_auth_file_or_credentials() {
        let expected = record();
        let encoded = serde_json::to_string(&expected).unwrap();
        assert!(!encoded.contains("auth_file"));
        assert!(!format!("{expected:?}").contains("provider detail"));
        assert_eq!(
            serde_json::from_str::<CooldownStateRecord>(&encoded).unwrap(),
            expected
        );
    }

    #[test]
    fn future_retry_blocks_until_latest_recovery_time() {
        let state = record();
        assert_eq!(state.blocking_until_ms(), Some(3_000));
        assert!(!state.is_available_at(2_999));
        assert!(state.is_available_at(3_000));
    }

    #[test]
    fn unavailable_without_retry_is_not_an_endless_cooldown() {
        let mut state = record();
        state.status = "error".to_owned();
        state.next_retry_after_ms = None;
        state.quota.next_recover_at_ms = None;
        assert!(state.is_available_at(1_000));
    }

    #[test]
    fn disabled_state_remains_blocked_without_retry_time() {
        let mut state = record();
        state.status = "disabled".to_owned();
        state.next_retry_after_ms = None;
        state.quota.next_recover_at_ms = None;
        assert!(!state.is_available_at(i64::MAX));
    }

    #[test]
    fn record_identity_must_be_nonempty() {
        let mut state = record();
        state.auth_id = "  ".to_owned();
        assert_eq!(state.validate(), Err(CooldownStoreError::InvalidRecord));
    }
}
