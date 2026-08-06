// ref: sdk/cliproxy/auth/antigravity_credits_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, TimeZone, Utc};

use super::*;

struct Clock;
impl AntigravityCreditsClock for Clock {
    fn now(&self) -> DateTime<Utc> {
        Utc.timestamp_opt(1_700_000_000, 0).single().unwrap()
    }
}

struct FailingStore;
impl AntigravityCreditsStore for FailingStore {
    fn set(
        &self,
        _: &str,
        _: &AntigravityCreditsHint,
        _: Duration,
    ) -> Result<(), AntigravityCreditsStoreError> {
        Err(AntigravityCreditsStoreError::Write)
    }
    fn get(&self, _: &str) -> Result<Option<AntigravityCreditsHint>, AntigravityCreditsStoreError> {
        Err(AntigravityCreditsStoreError::Read)
    }
}

#[test]
fn credits_store_failure_fails_closed_without_ambient_cache() {
    let hints = AntigravityCreditsHints::new(Arc::new(FailingStore), Arc::new(Clock));
    assert_eq!(
        hints.has_known("auth"),
        Err(AntigravityCreditsStoreError::Read)
    );
    assert_eq!(
        hints.set("auth", AntigravityCreditsHint::default()),
        Err(AntigravityCreditsStoreError::Write)
    );
}

#[test]
fn credits_fallback_is_only_considered_for_antigravity() {
    let error = std::io::Error::other("429");
    assert!(should_attempt_antigravity_credits_fallback(
        true,
        &error,
        &["antigravity".into()]
    ));
    assert!(!should_attempt_antigravity_credits_fallback(
        true,
        &error,
        &["codex".into()]
    ));
}
