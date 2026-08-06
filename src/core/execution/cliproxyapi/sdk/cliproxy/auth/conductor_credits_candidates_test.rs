// ref: sdk/cliproxy/auth/conductor_credits_candidates_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;
use chrono::{DateTime, TimeZone, Utc};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Default)]
struct Store(Mutex<BTreeMap<String, AntigravityCreditsHint>>);
impl AntigravityCreditsStore for Store {
    fn set(
        &self,
        key: &str,
        hint: &AntigravityCreditsHint,
        _: Duration,
    ) -> Result<(), AntigravityCreditsStoreError> {
        self.0.lock().unwrap().insert(key.into(), hint.clone());
        Ok(())
    }
    fn get(
        &self,
        key: &str,
    ) -> Result<Option<AntigravityCreditsHint>, AntigravityCreditsStoreError> {
        Ok(self.0.lock().unwrap().get(key).cloned())
    }
}
struct Clock;
impl AntigravityCreditsClock for Clock {
    fn now(&self) -> DateTime<Utc> {
        Utc.timestamp_opt(1_700_000_000, 0).single().unwrap()
    }
}
fn auth(id: &str) -> Auth {
    let mut auth = Auth::default();
    auth.id = id.into();
    auth.provider = "antigravity".into();
    auth
}

#[test]
fn known_credits_precede_unknown_and_known_unavailable_is_excluded() {
    let hints = AntigravityCreditsHints::new(Arc::new(Store::default()), Arc::new(Clock));
    hints
        .set(
            "available",
            AntigravityCreditsHint {
                known: true,
                available: true,
                ..AntigravityCreditsHint::default()
            },
        )
        .unwrap();
    hints
        .set(
            "empty",
            AntigravityCreditsHint {
                known: true,
                available: false,
                ..AntigravityCreditsHint::default()
            },
        )
        .unwrap();
    let ranked = hints
        .rank_candidates(
            &[auth("unknown"), auth("empty"), auth("available")],
            "claude-sonnet",
        )
        .unwrap();
    assert_eq!(
        ranked
            .iter()
            .map(|auth| auth.id.as_str())
            .collect::<Vec<_>>(),
        ["available", "unknown"]
    );
    assert!(hints
        .rank_candidates(&[auth("available")], "gemini-flash")
        .unwrap()
        .is_empty());
}
