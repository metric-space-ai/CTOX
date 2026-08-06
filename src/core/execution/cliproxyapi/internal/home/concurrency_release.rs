// ref: internal/home/concurrency_release.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use super::client::Client;
use super::client::CredentialConcurrencyConfig;
use crate::sdk::cliproxy::executionregistry::{
    ReleaseAcknowledgement, ReleaseGroup, ReleaseTicket,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConcurrencyReleaseFrame {
    pub credential_id: String,
    pub model: String,
    pub release_seq: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseError(pub String);
impl fmt::Display for ReleaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ReleaseError {}

pub trait ReleaseSender: Send + Sync {
    fn send(&self, frame: &ConcurrencyReleaseFrame) -> Result<(), ReleaseError>;
}
impl<F> ReleaseSender for F
where
    F: Fn(&ConcurrencyReleaseFrame) -> Result<(), ReleaseError> + Send + Sync,
{
    fn send(&self, frame: &ConcurrencyReleaseFrame) -> Result<(), ReleaseError> {
        self(frame)
    }
}

impl ReleaseSender for Client {
    fn send(&self, frame: &ConcurrencyReleaseFrame) -> Result<(), ReleaseError> {
        let payload = serde_json::to_vec(frame).map_err(|error| ReleaseError(error.to_string()))?;
        self.push_concurrency_release(&payload)
            .map_err(|error| ReleaseError(error.to_string()))
    }
}

#[derive(Default)]
struct ReleaseState {
    latest: i64,
    acked: i64,
    waiters: Vec<(i64, ReleaseAcknowledgement)>,
}
struct FlusherState {
    groups: HashMap<ReleaseGroup, ReleaseState>,
    sender: Option<Arc<dyn ReleaseSender>>,
    config_provider: Arc<dyn Fn() -> CredentialConcurrencyConfig + Send + Sync>,
}

pub struct ReleaseFlusher {
    state: Mutex<FlusherState>,
    changed: Condvar,
}
impl ReleaseFlusher {
    pub fn new(
        config_provider: Arc<dyn Fn() -> CredentialConcurrencyConfig + Send + Sync>,
        sender: Option<Arc<dyn ReleaseSender>>,
    ) -> Self {
        Self {
            state: Mutex::new(FlusherState {
                groups: HashMap::new(),
                sender,
                config_provider,
            }),
            changed: Condvar::new(),
        }
    }
    pub fn set_config_provider(
        &self,
        provider: Arc<dyn Fn() -> CredentialConcurrencyConfig + Send + Sync>,
    ) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .config_provider = provider;
        self.changed.notify_all();
    }
    pub fn set_sender(&self, sender: Option<Arc<dyn ReleaseSender>>) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .sender = sender;
        self.changed.notify_all();
    }
    pub fn mark_dirty(&self, group: ReleaseGroup, sequence: i64) -> Option<ReleaseTicket> {
        if sequence <= 0 || group.credential_id.trim().is_empty() || group.model.trim().is_empty() {
            return None;
        }
        let acknowledgement = ReleaseAcknowledgement::default();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = state.groups.entry(group.clone()).or_default();
        if sequence <= entry.acked {
            acknowledgement.acknowledge();
        } else {
            entry.latest = entry.latest.max(sequence);
            entry.waiters.push((sequence, acknowledgement.clone()));
        }
        drop(state);
        self.changed.notify_all();
        ReleaseTicket::new(group, sequence, Some(acknowledgement))
    }
    /// Sends one stable snapshot. Marks occurring during send remain pending.
    pub fn flush_once(&self) -> bool {
        let (sender, pending) = {
            let state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                state.sender.clone(),
                state
                    .groups
                    .iter()
                    .filter(|(_, value)| value.latest > value.acked)
                    .map(|(group, value)| (group.clone(), value.latest))
                    .collect::<Vec<_>>(),
            )
        };
        let Some(sender) = sender else {
            return false;
        };
        let mut failed = false;
        for (group, sequence) in pending {
            let frame = ConcurrencyReleaseFrame {
                credential_id: group.credential_id.clone(),
                model: group.model.clone(),
                release_seq: sequence,
            };
            if sender.send(&frame).is_err() {
                failed = true;
                continue;
            }
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let entry = state.groups.entry(group).or_default();
            entry.acked = entry.acked.max(sequence);
            let mut pending_waiters = Vec::new();
            for (waiter_sequence, waiter) in entry.waiters.drain(..) {
                if waiter_sequence <= entry.acked {
                    waiter.acknowledge();
                } else {
                    pending_waiters.push((waiter_sequence, waiter));
                }
            }
            entry.waiters = pending_waiters;
        }
        self.changed.notify_all();
        failed
    }
    pub fn flush(&self, timeout: Duration) -> Result<(), ReleaseError> {
        let deadline = Instant::now() + timeout;
        loop {
            self.flush_once();
            let state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state
                .groups
                .values()
                .all(|value| value.latest <= value.acked)
            {
                return Ok(());
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(ReleaseError(
                    "home concurrency release flush timed out".into(),
                ));
            }
            let remaining = deadline
                .saturating_duration_since(now)
                .min(Duration::from_millis(5));
            let _ = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
    pub fn timings(&self) -> (Duration, Duration) {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = (state.config_provider)();
        (
            config.release_flush_interval,
            config
                .release_max_backoff
                .max(config.release_flush_interval),
        )
    }
    pub fn next_delay(&self, current: Duration, failed: bool) -> (Duration, bool) {
        let (flush, max) = self.timings();
        if !failed {
            (flush, false)
        } else {
            (current.saturating_mul(2).max(flush).min(max), true)
        }
    }
    pub fn idle(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .groups
            .values()
            .all(|value| value.latest <= value.acked)
    }
}
