// ref: sdk/cliproxy/auth/selector.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: selector facades share the instance-owned scheduler; session identity remains in the dedicated CTOX session module
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;

use super::cooldown_state::CooldownStateRecord;
use super::scheduler::{AuthScheduler, SchedulerPickOptions, SchedulerStrategy};

/// Immutable account metadata used by the provider-neutral selector.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AccountCandidate {
    pub auth_id: String,
    pub provider: String,
    pub priority: i32,
    pub weight: i64,
    pub websocket_enabled: bool,
    pub supported_models: Vec<String>,
    pub disabled: bool,
}

/// Deterministic fill-first facade over the instance-owned scheduler core.
pub struct FillFirstSelector(AuthScheduler);

impl Default for FillFirstSelector {
    fn default() -> Self {
        Self(AuthScheduler::new(SchedulerStrategy::FillFirst))
    }
}

impl FillFirstSelector {
    pub fn pick(
        &self,
        provider: &str,
        model: Option<&str>,
        now_ms: i64,
        candidates: &[AccountCandidate],
        cooldowns: &[CooldownStateRecord],
    ) -> Result<AccountCandidate, AccountSelectionError> {
        self.0.pick_single(
            provider,
            model,
            now_ms,
            candidates,
            cooldowns,
            &SchedulerPickOptions::default(),
        )
    }
}

impl fmt::Debug for FillFirstSelector {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FillFirstSelector")
            .finish_non_exhaustive()
    }
}

/// Smooth weighted facade. Credits are scoped to this selector instance and
/// reset when the eligible identity/weight vector changes.
pub struct WeightedRoundRobinSelector(AuthScheduler);

impl Default for WeightedRoundRobinSelector {
    fn default() -> Self {
        Self(AuthScheduler::new(SchedulerStrategy::WeightedRoundRobin))
    }
}

impl WeightedRoundRobinSelector {
    pub fn pick(
        &self,
        provider: &str,
        model: Option<&str>,
        now_ms: i64,
        candidates: &[AccountCandidate],
        cooldowns: &[CooldownStateRecord],
    ) -> Result<AccountCandidate, AccountSelectionError> {
        self.0.pick_single(
            provider,
            model,
            now_ms,
            candidates,
            cooldowns,
            &SchedulerPickOptions::default(),
        )
    }
}

impl fmt::Debug for WeightedRoundRobinSelector {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WeightedRoundRobinSelector")
            .finish_non_exhaustive()
    }
}

impl AccountCandidate {
    fn valid_for_provider(&self, provider: &str) -> bool {
        !self.auth_id.trim().is_empty()
            && self.provider.trim().eq_ignore_ascii_case(provider.trim())
    }
}

/// Deterministic provider/model-scoped round robin.
///
/// This ports the accepted selection core without importing upstream's full
/// incremental scheduler. Cursors are intentionally runtime-only; persisted
/// cooldowns determine eligibility after restart, while ordering is restored
/// by sorting account IDs.
#[derive(Default)]
pub struct RoundRobinSelector {
    cursors: Mutex<HashMap<String, usize>>,
}

impl fmt::Debug for RoundRobinSelector {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RoundRobinSelector")
            .finish_non_exhaustive()
    }
}

impl RoundRobinSelector {
    pub fn pick(
        &self,
        provider: &str,
        model: Option<&str>,
        now_ms: i64,
        candidates: &[AccountCandidate],
        cooldowns: &[CooldownStateRecord],
    ) -> Result<AccountCandidate, AccountSelectionError> {
        let mut matching = candidates
            .iter()
            .filter(|candidate| candidate.valid_for_provider(provider))
            .collect::<Vec<_>>();
        if matching.is_empty() {
            return Err(AccountSelectionError::NotFound);
        }
        let matching_count = matching.len();

        let requested_model = super::scheduler::canonical_model_key(model.unwrap_or(""));
        let requested_model = (!requested_model.is_empty()).then_some(requested_model);
        let mut earliest_retry_ms = None;
        let mut blocked_count = 0usize;
        matching.retain(|candidate| {
            if candidate.disabled {
                blocked_count += 1;
                return false;
            }
            let relevant = cooldowns.iter().filter(|state| {
                state.provider.trim().eq_ignore_ascii_case(provider.trim())
                    && state.auth_id.trim() == candidate.auth_id.trim()
                    && match state.model.as_deref().map(str::trim) {
                        None | Some("") => true,
                        Some(state_model) => requested_model.as_deref().is_some_and(|requested| {
                            super::scheduler::canonical_model_key(state_model) == requested
                        }),
                    }
            });
            let mut blocked = false;
            for state in relevant {
                if !state.is_available_at(now_ms) {
                    blocked = true;
                    if let Some(next) = state.blocking_until_ms().filter(|next| *next > now_ms) {
                        earliest_retry_ms =
                            Some(earliest_retry_ms.map_or(next, |current: i64| current.min(next)));
                    }
                }
            }
            if blocked {
                blocked_count += 1;
            }
            !blocked
        });

        if matching.is_empty() {
            return if blocked_count == matching_count {
                earliest_retry_ms
                    .map_or(Err(AccountSelectionError::Unavailable), |retry_after_ms| {
                        Err(AccountSelectionError::Cooldown { retry_after_ms })
                    })
            } else {
                Err(AccountSelectionError::Unavailable)
            };
        }

        let best_priority = matching
            .iter()
            .map(|candidate| candidate.priority)
            .max()
            .expect("matching candidates are nonempty");
        matching.retain(|candidate| candidate.priority == best_priority);
        matching.sort_by(|left, right| left.auth_id.cmp(&right.auth_id));

        let key = format!(
            "{}:{}",
            provider.trim().to_ascii_lowercase(),
            requested_model.as_deref().unwrap_or_default()
        );
        let mut cursors = self
            .cursors
            .lock()
            .map_err(|_| AccountSelectionError::State)?;
        let cursor = cursors.entry(key).or_default();
        if *cursor >= 2_147_483_640 {
            *cursor = 0;
        }
        let selected = (*matching[*cursor % matching.len()]).clone();
        *cursor += 1;
        Ok(selected)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountSelectionError {
    NotFound,
    Unavailable,
    Cooldown { retry_after_ms: i64 },
    State,
}

impl fmt::Display for AccountSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NotFound => "no auth candidates",
            Self::Unavailable => "no auth available",
            Self::Cooldown { .. } => "all credentials are cooling down",
            Self::State => "account selector state unavailable",
        })
    }
}

impl std::error::Error for AccountSelectionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdk::cliproxy::auth::CooldownQuotaState;

    fn candidate(auth_id: &str, priority: i32) -> AccountCandidate {
        AccountCandidate {
            auth_id: auth_id.to_owned(),
            provider: "claude".to_owned(),
            priority,
            weight: 1,
            websocket_enabled: false,
            supported_models: Vec::new(),
            disabled: false,
        }
    }

    fn cooldown(auth_id: &str, model: Option<&str>, until_ms: Option<i64>) -> CooldownStateRecord {
        CooldownStateRecord {
            provider: "claude".to_owned(),
            auth_id: auth_id.to_owned(),
            model: model.map(str::to_owned),
            status: "cooling".to_owned(),
            next_retry_after_ms: until_ms,
            reason: "rate_limit".to_owned(),
            quota: CooldownQuotaState::default(),
            last_error: None,
            updated_at_ms: 1_000,
        }
    }

    #[test]
    fn round_robin_order_is_stable_across_unsorted_inputs() {
        let selector = RoundRobinSelector::default();
        let candidates = [candidate("account-b", 0), candidate("account-a", 0)];
        let picks = (0..3)
            .map(|_| {
                selector
                    .pick("claude", Some("sonnet"), 1_000, &candidates, &[])
                    .unwrap()
                    .auth_id
            })
            .collect::<Vec<_>>();
        assert_eq!(picks, ["account-a", "account-b", "account-a"]);
    }

    #[test]
    fn highest_available_priority_wins() {
        let selector = RoundRobinSelector::default();
        let candidates = [candidate("ordinary", 0), candidate("preferred", 7)];
        assert_eq!(
            selector
                .pick("claude", None, 1_000, &candidates, &[])
                .unwrap()
                .auth_id,
            "preferred"
        );
    }

    #[test]
    fn model_cooldown_blocks_only_matching_model_and_expires() {
        let selector = RoundRobinSelector::default();
        let candidates = [candidate("account-a", 0), candidate("account-b", 0)];
        let states = [cooldown("account-a", Some("sonnet"), Some(2_000))];
        assert_eq!(
            selector
                .pick("claude", Some("sonnet"), 1_999, &candidates, &states)
                .unwrap()
                .auth_id,
            "account-b"
        );
        assert_eq!(
            selector
                .pick("claude", Some("opus"), 1_999, &candidates, &states)
                .unwrap()
                .auth_id,
            "account-a"
        );
        assert!(selector
            .pick("claude", Some("sonnet"), 2_000, &candidates, &states)
            .is_ok());
    }

    #[test]
    fn cooldown_without_retry_does_not_permanently_remove_account() {
        let selector = RoundRobinSelector::default();
        let states = [cooldown("account-a", None, None)];
        assert_eq!(
            selector
                .pick("claude", None, 1_000, &[candidate("account-a", 0)], &states)
                .unwrap()
                .auth_id,
            "account-a"
        );
    }

    #[test]
    fn all_active_cooldowns_report_earliest_retry() {
        let selector = RoundRobinSelector::default();
        let candidates = [candidate("account-a", 0), candidate("account-b", 0)];
        let states = [
            cooldown("account-a", None, Some(4_000)),
            cooldown("account-b", None, Some(3_000)),
        ];
        assert_eq!(
            selector.pick("claude", None, 1_000, &candidates, &states),
            Err(AccountSelectionError::Cooldown {
                retry_after_ms: 3_000
            })
        );
    }
}
