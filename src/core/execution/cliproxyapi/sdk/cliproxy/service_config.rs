// ref: sdk/cliproxy/service_config.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Transactional, instance-owned configuration commits. Runtime consumers
//! apply returned snapshots; this module deliberately owns no listener.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::internal::config::{CliproxyRuntimeConfig, RuntimeConfigError, ValidatedRuntimeConfig};
use crate::sdk::cliproxy::auth::SchedulerStrategy;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutingRuntimeState {
    pub strategy: SchedulerStrategy,
    pub session_affinity: bool,
    pub session_affinity_ttl: Duration,
}

impl Default for RoutingRuntimeState {
    fn default() -> Self {
        Self {
            strategy: SchedulerStrategy::RoundRobin,
            session_affinity: false,
            session_affinity_ttl: Duration::from_secs(60 * 60),
        }
    }
}

pub fn normalized_routing_runtime_state(config: &ValidatedRuntimeConfig) -> RoutingRuntimeState {
    RoutingRuntimeState {
        strategy: config.routing_strategy(),
        ..RoutingRuntimeState::default()
    }
}

#[derive(Clone, Debug)]
pub struct ConfigCommit {
    pub config: Arc<ValidatedRuntimeConfig>,
    pub sequence: u64,
}

#[derive(Debug)]
struct ConfigRuntimeState {
    config: Arc<ValidatedRuntimeConfig>,
    sequence: u64,
}

/// Serializes validation and publication so failed updates never replace the
/// last known-good runtime snapshot.
#[derive(Debug)]
pub struct ServiceConfigRuntime {
    state: Mutex<ConfigRuntimeState>,
}

impl ServiceConfigRuntime {
    pub fn new(config: ValidatedRuntimeConfig) -> Self {
        Self {
            state: Mutex::new(ConfigRuntimeState {
                config: Arc::new(config),
                sequence: 0,
            }),
        }
    }

    pub fn commit_config_update(
        &self,
        config: CliproxyRuntimeConfig,
    ) -> Result<ConfigCommit, RuntimeConfigError> {
        Ok(self.publish_validated_config(config.validate()?))
    }

    /// Publishes a config that was validated at an injected watcher or Home
    /// boundary. All runtime sources converge on this sequence owner.
    pub fn publish_validated_config(&self, config: ValidatedRuntimeConfig) -> ConfigCommit {
        let config = Arc::new(config);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.sequence = state.sequence.saturating_add(1);
        state.config = Arc::clone(&config);
        ConfigCommit {
            config,
            sequence: state.sequence,
        }
    }

    pub fn current(&self) -> ConfigCommit {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ConfigCommit {
            config: Arc::clone(&state.config),
            sequence: state.sequence,
        }
    }

    pub fn config_commit_current(&self, commit: &ConfigCommit) -> bool {
        commit.sequence != 0
            && self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .sequence
                == commit.sequence
    }
}
