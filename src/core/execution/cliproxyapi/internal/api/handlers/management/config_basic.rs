// ref: internal/api/handlers/management/config_basic.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::{Arc, Mutex};

use crate::internal::config::{CliproxyRuntimeConfig, RuntimeConfigError};
use crate::sdk::cliproxy::auth::SchedulerStrategy;

pub trait ManagementConfigStore: Send + Sync {
    fn load(&self) -> Result<CliproxyRuntimeConfig, ManagementConfigStoreError>;
    fn save(&self, config: &CliproxyRuntimeConfig) -> Result<(), ManagementConfigStoreError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagementConfigStoreError {
    Read,
    Write,
}

impl fmt::Display for ManagementConfigStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Read => "management config could not be read",
            Self::Write => "management config could not be written",
        })
    }
}

impl std::error::Error for ManagementConfigStoreError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManagementConfigError {
    Store(ManagementConfigStoreError),
    Invalid(RuntimeConfigError),
    InvalidRoutingStrategy,
    AccountNotFound,
}

impl fmt::Display for ManagementConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Store(_) => "management config store failed",
            Self::Invalid(_) => "management config validation failed",
            Self::InvalidRoutingStrategy => "routing strategy is invalid",
            Self::AccountNotFound => "configured account was not found",
        })
    }
}

impl std::error::Error for ManagementConfigError {}

/// Transactional management configuration facade. Stored values contain only
/// runtime settings and secret references; secret material is never accepted
/// or returned by this HTTP-facing layer.
pub struct ManagementConfigService {
    pub(super) store: Arc<dyn ManagementConfigStore>,
    pub(super) mutation: Mutex<()>,
}

impl ManagementConfigService {
    #[must_use]
    pub fn new(store: Arc<dyn ManagementConfigStore>) -> Self {
        Self {
            store,
            mutation: Mutex::new(()),
        }
    }

    pub fn snapshot(&self) -> Result<CliproxyRuntimeConfig, ManagementConfigError> {
        self.store.load().map_err(ManagementConfigError::Store)
    }

    pub fn replace(
        &self,
        config: CliproxyRuntimeConfig,
    ) -> Result<CliproxyRuntimeConfig, ManagementConfigError> {
        let _guard = self.lock_mutation();
        self.validate_and_save(config)
    }

    pub fn set_request_timeout_ms(
        &self,
        request_timeout_ms: u64,
    ) -> Result<CliproxyRuntimeConfig, ManagementConfigError> {
        self.mutate(|config| config.request_timeout_ms = request_timeout_ms)
    }

    pub fn set_routing_strategy(
        &self,
        strategy: &str,
    ) -> Result<CliproxyRuntimeConfig, ManagementConfigError> {
        let strategy = normalize_routing_strategy(strategy)?;
        self.mutate(|config| config.routing_strategy = strategy)
    }

    pub(super) fn mutate(
        &self,
        update: impl FnOnce(&mut CliproxyRuntimeConfig),
    ) -> Result<CliproxyRuntimeConfig, ManagementConfigError> {
        let _guard = self.lock_mutation();
        let mut config = self.store.load().map_err(ManagementConfigError::Store)?;
        update(&mut config);
        self.validate_and_save(config)
    }

    fn validate_and_save(
        &self,
        config: CliproxyRuntimeConfig,
    ) -> Result<CliproxyRuntimeConfig, ManagementConfigError> {
        let validated = config
            .clone()
            .validate()
            .map_err(ManagementConfigError::Invalid)?;
        let config = validated.into_config();
        self.store
            .save(&config)
            .map_err(ManagementConfigError::Store)?;
        Ok(config)
    }

    fn lock_mutation(&self) -> std::sync::MutexGuard<'_, ()> {
        self.mutation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl fmt::Debug for ManagementConfigService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementConfigService")
            .finish_non_exhaustive()
    }
}

pub fn normalize_routing_strategy(
    strategy: &str,
) -> Result<SchedulerStrategy, ManagementConfigError> {
    match strategy.trim().to_ascii_lowercase().as_str() {
        "" | "round-robin" | "roundrobin" | "rr" => Ok(SchedulerStrategy::RoundRobin),
        "weighted-round-robin" | "weightedroundrobin" | "wrr" => {
            Ok(SchedulerStrategy::WeightedRoundRobin)
        }
        "fill-first" | "fillfirst" | "ff" => Ok(SchedulerStrategy::FillFirst),
        _ => Err(ManagementConfigError::InvalidRoutingStrategy),
    }
}
