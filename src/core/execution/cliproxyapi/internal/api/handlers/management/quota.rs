// ref: internal/api/handlers/management/quota.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde::Serialize;

use crate::sdk::cliproxy::auth::{CooldownConductor, CooldownStoreError};

use super::management_auth_index_for_id;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagementQuotaAccount {
    pub auth_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ManagementQuotaResetResult {
    pub auth_index: String,
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagementQuotaResetError {
    InvalidAccount,
    StoreUnavailable,
}

impl fmt::Display for ManagementQuotaResetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidAccount => "invalid quota account",
            Self::StoreUnavailable => "failed to reset quota",
        })
    }
}

impl std::error::Error for ManagementQuotaResetError {}

pub trait ManagementQuotaResetSource: Send + Sync {
    fn reset_by_index(
        &self,
        auth_index: &str,
    ) -> Result<Option<ManagementQuotaResetResult>, ManagementQuotaResetError>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ManagementQuotaSwitches {
    pub switch_project: bool,
    pub switch_preview_model: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagementQuotaSwitchError {
    StoreUnavailable,
}

impl fmt::Display for ManagementQuotaSwitchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("quota policy unavailable")
    }
}

impl std::error::Error for ManagementQuotaSwitchError {}

/// Typed persistence boundary for the two upstream quota-exceeded fallback
/// toggles. The CTOX host may implement this against revisioned runtime config;
/// no ambient process configuration is consulted.
pub trait ManagementQuotaSwitchSource: Send + Sync {
    fn snapshot(&self) -> Result<ManagementQuotaSwitches, ManagementQuotaSwitchError>;
    fn set_switch_project(&self, value: bool) -> Result<(), ManagementQuotaSwitchError>;
    fn set_switch_preview_model(&self, value: bool) -> Result<(), ManagementQuotaSwitchError>;
}

/// Binds public management indexes to the same transition-locked conductor
/// that persists request-result cooldown changes.
pub struct CooldownManagementQuotaReset {
    accounts: BTreeMap<String, String>,
    conductor: Arc<CooldownConductor>,
}

impl CooldownManagementQuotaReset {
    pub fn new(
        accounts: Vec<ManagementQuotaAccount>,
        conductor: Arc<CooldownConductor>,
    ) -> Result<Self, ManagementQuotaResetError> {
        let mut indexes = BTreeMap::new();
        for account in accounts {
            let auth_id = account.auth_id.trim();
            let Some(index) = management_auth_index_for_id(auth_id) else {
                return Err(ManagementQuotaResetError::InvalidAccount);
            };
            if indexes
                .insert(index, auth_id.to_owned())
                .is_some_and(|existing| existing != auth_id)
            {
                return Err(ManagementQuotaResetError::InvalidAccount);
            }
        }
        Ok(Self {
            accounts: indexes,
            conductor,
        })
    }
}

impl ManagementQuotaResetSource for CooldownManagementQuotaReset {
    fn reset_by_index(
        &self,
        auth_index: &str,
    ) -> Result<Option<ManagementQuotaResetResult>, ManagementQuotaResetError> {
        let auth_index = auth_index.trim();
        let Some(auth_id) = self.accounts.get(auth_index) else {
            return Ok(None);
        };
        let models = self
            .conductor
            .reset_account(auth_id)
            .map_err(|_| ManagementQuotaResetError::StoreUnavailable)?;
        Ok(Some(ManagementQuotaResetResult {
            auth_index: auth_index.to_owned(),
            models,
        }))
    }
}

impl From<CooldownStoreError> for ManagementQuotaResetError {
    fn from(_: CooldownStoreError) -> Self {
        Self::StoreUnavailable
    }
}

impl fmt::Debug for CooldownManagementQuotaReset {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CooldownManagementQuotaReset")
            .field("account_count", &self.accounts.len())
            .field("conductor", &self.conductor)
            .finish()
    }
}
