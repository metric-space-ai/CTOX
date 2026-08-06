// ref: internal/api/handlers/management/auth_files.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::management_auth_index_for_id;

/// Secret-free management projection of a credential owned by CTOX's runtime
/// and secret stores. Raw auth-file JSON is deliberately not an HTTP DTO.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagementCredentialRecord {
    pub id: String,
    #[serde(default)]
    pub auth_index: String,
    pub label: String,
    pub provider: String,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ManagementCredentialFilter {
    pub name: Option<String>,
    pub auth_index: Option<String>,
    pub provider: Option<String>,
}

pub trait ManagementCredentialStore: Send + Sync {
    fn load(&self) -> Result<Vec<ManagementCredentialRecord>, ManagementCredentialStoreError>;
    fn replace_all(
        &self,
        records: &[ManagementCredentialRecord],
    ) -> Result<(), ManagementCredentialStoreError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManagementCredentialStoreError;

impl fmt::Display for ManagementCredentialStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("management credential store unavailable")
    }
}

impl std::error::Error for ManagementCredentialStoreError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManagementCredentialError {
    StoreUnavailable,
    InvalidRecord,
    NotFound,
    AuthIndexMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagementCredentialFailure {
    pub id: String,
    pub error: ManagementCredentialError,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ManagementCredentialBatchResult {
    pub accepted: Vec<String>,
    pub failed: Vec<ManagementCredentialFailure>,
}

pub struct ManagementCredentialService {
    pub(super) store: Arc<dyn ManagementCredentialStore>,
    pub(super) mutation: Mutex<()>,
}

impl ManagementCredentialService {
    #[must_use]
    pub fn new(store: Arc<dyn ManagementCredentialStore>) -> Self {
        Self {
            store,
            mutation: Mutex::new(()),
        }
    }

    pub fn list(
        &self,
        filter: &ManagementCredentialFilter,
    ) -> Result<Vec<ManagementCredentialRecord>, ManagementCredentialError> {
        let mut records = self
            .store
            .load()
            .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        records.retain(|record| matches_filter(record, filter));
        records.sort_by(|left, right| {
            left.label
                .to_ascii_lowercase()
                .cmp(&right.label.to_ascii_lowercase())
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(records)
    }

    pub(super) fn lock_mutation(&self) -> std::sync::MutexGuard<'_, ()> {
        self.mutation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl fmt::Debug for ManagementCredentialService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementCredentialService")
            .finish_non_exhaustive()
    }
}

pub(super) fn normalize_record(
    mut record: ManagementCredentialRecord,
) -> Result<ManagementCredentialRecord, ManagementCredentialError> {
    record.id = record.id.trim().to_owned();
    record.label = record.label.trim().to_owned();
    record.provider = record.provider.trim().to_ascii_lowercase();
    record.auth_index = record.auth_index.trim().to_owned();
    if record.id.is_empty() || record.label.is_empty() || record.provider.is_empty() {
        return Err(ManagementCredentialError::InvalidRecord);
    }
    if record.auth_index.is_empty() {
        record.auth_index = management_auth_index_for_id(&record.id)
            .ok_or(ManagementCredentialError::InvalidRecord)?;
    }
    record.models.sort();
    record.models.dedup();
    Ok(record)
}

fn matches_filter(
    record: &ManagementCredentialRecord,
    filter: &ManagementCredentialFilter,
) -> bool {
    filter.name.as_ref().is_none_or(|name| {
        let name = name.trim();
        record.id == name || record.label == name
    }) && filter
        .auth_index
        .as_ref()
        .is_none_or(|index| record.auth_index == index.trim())
        && filter
            .provider
            .as_ref()
            .is_none_or(|provider| record.provider.eq_ignore_ascii_case(provider.trim()))
}
