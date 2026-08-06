// ref: internal/api/handlers/management/auth_files_crud.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};

use super::{
    auth_files::normalize_record, ManagementCredentialBatchResult, ManagementCredentialError,
    ManagementCredentialFailure, ManagementCredentialRecord, ManagementCredentialService,
};

impl ManagementCredentialService {
    pub fn upsert_batch(
        &self,
        records: Vec<ManagementCredentialRecord>,
    ) -> Result<ManagementCredentialBatchResult, ManagementCredentialError> {
        let _guard = self.lock_mutation();
        let stored = self
            .store
            .load()
            .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        let mut by_id = stored
            .into_iter()
            .map(|record| (record.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let mut result = ManagementCredentialBatchResult::default();
        let mut seen = BTreeSet::new();
        for record in records {
            let requested_id = record.id.trim().to_owned();
            match normalize_record(record) {
                Ok(record) if seen.insert(record.id.clone()) => {
                    result.accepted.push(record.id.clone());
                    by_id.insert(record.id.clone(), record);
                }
                Ok(record) => result.failed.push(ManagementCredentialFailure {
                    id: record.id,
                    error: ManagementCredentialError::InvalidRecord,
                }),
                Err(error) => result.failed.push(ManagementCredentialFailure {
                    id: requested_id,
                    error,
                }),
            }
        }
        if !result.accepted.is_empty() {
            self.store
                .replace_all(&by_id.into_values().collect::<Vec<_>>())
                .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        }
        Ok(result)
    }

    pub fn delete_batch(
        &self,
        ids: &[String],
    ) -> Result<ManagementCredentialBatchResult, ManagementCredentialError> {
        let _guard = self.lock_mutation();
        let stored = self
            .store
            .load()
            .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        let mut by_id = stored
            .into_iter()
            .map(|record| (record.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let mut result = ManagementCredentialBatchResult::default();
        for requested in ids {
            let id = requested.trim();
            if id.is_empty() {
                result.failed.push(ManagementCredentialFailure {
                    id: String::new(),
                    error: ManagementCredentialError::InvalidRecord,
                });
            } else if by_id.remove(id).is_some() {
                result.accepted.push(id.to_owned());
            } else {
                result.failed.push(ManagementCredentialFailure {
                    id: id.to_owned(),
                    error: ManagementCredentialError::NotFound,
                });
            }
        }
        if !result.accepted.is_empty() {
            self.store
                .replace_all(&by_id.into_values().collect::<Vec<_>>())
                .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        }
        Ok(result)
    }

    pub fn set_disabled(
        &self,
        id: &str,
        auth_index: &str,
        disabled: bool,
    ) -> Result<ManagementCredentialRecord, ManagementCredentialError> {
        let _guard = self.lock_mutation();
        let mut records = self
            .store
            .load()
            .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        let record = records
            .iter_mut()
            .find(|record| record.id == id.trim())
            .ok_or(ManagementCredentialError::NotFound)?;
        if record.auth_index != auth_index.trim() {
            return Err(ManagementCredentialError::AuthIndexMismatch);
        }
        record.disabled = disabled;
        let updated = record.clone();
        self.store
            .replace_all(&records)
            .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        Ok(updated)
    }
}
