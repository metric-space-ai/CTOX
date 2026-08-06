// ref: internal/api/handlers/management/auth_files_fields.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: secret-free field/download/upload views over the injected credential store
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeSet;
use std::fmt;

use serde::Serialize;

use super::{
    ManagementCredentialError, ManagementCredentialFilter, ManagementCredentialRecord,
    ManagementCredentialService, ManagementRecentRequestBucket,
};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ManagementCredentialPatch {
    pub label: Option<String>,
    pub disabled: Option<bool>,
    pub models: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ManagementCredentialRuntimeDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    pub websockets: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub recent_requests: Vec<ManagementRecentRequestBucket>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ManagementCredentialView {
    #[serde(flatten)]
    pub credential: ManagementCredentialRecord,
    #[serde(flatten)]
    pub runtime: ManagementCredentialRuntimeDetails,
}

pub trait ManagementCredentialRuntimeSource: Send + Sync {
    fn details(&self, auth_id: &str) -> ManagementCredentialRuntimeDetails;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManagementCredentialDownload {
    pub filename: String,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManagementCredentialFieldError {
    Credential(ManagementCredentialError),
    InvalidFilename,
    InvalidPayload,
}

impl fmt::Display for ManagementCredentialFieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Credential(_) => "management credential operation failed",
            Self::InvalidFilename => "credential filename is invalid",
            Self::InvalidPayload => "credential payload is invalid",
        })
    }
}

impl std::error::Error for ManagementCredentialFieldError {}

impl From<ManagementCredentialError> for ManagementCredentialFieldError {
    fn from(error: ManagementCredentialError) -> Self {
        Self::Credential(error)
    }
}

impl ManagementCredentialService {
    pub fn patch_fields(
        &self,
        id: &str,
        auth_index: &str,
        patch: ManagementCredentialPatch,
    ) -> Result<ManagementCredentialRecord, ManagementCredentialFieldError> {
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
            return Err(ManagementCredentialError::AuthIndexMismatch.into());
        }
        if let Some(label) = patch.label {
            let label = label.trim();
            if !safe_credential_filename(label) {
                return Err(ManagementCredentialFieldError::InvalidFilename);
            }
            record.label = label.to_owned();
        }
        if let Some(disabled) = patch.disabled {
            record.disabled = disabled;
        }
        if let Some(models) = patch.models {
            record.models = normalize_models(models);
        }
        let updated = record.clone();
        self.store
            .replace_all(&records)
            .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        Ok(updated)
    }

    pub fn list_views(
        &self,
        filter: &ManagementCredentialFilter,
        runtime: &dyn ManagementCredentialRuntimeSource,
    ) -> Result<Vec<ManagementCredentialView>, ManagementCredentialError> {
        Ok(self
            .list(filter)?
            .into_iter()
            .map(|credential| ManagementCredentialView {
                runtime: runtime.details(&credential.id),
                credential,
            })
            .collect())
    }

    pub fn download_projection(
        &self,
        id: &str,
        auth_index: &str,
    ) -> Result<ManagementCredentialDownload, ManagementCredentialFieldError> {
        let record = self
            .list(&ManagementCredentialFilter {
                name: Some(id.trim().to_owned()),
                ..Default::default()
            })?
            .into_iter()
            .find(|record| record.id == id.trim())
            .ok_or(ManagementCredentialError::NotFound)?;
        if record.auth_index != auth_index.trim() {
            return Err(ManagementCredentialError::AuthIndexMismatch.into());
        }
        let payload = serde_json::to_vec_pretty(&record)
            .map_err(|_| ManagementCredentialFieldError::InvalidPayload)?;
        Ok(ManagementCredentialDownload {
            filename: format!("{}.json", safe_public_id(&record.id)?),
            payload,
        })
    }

    pub fn upload_projection(
        &self,
        filename: &str,
        payload: &[u8],
    ) -> Result<ManagementCredentialRecord, ManagementCredentialFieldError> {
        if !safe_credential_filename(filename) || payload.is_empty() || payload.len() > 1024 * 1024
        {
            return Err(if safe_credential_filename(filename) {
                ManagementCredentialFieldError::InvalidPayload
            } else {
                ManagementCredentialFieldError::InvalidFilename
            });
        }
        let mut record: ManagementCredentialRecord = serde_json::from_slice(payload)
            .map_err(|_| ManagementCredentialFieldError::InvalidPayload)?;
        record.label = filename.trim().to_owned();
        let result = self.upsert_batch(vec![record])?;
        if !result.failed.is_empty() || result.accepted.len() != 1 {
            return Err(ManagementCredentialFieldError::InvalidPayload);
        }
        self.list(&ManagementCredentialFilter {
            name: Some(result.accepted[0].clone()),
            ..Default::default()
        })?
        .into_iter()
        .next()
        .ok_or(ManagementCredentialError::NotFound.into())
    }
}

fn normalize_models(models: Vec<String>) -> Vec<String> {
    models
        .into_iter()
        .filter_map(|model| {
            let model = model.trim();
            (!model.is_empty()).then(|| model.to_owned())
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn safe_public_id(id: &str) -> Result<String, ManagementCredentialFieldError> {
    let id = id.trim();
    if id.is_empty()
        || id.len() > 160
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(ManagementCredentialFieldError::InvalidFilename);
    }
    Ok(id.to_owned())
}

pub fn safe_credential_filename(filename: &str) -> bool {
    let filename = filename.trim();
    !filename.is_empty()
        && filename.len() <= 255
        && !filename.contains(['/', '\\'])
        && !filename.contains("..")
        && filename != "."
        && filename.bytes().all(|byte| !byte.is_ascii_control())
}
