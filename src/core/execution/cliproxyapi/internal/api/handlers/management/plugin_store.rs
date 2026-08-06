// ref: internal/api/handlers/management/plugin_store.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: staged store mutations through an injected authority with durable-config rollback
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;

use serde::Serialize;

use super::{validate_plugin_id, ManagementPluginError, ManagementPluginService};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ManagementPluginStoreSource {
    pub id: String,
    pub name: String,
    pub url: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ManagementPluginStoreEntry {
    pub source_id: String,
    pub id: String,
    pub name: String,
    pub description: String,
    pub author: String,
    pub version: String,
    pub install_type: String,
    pub auth_required: bool,
    pub auth_configured: bool,
    pub installed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub installed_version: Option<String>,
    pub update_available: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ManagementPluginStoreCatalog {
    pub sources: Vec<ManagementPluginStoreSource>,
    pub plugins: Vec<ManagementPluginStoreEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_errors: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManagementPluginInstallRequest {
    pub source_id: String,
    pub id: String,
    pub version: Option<String>,
    pub platform: String,
}

#[derive(Clone, PartialEq, Eq)]
pub struct ManagementPluginStagedInstall {
    pub operation_id: String,
    pub source_id: String,
    pub id: String,
    pub version: String,
    pub install_ref: String,
}

impl fmt::Debug for ManagementPluginStagedInstall {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementPluginStagedInstall")
            .field("operation_id", &"[REDACTED]")
            .field("source_id", &self.source_id)
            .field("id", &self.id)
            .field("version", &self.version)
            .field("install_ref", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ManagementPluginInstallResult {
    pub source_id: String,
    pub id: String,
    pub version: String,
    pub status: String,
    pub restart_required: bool,
}

pub trait ManagementPluginStoreAuthority: Send + Sync {
    fn catalog(&self) -> Result<ManagementPluginStoreCatalog, ManagementPluginStoreAuthorityError>;
    fn stage_install(
        &self,
        request: &ManagementPluginInstallRequest,
    ) -> Result<ManagementPluginStagedInstall, ManagementPluginStoreAuthorityError>;
    fn commit_install(
        &self,
        staged: ManagementPluginStagedInstall,
    ) -> ManagementPluginInstallResult;
    fn rollback_install(&self, staged: ManagementPluginStagedInstall);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManagementPluginStoreAuthorityError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManagementPluginStoreError {
    Plugin(ManagementPluginError),
    AuthorityUnavailable,
    InvalidRequest,
    InvalidReceipt,
}

impl fmt::Display for ManagementPluginStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Plugin(_) => "plugin configuration update failed",
            Self::AuthorityUnavailable => "plugin store authority unavailable",
            Self::InvalidRequest => "plugin store request is invalid",
            Self::InvalidReceipt => "plugin store returned an invalid staged operation",
        })
    }
}

impl std::error::Error for ManagementPluginStoreError {}

impl From<ManagementPluginError> for ManagementPluginStoreError {
    fn from(error: ManagementPluginError) -> Self {
        Self::Plugin(error)
    }
}

pub struct ManagementPluginStoreService {
    plugins: Arc<ManagementPluginService>,
    authority: Arc<dyn ManagementPluginStoreAuthority>,
}

impl ManagementPluginStoreService {
    #[must_use]
    pub fn new(
        plugins: Arc<ManagementPluginService>,
        authority: Arc<dyn ManagementPluginStoreAuthority>,
    ) -> Self {
        Self { plugins, authority }
    }

    pub fn catalog(&self) -> Result<ManagementPluginStoreCatalog, ManagementPluginStoreError> {
        let mut catalog = self
            .authority
            .catalog()
            .map_err(|_| ManagementPluginStoreError::AuthorityUnavailable)?;
        validate_catalog(&catalog)?;
        catalog.plugins.sort_by(|left, right| {
            left.id
                .cmp(&right.id)
                .then_with(|| left.source_id.cmp(&right.source_id))
        });
        Ok(catalog)
    }

    pub fn install(
        &self,
        mut request: ManagementPluginInstallRequest,
    ) -> Result<ManagementPluginInstallResult, ManagementPluginStoreError> {
        request.id = validate_plugin_id(&request.id)?.to_owned();
        request.source_id = validate_plugin_id(&request.source_id)?.to_owned();
        request.platform = normalize_platform(&request.platform)?;
        request.version = request
            .version
            .map(|version| normalize_version(&version))
            .transpose()?;
        let staged = self
            .authority
            .stage_install(&request)
            .map_err(|_| ManagementPluginStoreError::AuthorityUnavailable)?;
        if !valid_receipt(&request, &staged) {
            self.authority.rollback_install(staged);
            return Err(ManagementPluginStoreError::InvalidReceipt);
        }
        if let Err(error) = self.plugins.set_enabled(&request.id, true) {
            self.authority.rollback_install(staged);
            return Err(error.into());
        }
        Ok(self.authority.commit_install(staged))
    }
}

impl fmt::Debug for ManagementPluginStoreService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementPluginStoreService")
            .finish_non_exhaustive()
    }
}

fn validate_catalog(
    catalog: &ManagementPluginStoreCatalog,
) -> Result<(), ManagementPluginStoreError> {
    if catalog
        .source_errors
        .iter()
        .any(|code| validate_plugin_id(code).is_err())
    {
        return Err(ManagementPluginStoreError::InvalidReceipt);
    }
    for source in &catalog.sources {
        validate_plugin_id(&source.id)?;
        if !safe_display(&source.name) {
            return Err(ManagementPluginStoreError::InvalidReceipt);
        }
        let url =
            url::Url::parse(&source.url).map_err(|_| ManagementPluginStoreError::InvalidReceipt)?;
        if url.scheme() != "https"
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(ManagementPluginStoreError::InvalidReceipt);
        }
    }
    let sources = catalog
        .sources
        .iter()
        .map(|source| source.id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    for plugin in &catalog.plugins {
        validate_plugin_id(&plugin.id)?;
        if !sources.contains(plugin.source_id.as_str())
            || normalize_version(&plugin.version).is_err()
            || [
                &plugin.name,
                &plugin.description,
                &plugin.author,
                &plugin.install_type,
            ]
            .into_iter()
            .any(|value| !safe_display(value))
        {
            return Err(ManagementPluginStoreError::InvalidReceipt);
        }
    }
    Ok(())
}

fn safe_display(value: &str) -> bool {
    value.len() <= 1_024 && !value.bytes().any(|byte| byte.is_ascii_control())
}

fn normalize_platform(platform: &str) -> Result<String, ManagementPluginStoreError> {
    let platform = platform.trim().to_ascii_lowercase();
    let Some((os, arch)) = platform.split_once('/') else {
        return Err(ManagementPluginStoreError::InvalidRequest);
    };
    if os.is_empty()
        || arch.is_empty()
        || !os.bytes().chain(arch.bytes()).all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
    {
        return Err(ManagementPluginStoreError::InvalidRequest);
    }
    Ok(format!("{os}/{arch}"))
}

fn normalize_version(version: &str) -> Result<String, ManagementPluginStoreError> {
    let version = version.trim().strip_prefix('v').unwrap_or(version.trim());
    if version.is_empty()
        || version.len() > 64
        || !version
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'+'))
    {
        Err(ManagementPluginStoreError::InvalidRequest)
    } else {
        Ok(version.to_owned())
    }
}

fn valid_receipt(
    request: &ManagementPluginInstallRequest,
    staged: &ManagementPluginStagedInstall,
) -> bool {
    !staged.operation_id.trim().is_empty()
        && !staged.install_ref.trim().is_empty()
        && staged.id == request.id
        && staged.source_id == request.source_id
        && request
            .version
            .as_ref()
            .is_none_or(|version| version == &staged.version)
        && normalize_version(&staged.version).is_ok()
}
