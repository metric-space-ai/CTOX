// ref: internal/api/handlers/management/auth_files_provider_oauth.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: provider-neutral OAuth orchestration over injected authorities and atomic credential projections
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::internal::auth::claude::{ClaudeTokenStorage, CLAUDE_DEVICE_IDS_METADATA_KEY};

use super::auth_files::normalize_record;
use super::{
    management_oauth_callback_path, normalize_oauth_provider, normalize_plugin_oauth_provider,
    ManagementCredentialError, ManagementCredentialRecord, ManagementCredentialService,
    ManagementOAuthSessionError, ManagementOAuthSessionSource, ManagementOAuthSessions,
};

#[derive(Clone, PartialEq, Eq)]
pub struct ManagementProviderOAuthStart {
    pub provider: String,
    pub state: String,
    pub authorization_url: String,
}

impl fmt::Debug for ManagementProviderOAuthStart {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementProviderOAuthStart")
            .field("provider", &self.provider)
            .field("state", &self.state)
            .field("authorization_url", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ManagementProviderOAuthPoll {
    pub pending: bool,
    pub error: Option<String>,
    pub credentials: Vec<ManagementCredentialRecord>,
}

/// Builds the non-secret runtime identity projection that a Claude OAuth
/// authority persists alongside the separately stored token handles.
#[must_use]
pub fn claude_oauth_runtime_metadata(storage: &ClaudeTokenStorage) -> BTreeMap<String, Value> {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "email".to_owned(),
        Value::String(storage.email().to_owned()),
    );
    insert_exact_nonempty(&mut metadata, "account_uuid", storage.account_uuid());
    insert_exact_nonempty(
        &mut metadata,
        "organization_uuid",
        storage.organization_uuid(),
    );
    insert_exact_nonempty(
        &mut metadata,
        "organization_name",
        storage.organization_name(),
    );
    if !storage.device_ids().is_empty() {
        metadata.insert(
            CLAUDE_DEVICE_IDS_METADATA_KEY.to_owned(),
            Value::Array(
                storage
                    .device_ids()
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
    }
    metadata
}

fn insert_exact_nonempty(metadata: &mut BTreeMap<String, Value>, key: &str, value: &str) {
    if !value.is_empty() {
        metadata.insert(key.to_owned(), Value::String(value.to_owned()));
    }
}

pub trait ManagementProviderOAuthAuthority: Send + Sync {
    fn begin(
        &self,
        provider: &str,
        state: &str,
        callback_path: &str,
    ) -> Result<String, ManagementProviderOAuthAuthorityError>;
    fn poll(
        &self,
        provider: &str,
        state: &str,
    ) -> Result<ManagementProviderOAuthPoll, ManagementProviderOAuthAuthorityError>;
    fn cancel(
        &self,
        provider: &str,
        state: &str,
    ) -> Result<(), ManagementProviderOAuthAuthorityError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManagementProviderOAuthAuthorityError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManagementProviderOAuthError {
    Session(ManagementOAuthSessionError),
    Credential(ManagementCredentialError),
    AuthorityUnavailable,
    InvalidResponse,
    StateUnavailable,
    VirtualChildConflict,
}

impl fmt::Display for ManagementProviderOAuthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Session(_) => "provider OAuth session is invalid",
            Self::Credential(_) => "provider credential update failed",
            Self::AuthorityUnavailable => "provider OAuth authority unavailable",
            Self::InvalidResponse => "provider OAuth response is invalid",
            Self::StateUnavailable => "provider OAuth state unavailable",
            Self::VirtualChildConflict => "plugin OAuth virtual child cannot be mutated directly",
        })
    }
}

impl std::error::Error for ManagementProviderOAuthError {}

impl From<ManagementOAuthSessionError> for ManagementProviderOAuthError {
    fn from(error: ManagementOAuthSessionError) -> Self {
        Self::Session(error)
    }
}

impl From<ManagementCredentialError> for ManagementProviderOAuthError {
    fn from(error: ManagementCredentialError) -> Self {
        Self::Credential(error)
    }
}

pub struct ManagementProviderOAuth {
    sessions: Arc<ManagementOAuthSessions>,
    credentials: Arc<ManagementCredentialService>,
    authority: Arc<dyn ManagementProviderOAuthAuthority>,
    plugin_children: Mutex<BTreeMap<String, BTreeSet<String>>>,
}

impl ManagementProviderOAuth {
    #[must_use]
    pub fn new(
        sessions: Arc<ManagementOAuthSessions>,
        credentials: Arc<ManagementCredentialService>,
        authority: Arc<dyn ManagementProviderOAuthAuthority>,
    ) -> Self {
        Self {
            sessions,
            credentials,
            authority,
            plugin_children: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn begin_builtin(
        &self,
        provider: &str,
        state: &str,
    ) -> Result<ManagementProviderOAuthStart, ManagementProviderOAuthError> {
        let provider = normalize_oauth_provider(provider)?;
        self.begin(
            &provider,
            state,
            ManagementOAuthSessionSource::Builtin,
            BTreeMap::new(),
        )
    }

    pub fn begin_plugin(
        &self,
        provider: &str,
        state: &str,
        metadata: BTreeMap<String, Value>,
    ) -> Result<ManagementProviderOAuthStart, ManagementProviderOAuthError> {
        let provider = normalize_plugin_oauth_provider(provider)?;
        self.begin(
            &provider,
            state,
            ManagementOAuthSessionSource::Plugin,
            metadata,
        )
    }

    fn begin(
        &self,
        provider: &str,
        state: &str,
        source: ManagementOAuthSessionSource,
        metadata: BTreeMap<String, Value>,
    ) -> Result<ManagementProviderOAuthStart, ManagementProviderOAuthError> {
        let callback_path = management_oauth_callback_path(provider)
            .ok_or(ManagementProviderOAuthError::InvalidResponse)?;
        let authorization_url = self
            .authority
            .begin(provider, state, &callback_path)
            .map_err(|_| ManagementProviderOAuthError::AuthorityUnavailable)?;
        if authorization_url.trim().is_empty() {
            return Err(ManagementProviderOAuthError::InvalidResponse);
        }
        match source {
            ManagementOAuthSessionSource::Builtin => {
                self.sessions.register_builtin(state, provider)?
            }
            ManagementOAuthSessionSource::Plugin => {
                self.sessions.register_plugin(state, provider, metadata)?
            }
        }
        Ok(ManagementProviderOAuthStart {
            provider: provider.to_owned(),
            state: state.trim().to_owned(),
            authorization_url,
        })
    }

    pub fn poll(
        &self,
        provider: &str,
        state: &str,
    ) -> Result<ManagementProviderOAuthPoll, ManagementProviderOAuthError> {
        self.sessions.guard_pending_for_save(state, provider)?;
        let response = self
            .authority
            .poll(provider, state)
            .map_err(|_| ManagementProviderOAuthError::AuthorityUnavailable)?;
        if response.pending {
            if response.error.is_some() || !response.credentials.is_empty() {
                return Err(ManagementProviderOAuthError::InvalidResponse);
            }
            return Ok(response);
        }
        if let Some(message) = response.error.as_deref() {
            if !response.credentials.is_empty() {
                return Err(ManagementProviderOAuthError::InvalidResponse);
            }
            self.sessions.set_error(state, message)?;
            return Ok(response);
        }
        if response.credentials.is_empty() {
            return Err(ManagementProviderOAuthError::InvalidResponse);
        }
        let ids = self.replace_credentials_atomically(response.credentials.clone())?;
        let session = self
            .sessions
            .details(state)?
            .ok_or(ManagementProviderOAuthError::InvalidResponse)?;
        if session.source == ManagementOAuthSessionSource::Plugin {
            self.plugin_children
                .lock()
                .map_err(|_| ManagementProviderOAuthError::StateUnavailable)?
                .insert(session.provider, ids);
        }
        if !self.sessions.complete(state)? {
            return Err(ManagementProviderOAuthError::InvalidResponse);
        }
        Ok(response)
    }

    pub fn cancel(
        &self,
        provider: &str,
        state: &str,
    ) -> Result<bool, ManagementProviderOAuthError> {
        self.sessions.guard_pending_for_save(state, provider)?;
        self.authority
            .cancel(provider, state)
            .map_err(|_| ManagementProviderOAuthError::AuthorityUnavailable)?;
        Ok(self.sessions.cancel(state)?)
    }

    pub fn set_plugin_source_disabled(
        &self,
        provider: &str,
        disabled: bool,
    ) -> Result<usize, ManagementProviderOAuthError> {
        let provider = normalize_plugin_oauth_provider(provider)?;
        let ids = self.plugin_ids(&provider)?;
        self.mutate_plugin_records(&ids, |record| record.disabled = disabled)
    }

    pub fn delete_plugin_source(
        &self,
        provider: &str,
    ) -> Result<usize, ManagementProviderOAuthError> {
        let provider = normalize_plugin_oauth_provider(provider)?;
        let ids = self.plugin_ids(&provider)?;
        let count = self.remove_plugin_records(&ids)?;
        self.plugin_children
            .lock()
            .map_err(|_| ManagementProviderOAuthError::StateUnavailable)?
            .remove(&provider);
        Ok(count)
    }

    pub fn guard_not_virtual_child(&self, id: &str) -> Result<(), ManagementProviderOAuthError> {
        let children = self
            .plugin_children
            .lock()
            .map_err(|_| ManagementProviderOAuthError::StateUnavailable)?;
        if children.values().any(|ids| ids.contains(id.trim())) {
            Err(ManagementProviderOAuthError::VirtualChildConflict)
        } else {
            Ok(())
        }
    }

    fn plugin_ids(&self, provider: &str) -> Result<BTreeSet<String>, ManagementProviderOAuthError> {
        self.plugin_children
            .lock()
            .map_err(|_| ManagementProviderOAuthError::StateUnavailable)?
            .get(provider)
            .cloned()
            .ok_or(ManagementProviderOAuthError::InvalidResponse)
    }

    fn replace_credentials_atomically(
        &self,
        additions: Vec<ManagementCredentialRecord>,
    ) -> Result<BTreeSet<String>, ManagementProviderOAuthError> {
        let _guard = self.credentials.lock_mutation();
        let mut stored = self
            .credentials
            .store
            .load()
            .map_err(|_| ManagementCredentialError::StoreUnavailable)?
            .into_iter()
            .map(|record| (record.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let mut ids = BTreeSet::new();
        for record in additions {
            let record = normalize_record(record)?;
            if !ids.insert(record.id.clone()) {
                return Err(ManagementProviderOAuthError::InvalidResponse);
            }
            stored.insert(record.id.clone(), record);
        }
        self.credentials
            .store
            .replace_all(&stored.into_values().collect::<Vec<_>>())
            .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        Ok(ids)
    }

    fn mutate_plugin_records(
        &self,
        ids: &BTreeSet<String>,
        mutate: impl Fn(&mut ManagementCredentialRecord),
    ) -> Result<usize, ManagementProviderOAuthError> {
        let _guard = self.credentials.lock_mutation();
        let mut records = self
            .credentials
            .store
            .load()
            .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        let mut count = 0;
        for record in &mut records {
            if ids.contains(&record.id) {
                mutate(record);
                count += 1;
            }
        }
        self.credentials
            .store
            .replace_all(&records)
            .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        Ok(count)
    }

    fn remove_plugin_records(
        &self,
        ids: &BTreeSet<String>,
    ) -> Result<usize, ManagementProviderOAuthError> {
        let _guard = self.credentials.lock_mutation();
        let mut records = self
            .credentials
            .store
            .load()
            .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        let before = records.len();
        records.retain(|record| !ids.contains(&record.id));
        self.credentials
            .store
            .replace_all(&records)
            .map_err(|_| ManagementCredentialError::StoreUnavailable)?;
        Ok(before - records.len())
    }
}

impl fmt::Debug for ManagementProviderOAuth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementProviderOAuth")
            .finish_non_exhaustive()
    }
}
