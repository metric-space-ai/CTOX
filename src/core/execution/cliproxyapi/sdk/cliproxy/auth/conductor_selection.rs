// ref: sdk/cliproxy/auth/conductor_selection.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, RwLock};

use super::{canonical_model_key, AccountCandidate, Auth, AuthLifecycle, AuthStatus};

/// Secret-free scheduling metadata supplied by the typed CTOX runtime config
/// and model discovery owners.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SchedulerCapabilities {
    pub priority: i32,
    pub weight: i64,
    pub websocket_enabled: bool,
    pub supported_models: Vec<String>,
}

/// Injected replacement for upstream's ambient global model registry and
/// scheduling values hidden in arbitrary Auth metadata.
pub trait SchedulerCapabilitySource: Send + Sync {
    fn capabilities_for(&self, auth_id: &str, provider: &str) -> Option<SchedulerCapabilities>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulerViewError {
    InvalidWeight,
}

impl fmt::Display for SchedulerViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("scheduler capability weight is invalid")
    }
}

impl std::error::Error for SchedulerViewError {}

/// Derived, secret-free scheduler entry index. This is CTOX's typed replacement
/// for upstream's mutable manager maps and arbitrary scheduler metadata.
///
/// `AuthLifecycle` remains the only Auth owner. Entries contain only routing
/// metadata and are rebuilt explicitly after lifecycle or model-capability
/// changes. A full rebuild is prepared off-lock and published atomically, so a
/// bad capability snapshot cannot expose a half-rebuilt candidate set.
pub struct AuthSchedulerView {
    lifecycle: Arc<AuthLifecycle>,
    capabilities: Arc<dyn SchedulerCapabilitySource>,
    entries: RwLock<BTreeMap<String, AccountCandidate>>,
}

impl AuthSchedulerView {
    #[must_use]
    pub fn new(
        lifecycle: Arc<AuthLifecycle>,
        capabilities: Arc<dyn SchedulerCapabilitySource>,
    ) -> Self {
        Self {
            lifecycle,
            capabilities,
            entries: RwLock::new(BTreeMap::new()),
        }
    }

    /// Rebuilds one entry from the current lifecycle and capability snapshots.
    /// Returns true when the auth is schedulable; missing, disabled or not-yet-
    /// discovered accounts are removed from the view.
    pub fn refresh_entry(&self, auth_id: &str) -> Result<bool, SchedulerViewError> {
        let auth_id = auth_id.trim();
        if auth_id.is_empty() {
            return Ok(false);
        }
        let candidate = self
            .lifecycle
            .get_cached(auth_id)
            .map(|auth| self.candidate_for(&auth))
            .transpose()?
            .flatten();
        let mut entries = self
            .entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(candidate) = candidate {
            entries.insert(auth_id.to_owned(), candidate);
            Ok(true)
        } else {
            entries.remove(auth_id);
            Ok(false)
        }
    }

    /// Atomically rebuilds all entries from one lifecycle snapshot.
    pub fn refresh_all(&self) -> Result<usize, SchedulerViewError> {
        let mut rebuilt = BTreeMap::new();
        for auth in self.lifecycle.snapshot_cached() {
            if let Some(candidate) = self.candidate_for(&auth)? {
                rebuilt.insert(auth.id.clone(), candidate);
            }
        }
        let len = rebuilt.len();
        *self
            .entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = rebuilt;
        Ok(len)
    }

    pub fn remove(&self, auth_id: &str) -> bool {
        self.entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(auth_id.trim())
            .is_some()
    }

    pub fn clear(&self) {
        self.entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    #[must_use]
    pub fn snapshot(&self) -> Vec<AccountCandidate> {
        self.entries
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .cloned()
            .collect()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn candidate_for(&self, auth: &Auth) -> Result<Option<AccountCandidate>, SchedulerViewError> {
        let auth_id = auth.id.trim();
        let provider = auth.provider.trim().to_ascii_lowercase();
        if auth_id.is_empty()
            || provider.is_empty()
            || auth.disabled
            || auth.unavailable
            || auth.status == AuthStatus::Disabled
        {
            return Ok(None);
        }
        let Some(capabilities) = self.capabilities.capabilities_for(auth_id, &provider) else {
            return Ok(None);
        };
        let weight = crate::internal::credentialweight::normalize(capabilities.weight)
            .map_err(|_| SchedulerViewError::InvalidWeight)?;
        let supported_models = capabilities
            .supported_models
            .iter()
            .map(|model| canonical_model_key(model))
            .filter(|model| !model.is_empty())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if supported_models.is_empty() {
            return Ok(None);
        }
        Ok(Some(AccountCandidate {
            auth_id: auth_id.to_owned(),
            provider,
            priority: capabilities.priority,
            weight,
            websocket_enabled: capabilities.websocket_enabled,
            supported_models,
            disabled: false,
        }))
    }
}

impl fmt::Debug for AuthSchedulerView {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthSchedulerView")
            .field("entries", &self.len())
            .finish_non_exhaustive()
    }
}
