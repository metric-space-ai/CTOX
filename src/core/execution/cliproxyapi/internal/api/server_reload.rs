// ref: internal/api/server_reload.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::{Arc, RwLock};

use crate::internal::config::{ProviderCompatConfig, SdkConfig};

use super::server_options::{effective_sdk_config, ServerReloadHook};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerConfigSnapshot {
    pub revision: u64,
    pub providers: ProviderCompatConfig,
    pub sdk: SdkConfig,
}

/// Instance-owned, atomically published server config. Callbacks run only
/// after the write lock is released and receive no credential-bearing config.
pub struct ServerConfigReloader {
    snapshot: RwLock<Arc<ServerConfigSnapshot>>,
    hook: Option<Arc<dyn ServerReloadHook>>,
}

impl ServerConfigReloader {
    #[must_use]
    pub fn new(initial: ServerConfigSnapshot, hook: Option<Arc<dyn ServerReloadHook>>) -> Self {
        Self {
            snapshot: RwLock::new(Arc::new(initial)),
            hook,
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> Arc<ServerConfigSnapshot> {
        self.snapshot
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn publish(&self, mut next: ServerConfigSnapshot) -> bool {
        let current = self.snapshot();
        if next.revision <= current.revision {
            return false;
        }
        next.sdk = effective_sdk_config(&next.sdk, &next.providers);
        let revision = next.revision;
        *self
            .snapshot
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::new(next);
        if let Some(hook) = &self.hook {
            hook.config_reloaded(revision);
        }
        true
    }
}

impl fmt::Debug for ServerConfigReloader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServerConfigReloader")
            .field("revision", &self.snapshot().revision)
            .field("has_hook", &self.hook.is_some())
            .finish()
    }
}
