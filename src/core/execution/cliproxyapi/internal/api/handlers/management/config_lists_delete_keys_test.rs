// ref: internal/api/handlers/management/config_lists_delete_keys_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use crate::internal::config::{
    ClaudeSubscriptionAccountConfig, CliproxyRuntimeConfig, RuntimeSecretRef,
};
use crate::sdk::cliproxy::auth::SchedulerStrategy;

use super::{
    ManagementConfigError, ManagementConfigService, ManagementConfigStore,
    ManagementConfigStoreError,
};

struct Store(Mutex<CliproxyRuntimeConfig>);

impl ManagementConfigStore for Store {
    fn load(&self) -> Result<CliproxyRuntimeConfig, ManagementConfigStoreError> {
        Ok(self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone())
    }

    fn save(&self, config: &CliproxyRuntimeConfig) -> Result<(), ManagementConfigStoreError> {
        *self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = config.clone();
        Ok(())
    }
}

fn account(id: &str) -> ClaudeSubscriptionAccountConfig {
    let secret = |kind: &str| RuntimeSecretRef {
        scope: "cliproxy".to_owned(),
        name: format!("{id}-{kind}"),
    };
    ClaudeSubscriptionAccountConfig {
        id: id.to_owned(),
        disabled: false,
        priority: 0,
        weight: 1,
        websockets: false,
        models: Vec::new(),
        access_token_secret: secret("access"),
        refresh_token_secret: secret("refresh"),
        upstream_scheme: "https".to_owned(),
        upstream_authority: "api.anthropic.com".to_owned(),
        proxy_url_secret: None,
        device_profile: None,
        timezone: String::new(),
    }
}

fn service() -> (Arc<Store>, ManagementConfigService) {
    let store = Arc::new(Store(Mutex::new(CliproxyRuntimeConfig {
        request_timeout_ms: 30_000,
        routing_strategy: SchedulerStrategy::RoundRobin,
        claude_accounts: vec![account("claude-a"), account("claude-b")],
        codex_accounts: Vec::new(),
        antigravity_accounts: Vec::new(),
    })));
    let service = ManagementConfigService::new(store.clone());
    (store, service)
}

#[test]
fn stable_id_deletes_only_matching_credential_reference() {
    let (store, service) = service();
    service.delete_account("claude", "claude-a").unwrap();
    let config = store.load().unwrap();
    assert_eq!(config.claude_accounts.len(), 1);
    assert_eq!(config.claude_accounts[0].id, "claude-b");
}

#[test]
fn unknown_stable_id_does_not_change_config() {
    let (store, service) = service();
    assert_eq!(
        service.delete_account("claude", "missing"),
        Err(ManagementConfigError::AccountNotFound)
    );
    assert_eq!(store.load().unwrap().claude_accounts.len(), 2);
}
