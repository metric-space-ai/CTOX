// ref: internal/api/handlers/management/config_weight_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use crate::internal::config::{
    ClaudeSubscriptionAccountConfig, CliproxyRuntimeConfig, RuntimeConfigError, RuntimeSecretRef,
};
use crate::sdk::cliproxy::auth::SchedulerStrategy;

use super::{
    ManagementAccountConfig, ManagementConfigError, ManagementConfigService, ManagementConfigStore,
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

fn secret(name: &str) -> RuntimeSecretRef {
    RuntimeSecretRef {
        scope: "cliproxy".to_owned(),
        name: name.to_owned(),
    }
}

fn account(weight: i64) -> ClaudeSubscriptionAccountConfig {
    ClaudeSubscriptionAccountConfig {
        id: "claude-1".to_owned(),
        disabled: false,
        priority: 0,
        weight,
        websockets: false,
        models: vec!["claude-sonnet".to_owned()],
        access_token_secret: secret("claude-access"),
        refresh_token_secret: secret("claude-refresh"),
        upstream_scheme: "https".to_owned(),
        upstream_authority: "api.anthropic.com".to_owned(),
        proxy_url_secret: None,
        device_profile: None,
        timezone: String::new(),
    }
}

fn config() -> CliproxyRuntimeConfig {
    CliproxyRuntimeConfig {
        request_timeout_ms: 30_000,
        routing_strategy: SchedulerStrategy::RoundRobin,
        claude_accounts: vec![account(1)],
        codex_accounts: Vec::new(),
        antigravity_accounts: Vec::new(),
    }
}

#[test]
fn account_weight_is_validated_before_store_publication() {
    let store = Arc::new(Store(Mutex::new(config())));
    let service = ManagementConfigService::new(store.clone());
    let error = service
        .upsert_account(ManagementAccountConfig::Claude(account(1_000_001)))
        .unwrap_err();
    assert_eq!(
        error,
        ManagementConfigError::Invalid(RuntimeConfigError::InvalidCredentialWeight)
    );
    assert_eq!(store.load().unwrap().claude_accounts[0].weight, 1);

    service
        .upsert_account(ManagementAccountConfig::Claude(account(7)))
        .unwrap();
    assert_eq!(store.load().unwrap().claude_accounts[0].weight, 7);
}

#[test]
fn invalid_basic_update_does_not_change_durable_snapshot() {
    let store = Arc::new(Store(Mutex::new(config())));
    let service = ManagementConfigService::new(store.clone());
    assert!(matches!(
        service.set_request_timeout_ms(0),
        Err(ManagementConfigError::Invalid(
            RuntimeConfigError::InvalidTimeout
        ))
    ));
    assert_eq!(store.load().unwrap().request_timeout_ms, 30_000);
}
