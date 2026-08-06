// ref: internal/api/handlers/management/handler_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use super::{
    management_support_plugin_header, ManagementAuthClock, ManagementAuthError,
    ManagementAuthenticator, ManagementConfigReload, ManagementConfigService,
    ManagementConfigStore, ManagementConfigStoreError, ManagementHandlerOwner,
};
use crate::internal::config::{
    ClaudeSubscriptionAccountConfig, CliproxyRuntimeConfig, RuntimeSecretRef,
};
use crate::sdk::cliproxy::auth::SchedulerStrategy;

struct Clock(AtomicI64);

impl ManagementAuthClock for Clock {
    fn now_ms(&self) -> i64 {
        self.0.load(Ordering::SeqCst)
    }
}

#[test]
fn localhost_ip_ban_blocks_correct_key_during_ban() {
    let authenticator =
        ManagementAuthenticator::new("test-secret", false, Arc::new(Clock(AtomicI64::new(0))))
            .unwrap();
    for _ in 0..5 {
        assert_eq!(
            authenticator.authenticate("127.0.0.1", true, Some("wrong-secret")),
            Err(ManagementAuthError::InvalidKey)
        );
    }
    assert!(matches!(
        authenticator.authenticate("127.0.0.1", true, Some("test-secret")),
        Err(ManagementAuthError::Banned { .. })
    ));
}

#[test]
fn middleware_support_header_reflects_local_transport_capability() {
    assert_eq!(
        management_support_plugin_header(),
        if cfg!(any(unix, windows)) { "1" } else { "0" }
    );
}

#[derive(Debug)]
struct Store(Mutex<CliproxyRuntimeConfig>);

impl ManagementConfigStore for Store {
    fn load(&self) -> Result<CliproxyRuntimeConfig, ManagementConfigStoreError> {
        Ok(self.0.lock().unwrap().clone())
    }

    fn save(&self, config: &CliproxyRuntimeConfig) -> Result<(), ManagementConfigStoreError> {
        *self.0.lock().unwrap() = config.clone();
        Ok(())
    }
}

#[derive(Debug, Default)]
struct Reload(Mutex<Vec<(u64, u64)>>);

impl ManagementConfigReload for Reload {
    fn apply(&self, generation: u64, config: &CliproxyRuntimeConfig) {
        self.0
            .lock()
            .unwrap()
            .push((generation, config.request_timeout_ms));
    }
}

fn secret(name: &str) -> RuntimeSecretRef {
    RuntimeSecretRef {
        scope: "cliproxy".to_owned(),
        name: name.to_owned(),
    }
}

fn runtime_config() -> CliproxyRuntimeConfig {
    CliproxyRuntimeConfig {
        request_timeout_ms: 30_000,
        routing_strategy: SchedulerStrategy::RoundRobin,
        claude_accounts: vec![ClaudeSubscriptionAccountConfig {
            id: "claude-primary".to_owned(),
            disabled: false,
            priority: 0,
            weight: 1,
            websockets: false,
            models: Vec::new(),
            access_token_secret: secret("claude-access"),
            refresh_token_secret: secret("claude-refresh"),
            upstream_scheme: "https".to_owned(),
            upstream_authority: "api.anthropic.com".to_owned(),
            proxy_url_secret: None,
            device_profile: None,
            timezone: String::new(),
        }],
        codex_accounts: Vec::new(),
        antigravity_accounts: Vec::new(),
    }
}

#[test]
fn handler_owner_applies_only_valid_durable_generations_in_order() {
    let store = Arc::new(Store(Mutex::new(runtime_config())));
    let reload = Arc::new(Reload::default());
    let owner = ManagementHandlerOwner::new(
        Arc::new(
            ManagementAuthenticator::new(
                "management-secret",
                false,
                Arc::new(Clock(AtomicI64::new(0))),
            )
            .unwrap(),
        ),
        Arc::new(ManagementConfigService::new(store.clone())),
    )
    .with_reload(reload.clone());

    owner.set_request_timeout_ms(45_000).unwrap();
    owner.set_request_timeout_ms(60_000).unwrap();
    assert!(owner.set_request_timeout_ms(0).is_err());

    assert_eq!(
        reload.0.lock().unwrap().as_slice(),
        &[(1, 45_000), (2, 60_000)]
    );
    assert_eq!(store.load().unwrap().request_timeout_ms, 60_000);
}

#[test]
fn handler_owner_debug_redacts_authentication_and_config_material() {
    let store = Arc::new(Store(Mutex::new(runtime_config())));
    let owner = ManagementHandlerOwner::new(
        Arc::new(
            ManagementAuthenticator::new(
                "never-render-management-secret",
                false,
                Arc::new(Clock(AtomicI64::new(0))),
            )
            .unwrap(),
        ),
        Arc::new(ManagementConfigService::new(store)),
    );
    let rendered = format!("{owner:?}");
    assert!(!rendered.contains("never-render-management-secret"));
    assert!(!rendered.contains("claude-access"));
}
