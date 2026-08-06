// ref: sdk/cliproxy/service_config_weight_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth::SchedulerStrategy;
use super::service_config::{normalized_routing_runtime_state, ServiceConfigRuntime};
use crate::internal::config::{
    ClaudeSubscriptionAccountConfig, CliproxyRuntimeConfig, RuntimeSecretRef,
};

fn config(weight: i64, strategy: SchedulerStrategy) -> CliproxyRuntimeConfig {
    CliproxyRuntimeConfig {
        request_timeout_ms: 30_000,
        routing_strategy: strategy,
        claude_accounts: vec![ClaudeSubscriptionAccountConfig {
            id: "claude".into(),
            disabled: false,
            priority: 0,
            weight,
            websockets: false,
            models: Vec::new(),
            access_token_secret: RuntimeSecretRef {
                scope: "s".into(),
                name: "access".into(),
            },
            refresh_token_secret: RuntimeSecretRef {
                scope: "s".into(),
                name: "refresh".into(),
            },
            upstream_scheme: "https".into(),
            upstream_authority: "api.anthropic.com".into(),
            proxy_url_secret: None,
            device_profile: None,
            timezone: String::new(),
        }],
        codex_accounts: Vec::new(),
        antigravity_accounts: Vec::new(),
    }
}

#[test]
fn weighted_round_robin_routing_state_is_preserved() {
    let validated = config(1, SchedulerStrategy::WeightedRoundRobin)
        .validate()
        .unwrap();
    assert_eq!(
        normalized_routing_runtime_state(&validated).strategy,
        SchedulerStrategy::WeightedRoundRobin
    );
}

#[test]
fn invalid_weight_commit_does_not_replace_active_config() {
    let runtime =
        ServiceConfigRuntime::new(config(1, SchedulerStrategy::RoundRobin).validate().unwrap());
    assert!(runtime
        .commit_config_update(config(i64::MAX, SchedulerStrategy::WeightedRoundRobin))
        .is_err());
    let current = runtime.current();
    assert_eq!(current.sequence, 0);
    assert_eq!(
        current.config.routing_strategy(),
        SchedulerStrategy::RoundRobin
    );
}
