// ref: internal/api/server_sdk_config_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::config::{ProviderCompatConfig, SdkConfig};

use super::server_options::effective_sdk_config;

#[test]
fn effective_sdk_config_copies_codex_multi_agent_setting_without_mutating_source() {
    let sdk = SdkConfig::default();
    let mut providers = ProviderCompatConfig::default();
    providers.codex.optimize_multi_agent_v2 = true;
    let effective = effective_sdk_config(&sdk, &providers);
    assert!(effective.codex_optimize_multi_agent_v2);
    assert!(!sdk.codex_optimize_multi_agent_v2);
}
