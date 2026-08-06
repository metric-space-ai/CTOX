// ref: internal/config/codex_websocket_header_defaults_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::parse::parse_provider_compat_config;

#[test]
fn codex_header_defaults_are_trimmed_and_consumable() {
    let config = parse_provider_compat_config(
        br#"codex-header-defaults:
  user-agent: "  my-codex-client/1.0  "
  beta-features: "  feature-a,feature-b  "
"#,
    )
    .unwrap();
    assert_eq!(
        config.codex_header_defaults.user_agent,
        "my-codex-client/1.0"
    );
    assert_eq!(
        config.codex_header_defaults.beta_features,
        "feature-a,feature-b"
    );
    assert!(!config.codex.disable_codex_cloaking);
    let defaults = config.codex_header_defaults.websocket_defaults(false);
    assert_eq!(defaults.user_agent.as_deref(), Some("my-codex-client/1.0"));
    assert_eq!(defaults.beta.as_deref(), Some("feature-a,feature-b"));
}

#[test]
fn codex_identity_flags_parse_without_ambient_configuration() {
    let config = parse_provider_compat_config(
        br#"codex:
  identity-confuse: true
  disable-codex-cloaking: true
  optimize-multi-agent-v2: true
"#,
    )
    .unwrap();
    assert!(config.codex.identity_confuse);
    assert!(config.codex.disable_codex_cloaking);
    assert!(config.codex.optimize_multi_agent_v2);
}
