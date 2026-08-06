// ref: internal/config/claude_header_defaults_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::parse::parse_provider_compat_config;

#[test]
fn claude_header_defaults_are_trimmed_and_preserve_explicit_false() {
    let config = parse_provider_compat_config(
        br#"claude-header-defaults:
  user-agent: "  claude-cli/2.1.70 (external, cli)  "
  package-version: "  0.80.0  "
  runtime-version: "  v24.5.0  "
  os: "  MacOS  "
  arch: "  arm64  "
  timeout: "  900  "
  timezone: "  Pacific/Honolulu  "
  stabilize-device-profile: false
"#,
    )
    .unwrap();
    let headers = config.claude_header_defaults;
    assert_eq!(headers.user_agent, "claude-cli/2.1.70 (external, cli)");
    assert_eq!(headers.package_version, "0.80.0");
    assert_eq!(headers.runtime_version, "v24.5.0");
    assert_eq!(headers.os, "MacOS");
    assert_eq!(headers.arch, "arm64");
    assert_eq!(headers.timeout, "900");
    assert_eq!(headers.timezone, "Pacific/Honolulu");
    assert_eq!(headers.stabilize_device_profile, Some(false));
}
