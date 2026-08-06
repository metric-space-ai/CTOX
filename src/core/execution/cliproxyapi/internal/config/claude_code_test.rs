// ref: internal/config/claude_code_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::parse::parse_provider_compat_config;

#[test]
fn claude_code_model_list_cloaking_defaults_and_override() {
    let default = parse_provider_compat_config(b"{}").unwrap();
    assert!(!default.claude_code.disable_cloaking_model_list);
    let disabled =
        parse_provider_compat_config(b"claude-code:\n  disable-cloaking-model-list: true\n")
            .unwrap();
    assert!(disabled.claude_code.disable_cloaking_model_list);
}
