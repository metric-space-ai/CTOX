// ref: internal/runtime/executor/claude_executor_wire_casing_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::claude_wire_header_name;

#[test]
fn protected_headers_use_captured_lowercase_wire_names() {
    assert_eq!(claude_wire_header_name("Anthropic-Beta"), "anthropic-beta");
    assert_eq!(
        claude_wire_header_name("X-Claude-Code-Session-Id"),
        "x-claude-code-session-id"
    );
    assert_eq!(claude_wire_header_name("custom-header"), "custom-header");
}

#[cfg(feature = "anthropic-fingerprint-transport")]
#[test]
fn native_messages_transport_installs_candidate_header_order() {
    use super::claude_executor_request::claude_messages_orig_headers;
    use super::helps::CLAUDE_CODE_MESSAGES_HEADER_ORDER;

    let actual = claude_messages_orig_headers()
        .iter()
        .map(|(_, name)| String::from_utf8_lossy(name.as_ref()).into_owned())
        .collect::<Vec<_>>();
    assert_eq!(actual, CLAUDE_CODE_MESSAGES_HEADER_ORDER);
}

#[cfg(feature = "anthropic-fingerprint-transport")]
#[test]
fn native_count_tokens_transport_installs_candidate_header_order() {
    use super::claude_executor_request::claude_count_tokens_orig_headers;
    use super::helps::CLAUDE_CODE_COUNT_TOKENS_HEADER_ORDER;

    let actual = claude_count_tokens_orig_headers()
        .iter()
        .map(|(_, name)| String::from_utf8_lossy(name.as_ref()).into_owned())
        .collect::<Vec<_>>();
    assert_eq!(actual, CLAUDE_CODE_COUNT_TOKENS_HEADER_ORDER);
}

#[cfg(feature = "anthropic-fingerprint-transport")]
#[test]
fn native_count_tokens_transport_merges_candidate_betas() {
    use std::collections::HashMap;

    use super::claude_executor::{
        ClaudeCredentialMode, ClaudeMessagesRequest, ClaudeUpstreamTarget,
    };
    use super::claude_executor_request::merged_claude_count_tokens_beta_header;
    use crate::internal::auth::claude::SecretString;

    let request = ClaudeMessagesRequest::new_with_session(
        ClaudeUpstreamTarget::new("https", "api.anthropic.com").unwrap(),
        ClaudeCredentialMode::OAuth,
        &SecretString::new("access-token").unwrap(),
        br#"{"messages":[{"role":"user","content":"hello"}]}"#.to_vec(),
        false,
        "11111111-2222-4333-8444-555555555555",
    )
    .unwrap()
    .with_upstream_metadata(vec!["custom-beta".to_owned()], HashMap::new());

    let betas = merged_claude_count_tokens_beta_header(&request);
    assert!(betas.contains("claude-code-20250219"));
    assert!(betas.contains("token-counting-2024-11-01"));
    assert!(betas.contains("custom-beta"));
}
