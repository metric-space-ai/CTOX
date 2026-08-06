// ref: internal/auth/claude/anthropic_auth_proxy_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::anthropic_auth::{new_claude_auth_with_proxy, ClaudeProxyOverride};
use super::token::SecretString;
use super::utls_transport::AnthropicProxyMode;

// Disposition: ported. The resolved SecretString is injected by CTOX's typed
// config/secret boundary; the auth-local direct override remains authoritative.
#[test]
fn proxy_override_direct_takes_precedence() {
    let configured = SecretString::new("socks5://proxy.example.com:1080").unwrap();
    let auth = new_claude_auth_with_proxy(Some(&configured), ClaudeProxyOverride::Direct).unwrap();

    assert_eq!(auth.transport().proxy_mode(), AnthropicProxyMode::Direct);
}

// Disposition: ported. A per-auth proxy works even when no configured proxy is
// present, without introducing a second proxy/config authority.
#[test]
fn proxy_override_is_applied_without_config() {
    let proxy = SecretString::new("socks5://proxy.example.com:1080").unwrap();
    let auth = new_claude_auth_with_proxy(None, ClaudeProxyOverride::Proxy(&proxy)).unwrap();

    assert_eq!(auth.transport().proxy_mode(), AnthropicProxyMode::Proxy);
}
