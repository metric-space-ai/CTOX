// ref: internal/runtime/executor/helps/claude_upstream_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox

use super::is_anthropic_upstream_url;
use url::Url;

#[test]
fn anthropic_upstream_url_matches_candidate_table() {
    for (name, target, expected) in [
        (
            "default HTTPS port",
            "https://api.anthropic.com/v1/messages",
            true,
        ),
        (
            "explicit HTTPS port",
            "https://api.anthropic.com:443/v1/messages",
            true,
        ),
        (
            "case insensitive host",
            "https://API.ANTHROPIC.COM/v1/messages",
            true,
        ),
        ("HTTP", "http://api.anthropic.com/v1/messages", false),
        (
            "custom port",
            "https://api.anthropic.com:8443/v1/messages",
            false,
        ),
        (
            "userinfo",
            "https://caller@api.anthropic.com/v1/messages",
            false,
        ),
        (
            "lookalike host",
            "https://api.anthropic.com.example/v1/messages",
            false,
        ),
    ] {
        let parsed = Url::parse(target).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            is_anthropic_upstream_url(&parsed),
            expected,
            "{name}: {target}"
        );
    }
}
