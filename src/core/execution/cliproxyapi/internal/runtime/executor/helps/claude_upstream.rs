// ref: internal/runtime/executor/helps/claude_upstream.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use url::Url;

/// Reports whether a resolved request targets Anthropic's first-party API
/// origin. Claude-specific wire behavior must not leak onto custom origins.
pub fn is_anthropic_upstream_url(target: &Url) -> bool {
    target.username().is_empty()
        && target.password().is_none()
        && target.scheme().eq_ignore_ascii_case("https")
        && target
            .host_str()
            .is_some_and(|host| host.eq_ignore_ascii_case("api.anthropic.com"))
        && target.port().is_none_or(|port| port == 443)
}
