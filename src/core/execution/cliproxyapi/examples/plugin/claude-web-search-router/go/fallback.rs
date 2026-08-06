// ref: examples/plugin/claude-web-search-router/go/fallback.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only
use super::Backend;
pub const CHAIN: [Backend; 4] = [
    Backend::AntigravityGoogle,
    Backend::CodexWebSearch,
    Backend::XaiWebSearch,
    Backend::Tavily,
];
pub fn has_provider(providers: &[String], needle: &str) -> bool {
    providers
        .iter()
        .any(|provider| provider.trim().eq_ignore_ascii_case(needle))
}
