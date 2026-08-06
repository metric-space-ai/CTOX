// ref: internal/runtime/executor/helps/thinking_providers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

/// Provider modules force-linked by the pinned Go helper.
///
/// Rust links provider modules statically through `internal::thinking`; this
/// manifest preserves the upstream list for parity checks without creating a
/// second mutable provider registry.
pub const THINKING_PROVIDER_MODULES: [&str; 8] = [
    "antigravity",
    "claude",
    "codex",
    "gemini",
    "interactions",
    "kimi",
    "openai",
    "xai",
];

#[must_use]
pub fn is_upstream_thinking_provider(provider: &str) -> bool {
    let provider = provider.trim();
    THINKING_PROVIDER_MODULES
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(provider))
}
