// ref: examples/plugin/claude-web-search-router/go/model_resolve.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
pub fn antigravity(configured: &str, registry_candidates: &[&str]) -> String {
    let configured = configured.trim();
    if !configured.is_empty() {
        configured.to_owned()
    } else {
        registry_candidates
            .first()
            .copied()
            .unwrap_or_default()
            .to_owned()
    }
}
pub fn codex(configured: &str) -> String {
    let m = configured.trim();
    if m.is_empty() {
        "gpt-5.4-mini".into()
    } else {
        m.into()
    }
}
pub fn xai(configured: &str) -> String {
    let m = configured.trim();
    if m.is_empty() {
        "grok-4.3".into()
    } else {
        m.into()
    }
}
