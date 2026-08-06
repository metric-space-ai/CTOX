// ref: examples/plugin/claude-web-search-router/go/model_resolve_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
use super::model_resolve;
#[test]
fn defaults_never_forward_claude_model() {
    assert_eq!(model_resolve::codex(""), "gpt-5.4-mini");
    assert_eq!(model_resolve::xai(""), "grok-4.3");
}
#[test]
fn configured_antigravity_wins() {
    assert_eq!(
        model_resolve::antigravity("gemini-fixture", &["registry"]),
        "gemini-fixture"
    );
}
