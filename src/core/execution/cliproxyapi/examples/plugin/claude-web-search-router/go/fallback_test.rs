// ref: examples/plugin/claude-web-search-router/go/fallback_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
use super::{execution_fallback, Backend, Config, RouteRequest};
#[test]
fn fallback_skips_unavailable_backends() {
    let cfg = Config {
        tavily_keys: vec!["fixture".into()],
        ..Default::default()
    };
    let req = RouteRequest {
        available_providers: vec!["codex".into()],
        ..Default::default()
    };
    let plans = execution_fallback::plans(&cfg, &req);
    assert_eq!(
        plans.iter().map(|p| p.backend).collect::<Vec<_>>(),
        vec![Backend::CodexWebSearch, Backend::Tavily]
    );
}
#[test]
fn fallback_exhausts_cleanly() {
    assert!(execution_fallback::plans(&Config::default(), &RouteRequest::default()).is_empty());
}
