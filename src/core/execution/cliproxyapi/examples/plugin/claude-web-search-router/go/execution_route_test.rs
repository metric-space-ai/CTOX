// ref: examples/plugin/claude-web-search-router/go/execution_route_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
use super::{execution_fallback, Backend, Config, RouteRequest};
#[test]
fn explicit_tavily_route_is_respected() {
    let cfg = Config {
        route: Some(Backend::Tavily),
        tavily_keys: vec!["fixture".into()],
        ..Default::default()
    };
    let plans = execution_fallback::plans(&cfg, &RouteRequest::default());
    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].backend, Backend::Tavily);
}

#[test]
fn explicit_default_provider_is_supported() {
    let cfg = Config {
        route: Some(Backend::DefaultProvider),
        default_provider: "fixture-provider".into(),
        default_provider_model: "fixture-model".into(),
        ..Default::default()
    };
    let req = RouteRequest {
        available_providers: vec!["fixture-provider".into()],
        ..Default::default()
    };
    let plans = execution_fallback::plans(&cfg, &req);
    assert_eq!(plans[0].backend, Backend::DefaultProvider);
    assert_eq!(plans[0].model, "fixture-model");
}
