// ref: examples/plugin/claude-web-search-router/go/detect_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
use super::{detect, route, Config, RouteRequest};
#[test]
fn detects_fixture() {
    let body = serde_json::json!({"tools":[{"type":"web_search_20250305","max_uses":3}],"messages":[{"role":"user","content":"Perform a web search for the query: rust"}]});
    assert!(detect::is_builtin_web_search(&body, true));
    assert_eq!(detect::query(&body), "rust");
    assert_eq!(detect::max_uses(&body, 5), 3);
    let request = RouteRequest {
        source_format: "claude".into(),
        body,
        available_providers: vec!["codex".into()],
        ..Default::default()
    };
    assert!(!route(&Config::default(), &request).is_empty());
}
