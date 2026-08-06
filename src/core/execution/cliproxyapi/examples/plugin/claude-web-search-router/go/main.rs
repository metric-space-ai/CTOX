// ref: examples/plugin/claude-web-search-router/go/main.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
use super::core::{registration, ExampleRegistration};
#[path = "claude_response.rs"]
mod claude_response;
#[path = "detect.rs"]
mod detect;
#[path = "execute_stream.rs"]
mod execute_stream;
#[path = "execution_fallback.rs"]
mod execution_fallback;
#[path = "fallback.rs"]
mod fallback;
#[path = "model_resolve.rs"]
mod model_resolve;
#[path = "penalty.rs"]
mod penalty;
#[path = "stream_forward.rs"]
mod stream_forward;
#[path = "tavily.rs"]
mod tavily;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Backend {
    AntigravityGoogle,
    CodexWebSearch,
    XaiWebSearch,
    Tavily,
    DefaultProvider,
}
#[derive(Clone, Debug)]
pub struct Config {
    pub enabled: bool,
    pub route: Option<Backend>,
    pub antigravity_model: String,
    pub codex_model: String,
    pub xai_model: String,
    pub default_provider: String,
    pub default_provider_model: String,
    pub tavily_keys: Vec<String>,
    pub require_web_search_only: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            route: None,
            antigravity_model: String::new(),
            codex_model: String::new(),
            xai_model: String::new(),
            default_provider: String::new(),
            default_provider_model: String::new(),
            tavily_keys: vec![],
            require_web_search_only: true,
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct RouteRequest {
    pub source_format: String,
    pub requested_model: String,
    pub body: serde_json::Value,
    pub available_providers: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub backend: Backend,
    pub model: String,
}
pub fn example() -> ExampleRegistration {
    registration(
        "example-claude-web-search-router",
        &["model_router", "executor", "claude:claude"],
    )
}
pub fn route(cfg: &Config, req: &RouteRequest) -> Vec<Plan> {
    if !cfg.enabled
        || !detect::is_claude_source(&req.source_format)
        || !detect::is_builtin_web_search(&req.body, cfg.require_web_search_only)
    {
        vec![]
    } else {
        execution_fallback::plans(cfg, req)
    }
}

#[path = "config_test.rs"]
mod config_test;
#[path = "detect_test.rs"]
mod detect_test;
#[path = "execution_route_test.rs"]
mod execution_route_test;
#[path = "fallback_test.rs"]
mod fallback_test;
#[path = "model_resolve_test.rs"]
mod model_resolve_test;
#[path = "penalty_test.rs"]
mod penalty_test;
#[path = "stream_forward_test.rs"]
mod stream_forward_test;
#[path = "tavily_test.rs"]
mod tavily_test;
