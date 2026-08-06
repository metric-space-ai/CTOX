// ref: examples/plugin/claude-web-search-router/go/execution_fallback.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
use super::{fallback, model_resolve, Backend, Config, Plan, RouteRequest};
fn plan(backend: Backend, cfg: &Config, req: &RouteRequest) -> Option<Plan> {
    match backend {
        Backend::AntigravityGoogle
            if fallback::has_provider(&req.available_providers, "antigravity") =>
        {
            let requested = req.requested_model.trim();
            let model = if requested.is_empty() {
                model_resolve::antigravity(&cfg.antigravity_model, &[])
            } else {
                model_resolve::antigravity(&cfg.antigravity_model, &[requested])
            };
            (!model.is_empty()).then_some(Plan { backend, model })
        }
        Backend::CodexWebSearch if fallback::has_provider(&req.available_providers, "codex") => {
            Some(Plan {
                backend,
                model: model_resolve::codex(&cfg.codex_model),
            })
        }
        Backend::XaiWebSearch if fallback::has_provider(&req.available_providers, "xai") => {
            Some(Plan {
                backend,
                model: model_resolve::xai(&cfg.xai_model),
            })
        }
        Backend::Tavily if !cfg.tavily_keys.is_empty() => Some(Plan {
            backend,
            model: String::new(),
        }),
        Backend::DefaultProvider
            if fallback::has_provider(&req.available_providers, &cfg.default_provider) =>
        {
            Some(Plan {
                backend,
                model: cfg.default_provider_model.clone(),
            })
        }
        _ => None,
    }
}
pub fn plans(cfg: &Config, req: &RouteRequest) -> Vec<Plan> {
    if let Some(route) = cfg.route {
        plan(route, cfg, req).into_iter().collect()
    } else {
        fallback::CHAIN
            .into_iter()
            .filter_map(|b| plan(b, cfg, req))
            .collect()
    }
}
