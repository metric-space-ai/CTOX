// ref: sdk/api/handlers/handlers_routing.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HandlerRouteDecision {
    pub provider: String,
    pub executor_plugin_id: String,
    pub model: String,
    pub reason: String,
}

#[must_use]
pub fn prefer_execution_provider(providers: &[String], preferred: &str) -> Vec<String> {
    let preferred = preferred.trim();
    if preferred.is_empty() {
        return providers.to_vec();
    }
    let mut result = providers.to_vec();
    if let Some(index) = result
        .iter()
        .position(|provider| provider.eq_ignore_ascii_case(preferred))
    {
        let provider = result.remove(index);
        result.insert(0, provider);
    }
    result
}

#[must_use]
pub fn exclude_execution_provider(providers: &[String], excluded: &str) -> Vec<String> {
    providers
        .iter()
        .filter(|provider| !provider.eq_ignore_ascii_case(excluded.trim()))
        .cloned()
        .collect()
}

#[must_use]
pub fn adjust_execution_providers_for_entry_protocol(
    entry_protocol: &str,
    providers: &[String],
) -> Vec<String> {
    if entry_protocol.trim().eq_ignore_ascii_case("interactions") {
        prefer_execution_provider(providers, "gemini-interactions")
    } else {
        providers.to_vec()
    }
}

#[must_use]
pub fn is_openai_image_only_model(model: &str) -> bool {
    let model = route_model_base_name(model).to_ascii_lowercase();
    let model = model
        .strip_prefix("codex/")
        .or_else(|| model.strip_prefix("xai/"))
        .unwrap_or(&model);
    model == "gpt-image-1.5"
        || model == "gpt-image-2"
        || model == "grok-imagine-image"
        || model == "grok-imagine-image-quality"
}

pub fn validate_image_only_model(model: &str, allow_image: bool) -> Result<(), String> {
    if allow_image || !is_openai_image_only_model(model) {
        return Ok(());
    }
    Err(format!(
        "model {} is only supported on /v1/images/generations or /v1/images/edits",
        model.trim()
    ))
}

#[must_use]
pub fn route_model_base_name(model: &str) -> String {
    let model = model.trim();
    let Some(open) = model.rfind('(') else {
        return model.to_owned();
    };
    if !model.ends_with(')') || open == 0 {
        return model.to_owned();
    }
    model[..open].trim().to_owned()
}

#[cfg(test)]
#[path = "handlers_model_router_test.rs"]
mod handlers_model_router_test;

#[cfg(test)]
#[path = "handlers_request_details_test.rs"]
mod handlers_request_details_test;
