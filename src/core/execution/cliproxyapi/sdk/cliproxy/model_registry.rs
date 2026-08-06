// ref: sdk/cliproxy/model_registry.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Public, provider-neutral facade for the model registry.
//!
//! Upstream exposes one process-global registry. CTOX owns registry lifetime at
//! the gateway/harness boundary, so callers inject the concrete registry and
//! receive a cloneable capability instead. This prevents one harness from
//! changing another harness's models or hook while preserving the upstream
//! operation set.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::internal::registry::ModelRegistry as InternalModelRegistry;

pub use crate::internal::registry::{ModelRegistryHook, RegistryModelInfo as ModelInfo};

/// Registry operations exposed by the CLIProxy SDK facade.
pub trait ModelRegistry: Send + Sync {
    fn register_client(&self, client_id: &str, client_provider: &str, models: &[ModelInfo]);
    fn unregister_client(&self, client_id: &str);
    fn set_model_quota_exceeded(&self, client_id: &str, model_id: &str);
    fn clear_model_quota_exceeded(&self, client_id: &str, model_id: &str);
    fn client_supports_model(&self, client_id: &str, model_id: &str) -> bool;
    fn available_models(&self, handler_type: &str) -> Vec<Map<String, Value>>;
    fn available_models_by_provider(&self, provider: &str) -> Vec<ModelInfo>;
}

impl ModelRegistry for InternalModelRegistry {
    fn register_client(&self, client_id: &str, client_provider: &str, models: &[ModelInfo]) {
        InternalModelRegistry::register_client(self, client_id, client_provider, models);
    }

    fn unregister_client(&self, client_id: &str) {
        InternalModelRegistry::unregister_client(self, client_id);
    }

    fn set_model_quota_exceeded(&self, client_id: &str, model_id: &str) {
        InternalModelRegistry::set_model_quota_exceeded(self, client_id, model_id);
    }

    fn clear_model_quota_exceeded(&self, client_id: &str, model_id: &str) {
        InternalModelRegistry::clear_model_quota_exceeded(self, client_id, model_id);
    }

    fn client_supports_model(&self, client_id: &str, model_id: &str) -> bool {
        InternalModelRegistry::client_supports_model(self, client_id, model_id)
    }

    fn available_models(&self, handler_type: &str) -> Vec<Map<String, Value>> {
        InternalModelRegistry::available_models(self, handler_type)
    }

    fn available_models_by_provider(&self, provider: &str) -> Vec<ModelInfo> {
        InternalModelRegistry::available_models_by_provider(self, provider)
    }
}

/// Converts an injected concrete registry into the external capability.
#[must_use]
pub fn shared_model_registry(registry: Arc<InternalModelRegistry>) -> Arc<dyn ModelRegistry> {
    registry
}

/// Installs an optional hook on one explicitly owned registry instance.
pub fn set_model_registry_hook(
    registry: &InternalModelRegistry,
    hook: Option<Arc<dyn ModelRegistryHook>>,
) {
    registry.set_hook(hook);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::registry::embedded_models_catalog;

    fn model(id: &str) -> ModelInfo {
        ModelInfo {
            id: id.to_owned(),
            provider_type: "test".to_owned(),
            ..ModelInfo::default()
        }
    }

    #[test]
    fn injected_facade_preserves_registry_operations_and_instance_isolation() {
        let first = Arc::new(InternalModelRegistry::new(Arc::new(
            embedded_models_catalog().expect("embedded catalog"),
        )));
        let second = Arc::new(InternalModelRegistry::new(Arc::new(
            embedded_models_catalog().expect("embedded catalog"),
        )));
        let facade = shared_model_registry(Arc::clone(&first));

        facade.register_client("client", "Claude", &[model("sdk-model")]);
        assert!(facade.client_supports_model("client", "SDK-MODEL"));
        assert_eq!(facade.available_models_by_provider("claude").len(), 1);
        assert!(!second.client_supports_model("client", "sdk-model"));

        facade.set_model_quota_exceeded("client", "sdk-model");
        facade.clear_model_quota_exceeded("client", "sdk-model");
        facade.unregister_client("client");
        assert!(!facade.client_supports_model("client", "sdk-model"));
    }
}
