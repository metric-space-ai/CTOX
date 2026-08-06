// ref: sdk/cliproxy/service_codex_models_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth::Auth;
use super::service_models::{
    ConfiguredModel, ProviderKeyConfig, ServiceModelConfig, ServiceModelRuntime,
};
use crate::internal::registry::{embedded_models_catalog, ModelRegistry as InternalRegistry};
use std::collections::BTreeMap;
use std::sync::Arc;

fn runtime(config: ServiceModelConfig) -> (ServiceModelRuntime, Arc<InternalRegistry>) {
    let catalog = Arc::new(embedded_models_catalog().unwrap());
    let registry = Arc::new(InternalRegistry::new(Arc::clone(&catalog)));
    let facade: Arc<dyn super::model_registry::ModelRegistry> = registry.clone();
    (ServiceModelRuntime::new(config, facade, catalog), registry)
}

fn auth(key: &str) -> Auth {
    let mut auth = Auth::default();
    auth.id = "codex-auth".into();
    auth.provider = "codex".into();
    auth.attributes = BTreeMap::from([
        ("auth_kind".into(), "api_key".into()),
        ("api_key".into(), key.into()),
        ("source".into(), "config:codex:test".into()),
        ("config_index".into(), "0".into()),
    ]);
    auth
}

#[test]
fn codex_api_key_empty_models_uses_defaults_and_configured_replaces_them() {
    let (service, registry) = runtime(ServiceModelConfig {
        codex_keys: vec![ProviderKeyConfig {
            api_key: "key".into(),
            ..ProviderKeyConfig::default()
        }],
        ..ServiceModelConfig::default()
    });
    service.register_models_for_auth(&auth("key"));
    let defaults = registry.available_models_by_provider("codex");
    assert!(defaults.iter().any(|model| model.id == "gpt-image-1.5"));

    let (service, registry) = runtime(ServiceModelConfig {
        codex_keys: vec![ProviderKeyConfig {
            api_key: "key".into(),
            models: vec![ConfiguredModel {
                name: "upstream".into(),
                alias: "configured".into(),
                ..ConfiguredModel::default()
            }],
            ..ProviderKeyConfig::default()
        }],
        ..ServiceModelConfig::default()
    });
    service.register_models_for_auth(&auth("key"));
    let models = registry.available_models_by_provider("codex");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "configured");
}

#[test]
fn codex_stale_index_requires_credential_match_and_falls_back() {
    let config = ServiceModelConfig {
        codex_keys: vec![
            ProviderKeyConfig {
                api_key: "wrong".into(),
                models: vec![ConfiguredModel {
                    name: "wrong".into(),
                    ..ConfiguredModel::default()
                }],
                ..ProviderKeyConfig::default()
            },
            ProviderKeyConfig {
                api_key: "right".into(),
                models: vec![ConfiguredModel {
                    name: "right-model".into(),
                    ..ConfiguredModel::default()
                }],
                ..ProviderKeyConfig::default()
            },
        ],
        ..ServiceModelConfig::default()
    };
    let (service, registry) = runtime(config);
    let mut current = auth("right");
    current.attributes.insert("config_index".into(), "0".into());
    service.register_models_for_auth(&current);
    assert!(registry.client_supports_model("codex-auth", "right-model"));
    let mut unmatched = auth("missing");
    unmatched.id = "unmatched".into();
    service.register_models_for_auth(&unmatched);
    assert!(!registry.client_supports_model("unmatched", "gpt-image-1.5"));
}
