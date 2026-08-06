// ref: sdk/cliproxy/auth/api_key_model_capabilities_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::internal::config::{CodexKey, CodexModel, OpenAiCompatibility, ProviderCompatConfig};
use crate::internal::registry::RegistryThinkingSupport;
use crate::sdk::cliproxy::executor::Request;

use super::api_key_model_capabilities::attach_resolved_api_key_model_info;
use super::*;

#[derive(Default)]
struct TestStore(Mutex<BTreeMap<String, Auth>>);

impl AuthStore for TestStore {
    fn list(&self) -> Result<Vec<Auth>, AuthStoreError> {
        Ok(self
            .0
            .lock()
            .map_err(|_| AuthStoreError::Read)?
            .values()
            .cloned()
            .collect())
    }

    fn save(&self, auth: &Auth) -> Result<String, AuthStoreError> {
        self.0
            .lock()
            .map_err(|_| AuthStoreError::Write)?
            .insert(auth.id.clone(), auth.clone());
        Ok(auth.id.clone())
    }

    fn delete(&self, id: &str) -> Result<(), AuthStoreError> {
        self.0
            .lock()
            .map_err(|_| AuthStoreError::Delete)?
            .remove(id);
        Ok(())
    }
}

struct NoCapabilities;

impl SchedulerCapabilitySource for NoCapabilities {
    fn capabilities_for(&self, _: &str, _: &str) -> Option<SchedulerCapabilities> {
        None
    }
}

pub(super) fn manager() -> AuthManager {
    let lifecycle = Arc::new(AuthLifecycle::new(
        Arc::new(TestStore::default()),
        Arc::new(RefreshSchedule::default()),
        Duration::from_secs(60),
    ));
    AuthManager::new(
        lifecycle.clone(),
        Arc::new(ProviderExecutorRegistry::default()),
        Arc::new(AuthSchedulerView::new(lifecycle, Arc::new(NoCapabilities))),
    )
}

fn model(name: &str, alias: &str, level: &str) -> CodexModel {
    CodexModel {
        name: name.into(),
        alias: alias.into(),
        thinking: Some(RegistryThinkingSupport {
            levels: vec![level.into()],
            ..RegistryThinkingSupport::default()
        }),
        ..CodexModel::default()
    }
}

pub(super) fn auth(id: &str, key: &str) -> Auth {
    let mut auth = Auth::default();
    auth.id = id.into();
    auth.provider = "claude".into();
    auth.prefix = "tenant".into();
    auth.attributes.extend([
        ("auth_kind".into(), "api_key".into()),
        ("api_key".into(), key.into()),
        ("source".into(), "config:claude[0]".into()),
    ]);
    auth
}

pub(super) fn register(manager: &AuthManager, auth: Auth) -> Auth {
    let now = DateTime::parse_from_rfc3339("2026-08-04T12:00:00Z")
        .expect("test time")
        .with_timezone(&Utc);
    manager
        .register(auth, AuthMutationOptions::default(), now)
        .expect("register auth")
}

fn levels(request: &Request) -> Vec<String> {
    resolved_api_key_model_info(request)
        .and_then(|info| info.thinking.clone())
        .map(|thinking| thinking.levels)
        .unwrap_or_default()
}

#[test]
fn selected_credential_owns_exact_capability() {
    let manager = manager();
    manager.set_provider_config(&ProviderCompatConfig {
        claude_api_key: vec![
            CodexKey {
                api_key: "key-high".into(),
                models: vec![model("shared", "public", "high")],
                ..CodexKey::default()
            },
            CodexKey {
                api_key: "key-max".into(),
                models: vec![model("shared", "public", "max")],
                ..CodexKey::default()
            },
        ],
        ..ProviderCompatConfig::default()
    });
    let high = register(&manager, auth("auth-high", "key-high"));
    let max = register(&manager, auth("auth-max", "key-max"));
    assert_eq!(
        levels(&manager.attach_resolved_api_key_model_info(
            Request::default(),
            &high,
            "tenant/public",
            "shared"
        )),
        ["high"]
    );
    assert_eq!(
        levels(&manager.attach_resolved_api_key_model_info(
            Request::default(),
            &max,
            "tenant/public",
            "shared"
        )),
        ["max"]
    );
}

#[test]
fn duplicate_key_uses_exact_config_index() {
    let manager = manager();
    manager.set_provider_config(&ProviderCompatConfig {
        claude_api_key: vec![
            CodexKey {
                api_key: "same-secret".into(),
                models: vec![model("shared", "public", "high")],
                ..CodexKey::default()
            },
            CodexKey {
                api_key: "same-secret".into(),
                models: vec![model("shared", "public", "max")],
                ..CodexKey::default()
            },
        ],
        ..ProviderCompatConfig::default()
    });
    let mut high = auth("high", "same-secret");
    high.attributes.insert("config_index".into(), "0".into());
    let mut max = auth("max", "same-secret");
    max.attributes.insert("config_index".into(), "1".into());
    let high = register(&manager, high);
    let max = register(&manager, max);
    assert_eq!(
        levels(&manager.attach_resolved_api_key_model_info(
            Request::default(),
            &high,
            "tenant/public",
            "shared"
        )),
        ["high"]
    );
    assert_eq!(
        levels(&manager.attach_resolved_api_key_model_info(
            Request::default(),
            &max,
            "tenant/public",
            "shared"
        )),
        ["max"]
    );
}

#[test]
fn exact_suffix_wins_and_unsuffixed_fallback_is_bounded() {
    let manager = manager();
    manager.set_provider_config(&ProviderCompatConfig {
        claude_api_key: vec![CodexKey {
            api_key: "suffix-secret".into(),
            models: vec![
                model("shared(high)", "public-high", "high"),
                model("shared(low)", "public-low", "low"),
            ],
            ..CodexKey::default()
        }],
        ..ProviderCompatConfig::default()
    });
    let auth = register(&manager, auth("suffix", "suffix-secret"));
    let request = manager.attach_resolved_api_key_model_info(
        Request::default(),
        &auth,
        "tenant/public-low",
        "shared(low)",
    );
    assert_eq!(levels(&request), ["low"]);
    let wrong = manager.attach_resolved_api_key_model_info(
        Request::default(),
        &auth,
        "tenant/public-low",
        "shared(high)",
    );
    assert!(resolved_api_key_model_info(&wrong).is_none());
}

#[test]
fn one_execution_keeps_immutable_snapshot_across_reload() {
    let manager = manager();
    let config = |level| ProviderCompatConfig {
        claude_api_key: vec![CodexKey {
            api_key: "reload-secret".into(),
            models: vec![model("shared", "public", level)],
            ..CodexKey::default()
        }],
        ..ProviderCompatConfig::default()
    };
    manager.set_provider_config(&config("high"));
    let auth = register(&manager, auth("reload", "reload-secret"));
    let (models, _, old) = manager.execution_model_candidates_with_alias(&auth, "tenant/public");
    manager.set_provider_config(&config("max"));
    let old_request = attach_resolved_api_key_model_info(
        &old,
        Request::default(),
        &auth,
        "tenant/public",
        &models[0],
    );
    assert_eq!(levels(&old_request), ["high"]);
    let new_request = manager.attach_resolved_api_key_model_info(
        Request::default(),
        &auth,
        "tenant/public",
        &models[0],
    );
    assert_eq!(levels(&new_request), ["max"]);
}

#[test]
fn keyless_openai_compat_supports_pool_and_force_mapping() {
    let manager = manager();
    let mut forced = model("shared", "public", "high");
    forced.force_mapping = true;
    manager.set_provider_config(&ProviderCompatConfig {
        openai_compatibility: vec![OpenAiCompatibility {
            name: "keyless".into(),
            prefix: "tenant".into(),
            base_url: "https://example.invalid/v1".into(),
            models: vec![forced, model("fallback", "public", "low")],
            ..OpenAiCompatibility::default()
        }],
        ..ProviderCompatConfig::default()
    });
    let mut auth = Auth::default();
    auth.id = "keyless".into();
    auth.provider = "openai-compatibility:keyless".into();
    auth.prefix = "tenant".into();
    auth.attributes.extend([
        ("source".into(), "config:keyless[0]".into()),
        ("compat_name".into(), "keyless".into()),
        ("provider_key".into(), "openai-compatibility:keyless".into()),
    ]);
    let auth = register(&manager, auth);
    let (models, alias, snapshot) =
        manager.execution_model_candidates_with_alias(&auth, "tenant/public");
    assert_eq!(models, ["shared", "fallback"]);
    assert!(alias.force_mapping);
    assert_eq!(alias.upstream_model, "shared");
    assert_eq!(
        levels(&attach_resolved_api_key_model_info(
            &snapshot,
            Request::default(),
            &auth,
            "tenant/public",
            "fallback"
        )),
        ["low"]
    );
}

#[test]
fn routing_and_request_debug_never_expose_secrets() {
    let manager = manager();
    manager.set_provider_config(&ProviderCompatConfig {
        claude_api_key: vec![CodexKey {
            api_key: "never-print-this".into(),
            models: vec![model("shared", "public", "high")],
            ..CodexKey::default()
        }],
        ..ProviderCompatConfig::default()
    });
    let auth = register(&manager, auth("redacted", "never-print-this"));
    let request = manager.attach_resolved_api_key_model_info(
        Request::default(),
        &auth,
        "tenant/public",
        "shared",
    );
    let debug = format!(
        "{:?} {:?}",
        manager.api_key_model_routing_snapshot(),
        request
    );
    assert!(!debug.contains("never-print-this"), "{debug}");
    assert!(request.metadata.extensions.is_empty());
}
