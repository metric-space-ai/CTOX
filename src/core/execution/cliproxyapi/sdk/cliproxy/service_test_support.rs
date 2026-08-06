// Origin: CTOX test support for upstream service mirrors
// License: AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Value};

use crate::internal::config::{
    ClaudeSubscriptionAccountConfig, CliproxyRuntimeConfig, RuntimeSecretRef,
    ValidatedRuntimeConfig,
};
use crate::sdk::access::Manager as AccessManager;

use super::auth::{
    Auth, AuthLifecycle, AuthManager, AuthRefresher, AuthSchedulerView, AuthStore, AuthStoreError,
    ProviderExecutorRegistration, ProviderExecutorRegistry, RefreshExecutorError, RefreshSchedule,
    SchedulerCapabilities, SchedulerCapabilitySource, SchedulerStrategy,
};
use super::builder::{PluginHost, PluginHostError};
use super::model_registry::{ModelInfo, ModelRegistry};
use super::service_auth::{
    AuthModelResolver, ServiceAuthBindings, ServiceAuthError, ServiceAuthRuntime,
};
use super::service_executors::{ExecutorFactoryError, ServiceExecutorFactory};
use super::service_plugins::ServicePluginRuntime;
use super::types::{PluginAuthParseFuture, PluginAuthParseRequest, PluginAuthParser};

#[derive(Default)]
pub struct MemoryAuthStore(Mutex<BTreeMap<String, Auth>>);

impl AuthStore for MemoryAuthStore {
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

#[derive(Default)]
struct TestCapabilities;

impl SchedulerCapabilitySource for TestCapabilities {
    fn capabilities_for(&self, _auth_id: &str, provider: &str) -> Option<SchedulerCapabilities> {
        Some(SchedulerCapabilities {
            weight: 1,
            supported_models: vec![format!("{provider}-model")],
            ..SchedulerCapabilities::default()
        })
    }
}

#[derive(Default)]
struct NoopRefresher;

impl AuthRefresher for NoopRefresher {
    fn refresh(&self, _auth: &mut Auth) -> Result<Option<Auth>, RefreshExecutorError> {
        Ok(None)
    }
}

pub fn registration(provider: &str) -> Arc<ProviderExecutorRegistration> {
    Arc::new(
        ProviderExecutorRegistration::new(provider, Arc::new(NoopRefresher))
            .expect("valid test provider"),
    )
}

#[derive(Default)]
pub struct RecordingExecutorFactory {
    calls: Mutex<Vec<String>>,
}

impl RecordingExecutorFactory {
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("factory calls").clone()
    }
}

impl ServiceExecutorFactory for RecordingExecutorFactory {
    fn registration_for(
        &self,
        provider_key: &str,
        _auth: &Auth,
    ) -> Result<Arc<ProviderExecutorRegistration>, ExecutorFactoryError> {
        self.calls
            .lock()
            .expect("factory calls")
            .push(provider_key.to_owned());
        Ok(registration(provider_key))
    }
}

#[derive(Default)]
pub struct TestModelRegistry {
    clients: Mutex<BTreeMap<String, (String, Vec<ModelInfo>)>>,
    unregistered: Mutex<Vec<String>>,
}

impl TestModelRegistry {
    pub fn models_for(&self, client_id: &str) -> Vec<ModelInfo> {
        self.clients
            .lock()
            .expect("registry clients")
            .get(client_id)
            .map(|(_, models)| models.clone())
            .unwrap_or_default()
    }

    pub fn unregisters(&self) -> Vec<String> {
        self.unregistered
            .lock()
            .expect("registry unregisters")
            .clone()
    }
}

impl ModelRegistry for TestModelRegistry {
    fn register_client(&self, client_id: &str, provider: &str, models: &[ModelInfo]) {
        self.clients
            .lock()
            .expect("registry clients")
            .insert(client_id.to_owned(), (provider.to_owned(), models.to_vec()));
    }

    fn unregister_client(&self, client_id: &str) {
        self.clients
            .lock()
            .expect("registry clients")
            .remove(client_id);
        self.unregistered
            .lock()
            .expect("registry unregisters")
            .push(client_id.to_owned());
    }

    fn set_model_quota_exceeded(&self, _client_id: &str, _model_id: &str) {}
    fn clear_model_quota_exceeded(&self, _client_id: &str, _model_id: &str) {}

    fn client_supports_model(&self, client_id: &str, model_id: &str) -> bool {
        self.models_for(client_id)
            .iter()
            .any(|model| model.id.eq_ignore_ascii_case(model_id))
    }

    fn available_models(&self, _handler_type: &str) -> Vec<Map<String, Value>> {
        Vec::new()
    }

    fn available_models_by_provider(&self, provider: &str) -> Vec<ModelInfo> {
        self.clients
            .lock()
            .expect("registry clients")
            .values()
            .filter(|(candidate, _)| candidate.eq_ignore_ascii_case(provider))
            .flat_map(|(_, models)| models.clone())
            .collect()
    }
}

struct StaticResolver;

impl AuthModelResolver for StaticResolver {
    fn models_for_auth(&self, auth: &Auth) -> Result<Vec<ModelInfo>, ServiceAuthError> {
        Ok(vec![ModelInfo {
            id: format!("{}-model", auth.provider.trim().to_ascii_lowercase()),
            provider_type: auth.provider.clone(),
            ..ModelInfo::default()
        }])
    }
}

pub struct RuntimeFixture {
    pub runtime: Arc<ServiceAuthRuntime>,
    pub factory: Arc<RecordingExecutorFactory>,
    pub registry: Arc<TestModelRegistry>,
}

pub fn runtime_fixture(
    cooldown: Option<Arc<dyn super::auth::CooldownStateStore>>,
) -> RuntimeFixture {
    let store = Arc::new(MemoryAuthStore::default());
    let lifecycle = Arc::new(AuthLifecycle::new(
        store,
        Arc::new(RefreshSchedule::default()),
        Duration::from_secs(1),
    ));
    let executors = Arc::new(ProviderExecutorRegistry::default());
    let scheduler = Arc::new(AuthSchedulerView::new(
        Arc::clone(&lifecycle),
        Arc::new(TestCapabilities),
    ));
    let manager = Arc::new(AuthManager::new(lifecycle, executors, scheduler));
    let factory = Arc::new(RecordingExecutorFactory::default());
    let registry = Arc::new(TestModelRegistry::default());
    let runtime = Arc::new(ServiceAuthRuntime::new(
        validated_config(),
        ServiceAuthBindings {
            auth_manager: manager,
            access_manager: Arc::new(AccessManager::new()),
            model_registry: registry.clone(),
            model_resolver: Arc::new(StaticResolver),
            executor_factory: factory.clone(),
            usage_manager: Arc::new(super::usage::Manager::new(64)),
            plugin_runtime: None,
            captured_cooldown_store: cooldown,
        },
    ));
    RuntimeFixture {
        runtime,
        factory,
        registry,
    }
}

pub fn validated_config() -> ValidatedRuntimeConfig {
    CliproxyRuntimeConfig {
        request_timeout_ms: 30_000,
        routing_strategy: SchedulerStrategy::RoundRobin,
        claude_accounts: vec![ClaudeSubscriptionAccountConfig {
            id: "configured-claude".into(),
            disabled: false,
            priority: 0,
            weight: 1,
            websockets: false,
            models: Vec::new(),
            access_token_secret: RuntimeSecretRef {
                scope: "test".into(),
                name: "access".into(),
            },
            refresh_token_secret: RuntimeSecretRef {
                scope: "test".into(),
                name: "refresh".into(),
            },
            upstream_scheme: "https".into(),
            upstream_authority: "api.anthropic.com".into(),
            proxy_url_secret: None,
            device_profile: None,
            timezone: String::new(),
        }],
        codex_accounts: Vec::new(),
        antigravity_accounts: Vec::new(),
    }
    .validate()
    .expect("valid test config")
}

pub fn auth(id: &str, provider: &str) -> Auth {
    let mut auth = Auth::default();
    auth.id = id.to_owned();
    auth.provider = provider.to_owned();
    auth.status = super::auth::AuthStatus::Active;
    auth
}

#[derive(Default)]
pub struct TestPluginRuntime {
    calls: Mutex<Vec<&'static str>>,
    registrations: Mutex<Vec<Arc<ProviderExecutorRegistration>>>,
    candidates: Mutex<Vec<String>>,
}

impl TestPluginRuntime {
    pub fn calls(&self) -> Vec<&'static str> {
        self.calls.lock().expect("plugin calls").clone()
    }

    pub fn add_registration(&self, registration: Arc<ProviderExecutorRegistration>) {
        self.registrations
            .lock()
            .expect("plugin registrations")
            .push(registration);
    }

    pub fn add_candidate(&self, provider: &str) {
        self.candidates
            .lock()
            .expect("plugin candidates")
            .push(provider.to_owned());
    }

    fn record(&self, event: &'static str) {
        self.calls.lock().expect("plugin calls").push(event);
    }
}

impl PluginHost for TestPluginRuntime {
    fn apply_config(&self, _config: &ValidatedRuntimeConfig) -> Result<(), PluginHostError> {
        self.record("config");
        Ok(())
    }

    fn register_frontend_auth_providers(&self) -> Result<(), PluginHostError> {
        self.record("frontend");
        Ok(())
    }

    fn access_providers(&self) -> Vec<crate::sdk::access::SharedProvider> {
        Vec::new()
    }
}

impl PluginAuthParser for TestPluginRuntime {
    fn parse_auth<'a>(
        &'a self,
        _request: PluginAuthParseRequest,
    ) -> PluginAuthParseFuture<'a, Option<Auth>> {
        Box::pin(async { Ok((None, false)) })
    }
}

impl ServicePluginRuntime for TestPluginRuntime {
    fn executor_registrations(&self) -> Vec<Arc<ProviderExecutorRegistration>> {
        self.registrations
            .lock()
            .expect("plugin registrations")
            .clone()
    }

    fn has_executor_candidate_provider(&self, provider: &str) -> bool {
        self.candidates
            .lock()
            .expect("plugin candidates")
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(provider))
    }

    fn owns_executor(&self, registration: &Arc<ProviderExecutorRegistration>) -> bool {
        self.registrations
            .lock()
            .expect("plugin registrations")
            .iter()
            .any(|candidate| Arc::ptr_eq(candidate, registration))
    }

    fn models_for_provider(&self, provider: &str) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: format!("plugin-{}-model", provider.trim().to_ascii_lowercase()),
            provider_type: provider.to_owned(),
            ..ModelInfo::default()
        }]
    }

    fn register_models(&self, _registry: Arc<dyn ModelRegistry>) {
        self.record("models");
    }

    fn register_usage_plugins(&self) {
        self.record("usage");
    }

    fn install_translator_hooks(&self) {
        self.record("translator");
    }

    fn refresh_management_routes(&self) {
        self.record("management");
    }
}
