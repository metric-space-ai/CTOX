// ref: sdk/cliproxy/service_auth.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::SystemTime;

use chrono::Utc;
use tokio::sync::mpsc;

use crate::internal::config::ValidatedRuntimeConfig;
use crate::sdk::access::Manager as AccessManager;

use super::auth::{
    Auth, AuthManager, AuthManagerError, AuthMutationOptions, AuthStatus, CooldownStateStore,
    GenericAuthRuntime, GenericExecutionError, ModelResumeSink, PersistenceIntent,
};
use super::model_registry::{ModelInfo, ModelRegistry};
use super::service_executors::ServiceExecutorFactory;
use super::service_plugins::ServicePluginRuntime;
use super::types::{AuthUpdate, AuthUpdateAction, WatcherWrapper};
use crate::sdk::pluginapi::{ExecutorRequest, ExecutorResponse, ExecutorStreamResponse};

const AUTH_UPDATE_QUEUE_CAPACITY: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceAuthError {
    AuthManager,
    ModelResolution,
}

impl fmt::Display for ServiceAuthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::AuthManager => "service auth-manager update failed",
            Self::ModelResolution => "service model resolution failed",
        })
    }
}

impl std::error::Error for ServiceAuthError {}

impl From<AuthManagerError> for ServiceAuthError {
    fn from(_: AuthManagerError) -> Self {
        Self::AuthManager
    }
}

pub trait AuthModelResolver: Send + Sync {
    fn models_for_auth(&self, auth: &Auth) -> Result<Vec<ModelInfo>, ServiceAuthError>;
}

/// Explicit host bindings for the service-auth runtime. Upstream recovers
/// several of these from package globals; CTOX binds every owner once.
pub struct ServiceAuthBindings {
    pub auth_manager: Arc<AuthManager>,
    pub access_manager: Arc<AccessManager>,
    pub model_registry: Arc<dyn ModelRegistry>,
    pub model_resolver: Arc<dyn AuthModelResolver>,
    pub executor_factory: Arc<dyn ServiceExecutorFactory>,
    pub usage_manager: Arc<super::usage::Manager>,
    pub plugin_runtime: Option<Arc<dyn ServicePluginRuntime>>,
    pub captured_cooldown_store: Option<Arc<dyn CooldownStateStore>>,
}

pub struct ServiceAuthRuntime {
    config: RwLock<ValidatedRuntimeConfig>,
    auth_manager: Arc<AuthManager>,
    access_manager: Arc<AccessManager>,
    model_registry: Arc<dyn ModelRegistry>,
    model_resolver: Arc<dyn AuthModelResolver>,
    executor_factory: Arc<dyn ServiceExecutorFactory>,
    usage_manager: Arc<super::usage::Manager>,
    plugin_runtime: RwLock<Option<Arc<dyn ServicePluginRuntime>>>,
    watcher: RwLock<Option<Arc<WatcherWrapper>>>,
    captured_cooldown_store: Option<Arc<dyn CooldownStateStore>>,
    generic_auth_runtime: Option<Arc<GenericAuthRuntime>>,
    auth_updates: Mutex<Option<mpsc::Sender<AuthUpdate>>>,
    pub(super) executor_registration: Mutex<()>,
}

impl fmt::Debug for ServiceAuthRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServiceAuthRuntime")
            .field(
                "auth_count",
                &self.auth_manager.lifecycle().snapshot_cached().len(),
            )
            .field("has_plugin_runtime", &self.plugin_runtime().is_some())
            .field("has_watcher", &self.watcher().is_some())
            .field(
                "has_captured_cooldown_store",
                &self.captured_cooldown_store.is_some(),
            )
            .field("has_auth_queue", &self.auth_update_sender().is_some())
            .finish()
    }
}

impl ServiceAuthRuntime {
    #[must_use]
    pub fn new(config: ValidatedRuntimeConfig, bindings: ServiceAuthBindings) -> Self {
        let generic_auth_runtime = bindings.captured_cooldown_store.as_ref().map(|store| {
            Arc::new(GenericAuthRuntime::new(
                bindings.auth_manager.clone(),
                store.clone(),
                Arc::new(RegistryResumeSink {
                    registry: bindings.model_registry.clone(),
                }),
            ))
        });
        Self {
            config: RwLock::new(config),
            auth_manager: bindings.auth_manager,
            access_manager: bindings.access_manager,
            model_registry: bindings.model_registry,
            model_resolver: bindings.model_resolver,
            executor_factory: bindings.executor_factory,
            usage_manager: bindings.usage_manager,
            plugin_runtime: RwLock::new(bindings.plugin_runtime),
            watcher: RwLock::new(None),
            captured_cooldown_store: bindings.captured_cooldown_store,
            generic_auth_runtime,
            auth_updates: Mutex::new(None),
            executor_registration: Mutex::new(()),
        }
    }

    #[must_use]
    pub fn config(&self) -> ValidatedRuntimeConfig {
        self.config
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn replace_config(&self, config: ValidatedRuntimeConfig) {
        *self
            .config
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = config;
    }

    #[must_use]
    pub fn auth_manager(&self) -> Arc<AuthManager> {
        Arc::clone(&self.auth_manager)
    }

    /// Active non-Home execution conductor. It is absent only when the host
    /// deliberately supplied no cooldown/outcome persistence boundary.
    #[must_use]
    pub fn generic_auth_runtime(&self) -> Option<Arc<GenericAuthRuntime>> {
        self.generic_auth_runtime.clone()
    }

    /// Owning non-Home unary route boundary. HTTP/server adapters must enter
    /// here instead of dispatching through the raw executor registry so auth
    /// preparation, refresh/replay, and persisted cooldown accounting cannot
    /// be bypassed.
    pub async fn execute_provider_route(
        &self,
        providers: &[String],
        request: ExecutorRequest,
    ) -> Result<ExecutorResponse, GenericExecutionError> {
        self.require_generic_auth_runtime()?
            .execute(providers, request)
            .await
    }

    /// Owning non-Home token-count route boundary.
    pub async fn count_tokens_provider_route(
        &self,
        providers: &[String],
        request: ExecutorRequest,
    ) -> Result<ExecutorResponse, GenericExecutionError> {
        self.require_generic_auth_runtime()?
            .count_tokens(providers, request)
            .await
    }

    /// Owning non-Home streaming route boundary. The returned stream remains
    /// conductor-accounted through its committed tail.
    pub async fn execute_stream_provider_route(
        &self,
        providers: &[String],
        request: ExecutorRequest,
    ) -> Result<ExecutorStreamResponse, GenericExecutionError> {
        self.require_generic_auth_runtime()?
            .execute_stream(providers, request)
            .await
    }

    fn require_generic_auth_runtime(
        &self,
    ) -> Result<Arc<GenericAuthRuntime>, GenericExecutionError> {
        self.generic_auth_runtime()
            .ok_or(GenericExecutionError::ConductorUnavailable)
    }

    #[must_use]
    pub fn access_manager(&self) -> Arc<AccessManager> {
        Arc::clone(&self.access_manager)
    }

    #[must_use]
    pub fn model_registry(&self) -> Arc<dyn ModelRegistry> {
        Arc::clone(&self.model_registry)
    }

    #[must_use]
    pub fn model_resolver(&self) -> Arc<dyn AuthModelResolver> {
        Arc::clone(&self.model_resolver)
    }

    #[must_use]
    pub fn executor_factory(&self) -> Arc<dyn ServiceExecutorFactory> {
        Arc::clone(&self.executor_factory)
    }

    pub fn register_usage_plugin(&self, plugin: Arc<dyn super::usage::Plugin>) {
        self.usage_manager.register(plugin);
    }

    #[must_use]
    pub fn usage_manager(&self) -> Arc<super::usage::Manager> {
        Arc::clone(&self.usage_manager)
    }

    pub fn set_plugin_runtime(&self, runtime: Option<Arc<dyn ServicePluginRuntime>>) {
        *self
            .plugin_runtime
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = runtime;
    }

    #[must_use]
    pub fn plugin_runtime(&self) -> Option<Arc<dyn ServicePluginRuntime>> {
        self.plugin_runtime
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Scheduler-facing plugin capability bound to this service instance.
    /// This replaces upstream's private manager field and unsafe test access.
    #[must_use]
    pub fn plugin_scheduler(&self) -> Option<Arc<dyn ServicePluginRuntime>> {
        self.plugin_runtime()
    }

    pub fn set_watcher(&self, watcher: Option<Arc<WatcherWrapper>>) {
        *self
            .watcher
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = watcher;
    }

    #[must_use]
    pub fn watcher(&self) -> Option<Arc<WatcherWrapper>> {
        self.watcher
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    #[must_use]
    pub fn resolve_cooldown_state_store(
        &self,
        save_cooldown_status: bool,
        home_enabled: bool,
    ) -> Option<Arc<dyn CooldownStateStore>> {
        if !save_cooldown_status || home_enabled {
            None
        } else {
            self.captured_cooldown_store.clone()
        }
    }

    pub fn start_auth_update_worker(self: &Arc<Self>) -> AuthUpdateWorker {
        let mut sender = self
            .auth_updates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(existing) = sender.as_ref() {
            return AuthUpdateWorker {
                sender: existing.clone(),
                task: None,
                runtime: Weak::new(),
                owns_queue: false,
            };
        }
        let (tx, mut rx) = mpsc::channel(AUTH_UPDATE_QUEUE_CAPACITY);
        *sender = Some(tx.clone());
        drop(sender);
        let runtime = Arc::clone(self);
        let task = tokio::spawn(async move {
            while let Some(first) = rx.recv().await {
                let mut updates = vec![first];
                while let Ok(next) = rx.try_recv() {
                    updates.push(next);
                }
                let _ = runtime.handle_auth_updates(updates);
            }
        });
        AuthUpdateWorker {
            sender: tx,
            task: Some(task),
            runtime: Arc::downgrade(self),
            owns_queue: true,
        }
    }

    fn auth_update_sender(&self) -> Option<mpsc::Sender<AuthUpdate>> {
        self.auth_updates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn emit_auth_update(&self, update: AuthUpdate) -> Result<(), ServiceAuthError> {
        if self
            .watcher()
            .is_some_and(|watcher| watcher.dispatch_runtime_auth_update(update.clone()))
        {
            return Ok(());
        }
        if self
            .auth_update_sender()
            .is_some_and(|sender| sender.try_send(update.clone()).is_ok())
        {
            return Ok(());
        }
        self.handle_auth_updates(vec![update])
    }

    pub fn handle_auth_updates(&self, updates: Vec<AuthUpdate>) -> Result<(), ServiceAuthError> {
        let updates = coalesce_auth_updates(updates);
        let mut needs_plugin_sync = false;
        for update in updates {
            match update.action {
                Some(AuthUpdateAction::Add | AuthUpdateAction::Modify) => {
                    let Some(auth) = update.auth.filter(|auth| !auth.id.trim().is_empty()) else {
                        continue;
                    };
                    self.apply_core_auth_add_or_update(auth)?;
                    needs_plugin_sync = true;
                }
                Some(AuthUpdateAction::Delete) => {
                    let id = auth_update_id(&update);
                    if !id.is_empty() {
                        self.apply_core_auth_removal(&id);
                        needs_plugin_sync = true;
                    }
                }
                None => {}
            }
        }
        if needs_plugin_sync {
            self.sync_plugin_runtime()?;
        }
        Ok(())
    }

    pub fn apply_core_auth_add_or_update(&self, mut auth: Auth) -> Result<(), ServiceAuthError> {
        auth.id = auth.id.trim().to_owned();
        if auth.id.is_empty() {
            return Ok(());
        }
        self.ensure_executors_for_auth(&auth, false)?;
        let existing = self.auth_manager.lifecycle().get_cached(&auth.id);
        if let Some(existing) = &existing {
            auth.created_at = existing.created_at;
            if auth_is_active(existing) && auth_is_active(&auth) {
                auth.last_refreshed_at = existing.last_refreshed_at;
                auth.next_refresh_after = existing.next_refresh_after;
                if auth.model_states.is_empty() {
                    auth.model_states = existing.model_states.clone();
                }
            }
        }
        let options = AuthMutationOptions {
            persistence: PersistenceIntent::SourceAlreadyPersisted,
        };
        let active = if existing.is_some() {
            self.auth_manager
                .update(auth, options, utc_now())?
                .unwrap_or_default()
        } else {
            self.auth_manager.register(auth, options, utc_now())?
        };
        self.complete_model_registration_for_auth(&active)
    }

    pub fn complete_model_registration_for_auth(
        &self,
        auth: &Auth,
    ) -> Result<(), ServiceAuthError> {
        let mut models = self.model_resolver.models_for_auth(auth)?;
        if let Some(plugin) = self.plugin_runtime() {
            models = append_plugin_models(models, plugin.models_for_provider(&auth.provider));
        }
        let models = normalize_models(models);
        if models.is_empty() {
            self.model_registry.unregister_client(&auth.id);
        } else {
            self.model_registry
                .register_client(&auth.id, &auth.provider, &models);
        }
        self.auth_manager.refresh_scheduler_entry(&auth.id)?;
        Ok(())
    }

    pub fn apply_core_auth_removal(&self, id: &str) -> bool {
        let id = id.trim();
        if id.is_empty() {
            return false;
        }
        self.model_registry.unregister_client(id);
        self.auth_manager.remove_runtime(id)
    }

    pub fn websocket_connected(&self, channel_id: &str) -> Result<bool, ServiceAuthError> {
        let channel_id = channel_id.trim();
        if channel_id.is_empty() || !channel_id.to_ascii_lowercase().starts_with("aistudio-") {
            return Ok(false);
        }
        if self
            .auth_manager
            .lifecycle()
            .get_cached(channel_id)
            .is_some_and(|auth| auth_is_active(&auth))
        {
            return Ok(false);
        }
        let now = utc_now();
        let mut auth = Auth::default();
        auth.id = channel_id.to_owned();
        auth.provider = "aistudio".to_owned();
        auth.label = channel_id.to_owned();
        auth.status = AuthStatus::Active;
        auth.created_at = now;
        auth.updated_at = now;
        auth.attributes
            .insert("runtime_only".to_owned(), "true".to_owned());
        auth.metadata.insert(
            "email".to_owned(),
            serde_json::Value::String(channel_id.to_owned()),
        );
        self.emit_auth_update(AuthUpdate {
            action: Some(AuthUpdateAction::Add),
            id: channel_id.to_owned(),
            auth: Some(auth),
        })?;
        Ok(true)
    }

    pub fn websocket_disconnected(&self, channel_id: &str, replaced: bool) -> bool {
        let channel_id = channel_id.trim();
        if channel_id.is_empty() || replaced {
            return false;
        }
        self.emit_auth_update(AuthUpdate {
            action: Some(AuthUpdateAction::Delete),
            id: channel_id.to_owned(),
            auth: None,
        })
        .is_ok()
    }
}

struct RegistryResumeSink {
    registry: Arc<dyn ModelRegistry>,
}

impl ModelResumeSink for RegistryResumeSink {
    fn resume_model(&self, auth_id: &str, model: &str) {
        self.registry.clear_model_quota_exceeded(auth_id, model);
    }
}

fn utc_now() -> chrono::DateTime<Utc> {
    chrono::DateTime::<Utc>::from(SystemTime::now())
}

pub struct AuthUpdateWorker {
    sender: mpsc::Sender<AuthUpdate>,
    task: Option<tokio::task::JoinHandle<()>>,
    runtime: Weak<ServiceAuthRuntime>,
    owns_queue: bool,
}

impl AuthUpdateWorker {
    #[must_use]
    pub fn sender(&self) -> mpsc::Sender<AuthUpdate> {
        self.sender.clone()
    }
}

impl Drop for AuthUpdateWorker {
    fn drop(&mut self) {
        if self.owns_queue {
            if let Some(runtime) = self.runtime.upgrade() {
                let mut active = runtime
                    .auth_updates
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if active
                    .as_ref()
                    .is_some_and(|sender| sender.same_channel(&self.sender))
                {
                    *active = None;
                }
            }
        }
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

#[must_use]
pub fn coalesce_auth_updates(updates: Vec<AuthUpdate>) -> Vec<AuthUpdate> {
    if updates.len() <= 1 {
        return updates;
    }
    let mut order = Vec::new();
    let mut by_id = BTreeMap::new();
    let mut unkeyed = Vec::new();
    for update in updates {
        let id = auth_update_id(&update);
        if id.is_empty() {
            unkeyed.push(update);
            continue;
        }
        if !by_id.contains_key(&id) {
            order.push(id.clone());
        }
        by_id.insert(id, update);
    }
    let mut result = order
        .into_iter()
        .filter_map(|id| by_id.remove(&id))
        .collect::<Vec<_>>();
    result.extend(unkeyed);
    result
}

#[must_use]
pub fn auth_update_id(update: &AuthUpdate) -> String {
    let id = update.id.trim();
    if !id.is_empty() {
        return id.to_owned();
    }
    update
        .auth
        .as_ref()
        .map_or_else(String::new, |auth| auth.id.trim().to_owned())
}

fn auth_is_active(auth: &Auth) -> bool {
    !auth.disabled && auth.status != AuthStatus::Disabled
}

fn normalize_models(models: Vec<ModelInfo>) -> Vec<ModelInfo> {
    let mut seen = BTreeSet::new();
    models
        .into_iter()
        .filter_map(|mut model| {
            model.id = model.id.trim().to_owned();
            if model.id.is_empty() || !seen.insert(model.id.clone()) {
                None
            } else {
                Some(model)
            }
        })
        .collect()
}

fn append_plugin_models(mut native: Vec<ModelInfo>, plugin: Vec<ModelInfo>) -> Vec<ModelInfo> {
    let mut seen = native
        .iter()
        .filter_map(|model| {
            let id = model.id.trim();
            (!id.is_empty()).then(|| id.to_owned())
        })
        .collect::<BTreeSet<_>>();
    native.extend(plugin.into_iter().filter(|model| {
        let id = model.id.trim();
        !id.is_empty() && seen.insert(id.to_owned())
    }));
    native
}
