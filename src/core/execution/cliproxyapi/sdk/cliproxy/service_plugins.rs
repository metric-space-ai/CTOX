// ref: sdk/cliproxy/service_plugins.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::internal::registry::ModelRefreshSink;

use super::auth::{Auth, AuthKind, AuthSourceKind, ProviderExecutorRegistration};
use super::builder::PluginHost;
use super::model_registry::ModelInfo;
use super::providers::LoadContext;
use super::service_auth::{ServiceAuthError, ServiceAuthRuntime};
use super::service_executors::{openai_compat_info_from_auth, ExecutorRegistrationOptions};
use super::types::PluginAuthParser;

pub const MODEL_REGISTRATION_MAX_WORKERS_PER_CATEGORY: usize = 5;
pub const MODEL_REGISTRATION_MAX_WORKERS_OPENAI_COMPATIBILITY: usize = 20;

pub trait ServicePluginRuntime: PluginHost + PluginAuthParser {
    fn executor_registrations(&self) -> Vec<Arc<ProviderExecutorRegistration>>;
    fn has_executor_candidate_provider(&self, provider: &str) -> bool;
    fn owns_executor(&self, registration: &Arc<ProviderExecutorRegistration>) -> bool;
    fn models_for_provider(&self, provider: &str) -> Vec<ModelInfo>;
    fn register_models(&self, registry: Arc<dyn super::model_registry::ModelRegistry>);
    fn register_usage_plugins(&self);
    fn install_translator_hooks(&self);
    fn refresh_management_routes(&self);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ModelRegistrationPhase {
    ConfigApiKey,
    Other,
}

pub struct ModelRegistrationTask {
    pub phase: ModelRegistrationPhase,
    pub category: String,
    run: Option<Box<dyn FnOnce() + Send>>,
}

impl fmt::Debug for ModelRegistrationTask {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelRegistrationTask")
            .field("phase", &self.phase)
            .field("category", &self.category)
            .field("has_run", &self.run.is_some())
            .finish()
    }
}

impl ModelRegistrationTask {
    #[must_use]
    pub fn new(
        phase: ModelRegistrationPhase,
        category: impl Into<String>,
        run: impl FnOnce() + Send + 'static,
    ) -> Self {
        Self {
            phase,
            category: category.into(),
            run: Some(Box::new(run)),
        }
    }
}

impl ServiceAuthRuntime {
    pub fn bind_model_refresh_sink(self: &Arc<Self>, sink: &ModelRefreshSink) {
        let runtime = Arc::downgrade(self);
        sink.set_callback(Some(Arc::new(move |providers| {
            let Some(runtime) = runtime.upgrade() else {
                return;
            };
            let providers = providers
                .into_iter()
                .map(|provider| provider.trim().to_ascii_lowercase())
                .filter(|provider| !provider.is_empty())
                .collect::<std::collections::BTreeSet<_>>();
            let auths = runtime
                .auth_manager()
                .lifecycle()
                .snapshot_cached()
                .into_iter()
                .filter(|auth| {
                    !auth.disabled && providers.contains(&auth.provider.trim().to_ascii_lowercase())
                })
                .collect();
            let _ = runtime.register_models_for_auth_batch(auths);
        })));
    }

    pub fn sync_plugin_runtime(&self) -> Result<bool, ServiceAuthError> {
        if !self.sync_plugin_runtime_config()? {
            return Ok(false);
        }
        self.sync_plugin_model_runtime()?;
        Ok(true)
    }

    pub fn sync_plugin_runtime_config(&self) -> Result<bool, ServiceAuthError> {
        let Some(plugin) = self.plugin_runtime() else {
            return Ok(false);
        };
        let config = self.config();
        plugin
            .apply_config(&config)
            .map_err(|_| ServiceAuthError::AuthManager)?;
        plugin
            .register_frontend_auth_providers()
            .map_err(|_| ServiceAuthError::AuthManager)?;
        self.access_manager()
            .set_shared_providers(&plugin.access_providers());
        plugin.register_usage_plugins();
        plugin.install_translator_hooks();
        plugin.refresh_management_routes();
        if let Some(watcher) = self.watcher() {
            watcher.set_plugin_auth_parser(plugin.clone());
        }
        Ok(true)
    }

    pub fn sync_plugin_model_runtime(&self) -> Result<(), ServiceAuthError> {
        let Some(plugin) = self.plugin_runtime() else {
            return Ok(());
        };
        plugin.register_models(self.model_registry());
        let auths = self.auth_manager().lifecycle().snapshot_cached();
        self.register_available_executors(ExecutorRegistrationOptions {
            include_plugins: true,
            auths: auths.clone(),
            ..ExecutorRegistrationOptions::default()
        })?;
        self.register_models_for_auth_batch(auths)?;
        self.auth_manager().refresh_scheduler_all()?;
        Ok(())
    }

    pub fn register_models_for_auth_batch(&self, auths: Vec<Auth>) -> Result<(), ServiceAuthError> {
        let error = Mutex::new(None);
        run_auth_registration_batch(&LoadContext::default(), auths, |auth| {
            if error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_none()
            {
                if let Err(cause) = self.complete_model_registration_for_auth(auth) {
                    *error
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cause);
                }
            }
        });
        error
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .map_or(Ok(()), Err)
    }
}

fn run_auth_registration_batch(
    context: &LoadContext,
    auths: Vec<Auth>,
    run: impl Fn(&Auth) + Sync,
) {
    let mut phases = BTreeMap::<ModelRegistrationPhase, Vec<Auth>>::new();
    for auth in auths {
        phases
            .entry(model_registration_phase(&auth))
            .or_default()
            .push(auth);
    }
    for phase in [
        ModelRegistrationPhase::ConfigApiKey,
        ModelRegistrationPhase::Other,
    ] {
        let Some(auths) = phases.remove(&phase) else {
            continue;
        };
        let mut order = Vec::new();
        let mut groups = BTreeMap::<String, VecDeque<Auth>>::new();
        for auth in auths {
            let category = model_registration_category(&auth);
            if !groups.contains_key(&category) {
                order.push(category.clone());
            }
            groups.entry(category).or_default().push_back(auth);
        }
        std::thread::scope(|scope| {
            for category in order {
                let queue = Arc::new(Mutex::new(groups.remove(&category).unwrap_or_default()));
                let worker_count = queue
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .len()
                    .min(model_registration_max_workers_for_category(&category));
                let run = &run;
                for _ in 0..worker_count {
                    let queue = Arc::clone(&queue);
                    scope.spawn(move || loop {
                        if context.is_cancelled() {
                            return;
                        }
                        let auth = queue
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .pop_front();
                        let Some(auth) = auth else {
                            return;
                        };
                        run(&auth);
                    });
                }
            }
        });
        if context.is_cancelled() {
            return;
        }
    }
}

#[must_use]
pub fn model_registration_phase(auth: &Auth) -> ModelRegistrationPhase {
    if auth.auth_kind() == Some(AuthKind::ApiKey)
        && auth.auth_source_kind() == Some(AuthSourceKind::Config)
    {
        ModelRegistrationPhase::ConfigApiKey
    } else {
        ModelRegistrationPhase::Other
    }
}

#[must_use]
pub fn model_registration_category(auth: &Auth) -> String {
    let (compat, _, detected) = openai_compat_info_from_auth(auth);
    let provider = if detected && !compat.is_empty() {
        compat
    } else if auth.provider.trim().is_empty() {
        "unknown".to_owned()
    } else {
        auth.provider.trim().to_ascii_lowercase()
    };
    match auth.auth_kind() {
        Some(kind) => format!("{provider}:{}", kind.as_str()),
        None => provider,
    }
}

#[must_use]
pub fn model_registration_max_workers_for_category(category: &str) -> usize {
    let category = category.trim().to_ascii_lowercase();
    if category.starts_with("openai-compatible-") || category.starts_with("openai-compatibility") {
        MODEL_REGISTRATION_MAX_WORKERS_OPENAI_COMPATIBILITY
    } else {
        MODEL_REGISTRATION_MAX_WORKERS_PER_CATEGORY
    }
}

pub fn run_model_registration_tasks(context: &LoadContext, tasks: Vec<ModelRegistrationTask>) {
    if tasks.is_empty() || context.is_cancelled() {
        return;
    }
    let mut phases = BTreeMap::<ModelRegistrationPhase, Vec<ModelRegistrationTask>>::new();
    for task in tasks {
        phases.entry(task.phase).or_default().push(task);
    }
    for phase in [
        ModelRegistrationPhase::ConfigApiKey,
        ModelRegistrationPhase::Other,
    ] {
        let Some(tasks) = phases.remove(&phase) else {
            continue;
        };
        run_model_registration_task_phase(context, tasks);
        if context.is_cancelled() {
            return;
        }
    }
}

fn run_model_registration_task_phase(context: &LoadContext, tasks: Vec<ModelRegistrationTask>) {
    let mut order = Vec::new();
    let mut groups = BTreeMap::<String, VecDeque<ModelRegistrationTask>>::new();
    for mut task in tasks {
        if task.run.is_none() {
            continue;
        }
        task.category = task.category.trim().to_ascii_lowercase();
        if task.category.is_empty() {
            task.category = "unknown".to_owned();
        }
        if !groups.contains_key(&task.category) {
            order.push(task.category.clone());
        }
        groups
            .entry(task.category.clone())
            .or_default()
            .push_back(task);
    }

    std::thread::scope(|scope| {
        for category in order {
            let queue = Arc::new(Mutex::new(groups.remove(&category).unwrap_or_default()));
            let worker_count = queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len()
                .min(model_registration_max_workers_for_category(&category));
            for _ in 0..worker_count {
                let queue = Arc::clone(&queue);
                scope.spawn(move || loop {
                    if context.is_cancelled() {
                        return;
                    }
                    let task = queue
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .pop_front();
                    let Some(mut task) = task else {
                        return;
                    };
                    if let Some(run) = task.run.take() {
                        run();
                    }
                });
            }
        }
    });
}
