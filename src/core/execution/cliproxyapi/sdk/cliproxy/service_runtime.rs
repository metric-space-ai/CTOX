// Origin: CTOX integration for sdk/cliproxy/service_executionregistry_test.go
// License: AGPL-3.0-only

//! Instance-owned convergence graph for Home and watcher runtime updates.
//!
//! The upstream service keeps these owners behind one Go `Service` mutex.
//! CTOX keeps the existing typed owners and binds them here: Home subscriber
//! generations, one reusable log-forwarder, config sequencing, selector
//! identity and watcher/Home apply serialization. Network and durable I/O
//! remain injected capabilities.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::internal::config::ValidatedRuntimeConfig;

use super::auth::{AccountRouter, CooldownStateStore};
use super::executionregistry::{Registry, WaitBudget};
use super::home_plugins::{HomePluginControl, HomePluginRuntime};
use super::service_auth::ServiceAuthRuntime;
use super::service_config::{
    normalized_routing_runtime_state, ConfigCommit, RoutingRuntimeState, ServiceConfigRuntime,
};
use super::service_home::{
    HomeConfigAuthority, HomeConfigCommit, HomeLifecycleCoordinator, HomeLifecycleError,
    HomeLifetime, HomeOverlayInput, HomeOverlaySnapshot, HomePublishedLifetime,
    HomePublisherAuthority, HomePublisherLease, HomeReplacementMode, HomeRetryPolicy,
};

#[derive(Clone)]
pub struct HomeLogBinding {
    pub generation: u64,
    pub registry: Arc<Registry>,
}

/// Injected log transport. Ownership and stale-generation fencing are handled
/// by `ServiceRuntimeGraph`, not by the transport implementation.
pub trait HomeLogForwarder: Send + Sync {
    fn bind(&self, binding: HomeLogBinding) -> Result<(), HomeLifecycleError>;
    fn deactivate(&self, generation: u64);
    fn stop(&self);
}

struct LogForwarderOwner {
    forwarder: Arc<dyn HomeLogForwarder>,
    generation: Mutex<Option<u64>>,
    stopped: AtomicBool,
}

impl LogForwarderOwner {
    fn bind(&self, binding: HomeLogBinding) -> Result<(), HomeLifecycleError> {
        self.forwarder.bind(binding.clone())?;
        *lock(&self.generation) = Some(binding.generation);
        Ok(())
    }

    fn deactivate(&self, generation: u64) {
        let mut active = lock(&self.generation);
        if *active == Some(generation) {
            self.forwarder.deactivate(generation);
            *active = None;
        }
    }

    fn stop(&self) {
        if !self.stopped.swap(true, Ordering::AcqRel) {
            self.forwarder.stop();
        }
    }
}

struct ForwardingPublisher {
    downstream: Arc<dyn HomePublisherAuthority>,
    logs: Arc<LogForwarderOwner>,
}

struct ForwardingPublished {
    generation: u64,
    downstream: Arc<dyn HomePublishedLifetime>,
    logs: Arc<LogForwarderOwner>,
}

impl HomePublishedLifetime for ForwardingPublished {
    fn stop_and_wait(&self, budget: WaitBudget) -> Result<(), HomeLifecycleError> {
        let result = self.downstream.stop_and_wait(budget);
        self.logs.deactivate(self.generation);
        result
    }
}

impl HomePublisherAuthority for ForwardingPublisher {
    fn publish(
        &self,
        lease: HomePublisherLease,
    ) -> Result<Arc<dyn HomePublishedLifetime>, HomeLifecycleError> {
        let generation = lease.generation;
        self.logs.bind(HomeLogBinding {
            generation,
            registry: Arc::clone(&lease.registry),
        })?;
        match self.downstream.publish(lease) {
            Ok(downstream) => Ok(Arc::new(ForwardingPublished {
                generation,
                downstream,
                logs: Arc::clone(&self.logs),
            })),
            Err(error) => {
                let active_generation = *lock(&self.logs.generation);
                if let Some(generation) = active_generation {
                    self.logs.deactivate(generation);
                }
                Err(error)
            }
        }
    }
}

struct SelectorOwner {
    routing: RoutingRuntimeState,
    generation: u64,
    router: Arc<AccountRouter>,
}

struct RuntimeApplyGraph {
    gate: Mutex<()>,
    config: Arc<ServiceConfigRuntime>,
    auth: Arc<ServiceAuthRuntime>,
    cooldowns: Arc<dyn CooldownStateStore>,
    selector: Mutex<SelectorOwner>,
}

impl RuntimeApplyGraph {
    fn apply_projection(&self, config: ValidatedRuntimeConfig) -> ConfigCommit {
        let commit = self.config.publish_validated_config(config.clone());
        self.auth.replace_config(config.clone());
        let routing = normalized_routing_runtime_state(&config);
        let mut selector = lock(&self.selector);
        if selector.routing != routing {
            selector.routing = routing;
            selector.generation = selector.generation.saturating_add(1);
            selector.router = Arc::new(AccountRouter::with_strategy(
                Arc::clone(&self.cooldowns),
                routing.strategy,
            ));
        }
        commit
    }

    fn apply_watcher(&self, config: ValidatedRuntimeConfig) -> ConfigCommit {
        let _gate = lock(&self.gate);
        self.apply_projection(config)
    }
}

struct SerializedHomeConfig {
    downstream: Arc<dyn HomeConfigAuthority>,
    apply: Arc<RuntimeApplyGraph>,
}

impl HomeConfigAuthority for SerializedHomeConfig {
    fn stage(
        &self,
        context: &super::service_home::HomeCancellation,
        input: &HomeOverlayInput,
    ) -> Result<HomeOverlaySnapshot, HomeLifecycleError> {
        self.downstream.stage(context, input)
    }

    fn commit(
        &self,
        context: &super::service_home::HomeCancellation,
        snapshot: &HomeOverlaySnapshot,
    ) -> Result<HomeConfigCommit, HomeLifecycleError> {
        self.downstream.commit(context, snapshot)
    }

    fn apply_runtime(
        &self,
        context: &super::service_home::HomeCancellation,
        commit: &HomeConfigCommit,
    ) -> Result<(), HomeLifecycleError> {
        let _gate = lock(&self.apply.gate);
        self.downstream.apply_runtime(context, commit)?;
        if let Some(config) = commit.runtime_config.clone() {
            self.apply.apply_projection(config);
        }
        Ok(())
    }
}

pub struct ServiceRuntimeGraph {
    home: HomeLifecycleCoordinator,
    apply: Arc<RuntimeApplyGraph>,
    logs: Arc<LogForwarderOwner>,
}

impl ServiceRuntimeGraph {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        initial_config: ValidatedRuntimeConfig,
        auth: Arc<ServiceAuthRuntime>,
        cooldowns: Arc<dyn CooldownStateStore>,
        home_config: Arc<dyn HomeConfigAuthority>,
        plugin_control: Arc<dyn HomePluginControl>,
        plugin_runtime: Arc<dyn HomePluginRuntime>,
        publisher: Arc<dyn HomePublisherAuthority>,
        retry: Arc<dyn HomeRetryPolicy>,
        log_forwarder: Arc<dyn HomeLogForwarder>,
    ) -> Self {
        let routing = normalized_routing_runtime_state(&initial_config);
        let selector = SelectorOwner {
            routing,
            generation: 1,
            router: Arc::new(AccountRouter::with_strategy(
                Arc::clone(&cooldowns),
                routing.strategy,
            )),
        };
        let apply = Arc::new(RuntimeApplyGraph {
            gate: Mutex::new(()),
            config: Arc::new(ServiceConfigRuntime::new(initial_config)),
            auth,
            cooldowns,
            selector: Mutex::new(selector),
        });
        let logs = Arc::new(LogForwarderOwner {
            forwarder: log_forwarder,
            generation: Mutex::new(None),
            stopped: AtomicBool::new(false),
        });
        let home = HomeLifecycleCoordinator::new(
            Arc::new(SerializedHomeConfig {
                downstream: home_config,
                apply: Arc::clone(&apply),
            }),
            plugin_control,
            plugin_runtime,
            Arc::new(ForwardingPublisher {
                downstream: publisher,
                logs: Arc::clone(&logs),
            }),
            retry,
        );
        Self { home, apply, logs }
    }

    pub fn start_home_lifetime(
        &self,
        mode: HomeReplacementMode,
        budget: WaitBudget,
    ) -> Result<HomeLifetime, HomeLifecycleError> {
        self.home.start_lifetime(mode, budget)
    }

    pub fn apply_home_overlay(
        &self,
        lifetime: &HomeLifetime,
        sequence: u64,
        input: &HomeOverlayInput,
    ) -> Result<(), HomeLifecycleError> {
        let mut work = self.home.stage_until_ready(lifetime, sequence, input)?;
        self.home.commit_finalize_until_done(lifetime, &mut work)
    }

    pub fn apply_watcher_config(&self, config: ValidatedRuntimeConfig) -> ConfigCommit {
        self.apply.apply_watcher(config)
    }

    pub fn selector(&self) -> Arc<AccountRouter> {
        Arc::clone(&lock(&self.apply.selector).router)
    }

    pub fn selector_generation(&self) -> u64 {
        lock(&self.apply.selector).generation
    }

    pub fn current_config(&self) -> ConfigCommit {
        self.apply.config.current()
    }

    pub fn shutdown(&self, budget: WaitBudget) -> Result<(), HomeLifecycleError> {
        let result = self.home.shutdown(budget);
        self.logs.stop();
        result
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
