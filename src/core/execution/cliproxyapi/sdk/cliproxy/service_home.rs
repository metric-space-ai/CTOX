// ref: sdk/cliproxy/service_home.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Instance-owned Home overlay and subscriber-lifetime coordination.
//!
//! Upstream binds this state machine to Redis, package globals and a concrete
//! Home client. CTOX injects its durable control/queue/config authorities and
//! keeps only lifecycle ordering here: stage -> commit -> runtime apply ->
//! plugin finalization -> publication. Generation tokens fence stale workers;
//! replacement and shutdown cancel them before releasing ownership.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::internal::config::ValidatedRuntimeConfig;
use crate::internal::homeplugins::{OperationContext, SyncError};

use super::executionregistry::{Registry, RegistryError, WaitBudget};
use super::home_plugins::{
    HomePluginConfigSnapshot, HomePluginControl, HomePluginControlError, HomePluginCoordinator,
    HomePluginFinalization, HomePluginRuntime, HomePluginStatusWork,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HomeLifecycleErrorKind {
    Cancelled,
    Stage,
    Commit,
    Runtime,
    Plugin,
    Publish,
    Drain,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HomeLifecycleError {
    pub kind: HomeLifecycleErrorKind,
    pub detail: String,
}

impl HomeLifecycleError {
    pub fn new(kind: HomeLifecycleErrorKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }

    fn cancelled() -> Self {
        Self::new(HomeLifecycleErrorKind::Cancelled, "home lifetime cancelled")
    }
}

impl fmt::Display for HomeLifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for HomeLifecycleError {}

impl From<RegistryError> for HomeLifecycleError {
    fn from(error: RegistryError) -> Self {
        Self::new(HomeLifecycleErrorKind::Drain, error.to_string())
    }
}

impl From<HomePluginControlError> for HomeLifecycleError {
    fn from(error: HomePluginControlError) -> Self {
        Self::new(HomeLifecycleErrorKind::Plugin, error.to_string())
    }
}

/// Cancellation capability shared by one subscriber generation and every
/// stage/finalization operation derived from it.
#[derive(Debug, Default)]
pub struct HomeCancellation(AtomicBool);

impl HomeCancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl OperationContext for HomeCancellation {
    fn is_cancelled(&self) -> bool {
        self.is_cancelled()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HomeOverlayInput {
    pub revision: u64,
    pub payload: Vec<u8>,
    pub plugin_config: HomePluginConfigSnapshot,
    pub observation_barrier_revision: i64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HomeOverlaySnapshot {
    pub revision: u64,
    pub payload: Vec<u8>,
    pub plugin_config: HomePluginConfigSnapshot,
    pub observation_barrier_revision: i64,
    /// Typed runtime projection produced by the injected Home config parser.
    /// Keeping it beside the opaque upstream payload lets the service-owned
    /// graph update routing without reparsing bytes or reaching for globals.
    pub runtime_config: Option<ValidatedRuntimeConfig>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HomeConfigCommit {
    pub revision: u64,
    pub runtime_revision: u64,
    pub runtime_config: Option<ValidatedRuntimeConfig>,
}

/// Typed config boundary owned by CTOX. `stage` performs merge/validation only;
/// it must not publish runtime state or credentials.
pub trait HomeConfigAuthority: Send + Sync {
    fn stage(
        &self,
        context: &HomeCancellation,
        input: &HomeOverlayInput,
    ) -> Result<HomeOverlaySnapshot, HomeLifecycleError>;

    fn commit(
        &self,
        context: &HomeCancellation,
        snapshot: &HomeOverlaySnapshot,
    ) -> Result<HomeConfigCommit, HomeLifecycleError>;

    fn apply_runtime(
        &self,
        context: &HomeCancellation,
        commit: &HomeConfigCommit,
    ) -> Result<(), HomeLifecycleError>;
}

/// Injected bounded retry/cancellation policy. Production implementations are
/// backed by the CTOX scheduler clock; tests need no wall-clock sleeps.
pub trait HomeRetryPolicy: Send + Sync {
    fn wait(&self, context: &HomeCancellation, attempt: u32) -> bool;
}

pub struct HomePublisherLease {
    pub generation: u64,
    pub registry: Arc<Registry>,
    pub cancellation: Arc<HomeCancellation>,
}

pub trait HomePublishedLifetime: Send + Sync {
    fn stop_and_wait(&self, budget: WaitBudget) -> Result<(), HomeLifecycleError>;
}

/// Publishes dispatch, in-flight observation and usage forwarding for exactly
/// one acknowledged generation. The returned handle pins every dependency
/// until replacement or shutdown waits for it.
pub trait HomePublisherAuthority: Send + Sync {
    fn publish(
        &self,
        lease: HomePublisherLease,
    ) -> Result<Arc<dyn HomePublishedLifetime>, HomeLifecycleError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HomeReplacementMode {
    /// Explicit replacement or ambiguous failover: cancel and drain all work.
    Drain,
    /// Safe heartbeat reconnect: settle pending dispatches and keep scopes.
    PreserveInFlight,
}

#[derive(Clone)]
pub struct HomeLifetime {
    generation: u64,
    cancellation: Arc<HomeCancellation>,
    registry: Arc<Registry>,
}

impl HomeLifetime {
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[must_use]
    pub fn cancellation(&self) -> Arc<HomeCancellation> {
        Arc::clone(&self.cancellation)
    }

    #[must_use]
    pub fn registry(&self) -> Arc<Registry> {
        Arc::clone(&self.registry)
    }
}

pub struct HomeOverlayWork {
    pub sequence: u64,
    pub snapshot: HomeOverlaySnapshot,
    pub plugin: HomePluginFinalization,
    pub commit: Option<HomeConfigCommit>,
    pub runtime_applied: bool,
    pub published: bool,
}

impl fmt::Debug for HomeOverlayWork {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HomeOverlayWork")
            .field("sequence", &self.sequence)
            .field("snapshot_revision", &self.snapshot.revision)
            .field("committed", &self.commit.is_some())
            .field("runtime_applied", &self.runtime_applied)
            .field("published", &self.published)
            .finish()
    }
}

/// FIFO work owner used by the injected subscriber adapter. Enqueue owns the
/// payload and assigns a monotonic sequence; dequeue never observes aliases.
#[derive(Default)]
pub struct HomeConfigWorkQueue {
    next: AtomicU64,
    items: Mutex<VecDeque<(u64, HomeOverlayInput)>>,
}

impl HomeConfigWorkQueue {
    pub fn enqueue(&self, input: HomeOverlayInput) -> u64 {
        let sequence = self.next.fetch_add(1, Ordering::AcqRel) + 1;
        self.items
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back((sequence, input));
        sequence
    }

    pub fn dequeue(&self, context: &HomeCancellation) -> Option<(u64, HomeOverlayInput)> {
        if context.is_cancelled() {
            return None;
        }
        self.items
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .filter(|_| !context.is_cancelled())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.items
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

struct ActiveLifetime {
    lifetime: HomeLifetime,
    published: Option<Arc<dyn HomePublishedLifetime>>,
    completed_sequence: u64,
}

#[derive(Default)]
struct Ownership {
    generation: u64,
    active: Option<ActiveLifetime>,
}

pub struct HomeLifecycleCoordinator {
    config: Arc<dyn HomeConfigAuthority>,
    plugin_control: Arc<dyn HomePluginControl>,
    plugin_runtime: Arc<dyn HomePluginRuntime>,
    publisher: Arc<dyn HomePublisherAuthority>,
    retry: Arc<dyn HomeRetryPolicy>,
    ownership: Mutex<Ownership>,
    transition: Mutex<()>,
    stage: Mutex<()>,
    commit: Mutex<()>,
    finalization: Mutex<()>,
    apply: Mutex<()>,
}

impl HomeLifecycleCoordinator {
    pub fn new(
        config: Arc<dyn HomeConfigAuthority>,
        plugin_control: Arc<dyn HomePluginControl>,
        plugin_runtime: Arc<dyn HomePluginRuntime>,
        publisher: Arc<dyn HomePublisherAuthority>,
        retry: Arc<dyn HomeRetryPolicy>,
    ) -> Self {
        Self {
            config,
            plugin_control,
            plugin_runtime,
            publisher,
            retry,
            ownership: Mutex::new(Ownership::default()),
            transition: Mutex::new(()),
            stage: Mutex::new(()),
            commit: Mutex::new(()),
            finalization: Mutex::new(()),
            apply: Mutex::new(()),
        }
    }

    pub fn start_lifetime(
        &self,
        mode: HomeReplacementMode,
        budget: WaitBudget,
    ) -> Result<HomeLifetime, HomeLifecycleError> {
        let _transition = lock(&self.transition);
        let previous = {
            let ownership = lock(&self.ownership);
            ownership
                .active
                .as_ref()
                .map(|active| active.lifetime.clone())
        };
        if let Some(previous) = &previous {
            // A commit already in progress wins. Cancellation is published
            // while holding the same gate, so no later commit can start.
            let _commit = lock(&self.commit);
            previous.cancellation.cancel();
        }
        // Stage/runtime/plugin authorities may need cancellation to unblock an
        // in-progress call. Replacement waits for each owner before detaching
        // the generation's dependencies.
        let _apply = lock(&self.apply);
        let _stage = lock(&self.stage);
        let _finalization = lock(&self.finalization);
        let mut old = lock(&self.ownership).active.take();
        let registry = if let Some(active) = old.as_mut() {
            if let Some(published) = active.published.take() {
                published.stop_and_wait(budget)?;
            }
            match mode {
                HomeReplacementMode::Drain => {
                    active.lifetime.registry.drain(budget)?;
                    Arc::new(Registry::new())
                }
                HomeReplacementMode::PreserveInFlight => {
                    active.lifetime.registry.wait_pending(budget)?;
                    Arc::clone(&active.lifetime.registry)
                }
            }
        } else {
            Arc::new(Registry::new())
        };
        let mut ownership = lock(&self.ownership);
        ownership.generation = ownership.generation.saturating_add(1);
        let lifetime = HomeLifetime {
            generation: ownership.generation,
            cancellation: Arc::new(HomeCancellation::default()),
            registry,
        };
        ownership.active = Some(ActiveLifetime {
            lifetime: lifetime.clone(),
            published: None,
            completed_sequence: 0,
        });
        Ok(lifetime)
    }

    pub fn shutdown(&self, budget: WaitBudget) -> Result<(), HomeLifecycleError> {
        let _transition = lock(&self.transition);
        let lifetime = {
            let ownership = lock(&self.ownership);
            ownership
                .active
                .as_ref()
                .map(|active| active.lifetime.clone())
        };
        let Some(lifetime) = lifetime else {
            return Ok(());
        };
        {
            let _commit = lock(&self.commit);
            lifetime.cancellation.cancel();
        }
        let _apply = lock(&self.apply);
        let _stage = lock(&self.stage);
        let _finalization = lock(&self.finalization);
        let mut active = lock(&self.ownership).active.take();
        if let Some(active) = active.as_mut() {
            if let Some(published) = active.published.take() {
                published.stop_and_wait(budget)?;
            }
            active.lifetime.registry.drain(budget)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn active_generation(&self) -> Option<u64> {
        lock(&self.ownership)
            .active
            .as_ref()
            .map(|active| active.lifetime.generation)
    }

    pub fn observe_barrier(
        &self,
        lifetime: &HomeLifetime,
        revision: i64,
    ) -> Result<(), HomeLifecycleError> {
        self.require_active(lifetime)?;
        lifetime.registry.observe_barrier(revision);
        Ok(())
    }

    pub fn stage_until_ready(
        &self,
        lifetime: &HomeLifetime,
        sequence: u64,
        input: &HomeOverlayInput,
    ) -> Result<HomeOverlayWork, HomeLifecycleError> {
        if input.observation_barrier_revision > 0 {
            self.observe_barrier(lifetime, input.observation_barrier_revision)?;
        }
        let mut attempt = 0;
        loop {
            self.require_active(lifetime)?;
            match self.stage_once(lifetime, sequence, input) {
                Ok(work) => return Ok(work),
                Err(error) if error.kind != HomeLifecycleErrorKind::Cancelled => {
                    attempt += 1;
                    if !self.retry.wait(&lifetime.cancellation, attempt) {
                        return if lifetime.cancellation.is_cancelled() {
                            Err(HomeLifecycleError::cancelled())
                        } else {
                            Err(error)
                        };
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn stage_once(
        &self,
        lifetime: &HomeLifetime,
        sequence: u64,
        input: &HomeOverlayInput,
    ) -> Result<HomeOverlayWork, HomeLifecycleError> {
        let _stage = lock(&self.stage);
        self.require_active(lifetime)?;
        let snapshot = self.config.stage(&lifetime.cancellation, input)?;
        self.require_active(lifetime)?;
        let plugins =
            HomePluginCoordinator::new(self.plugin_control.as_ref(), self.plugin_runtime.as_ref());
        let sync = plugins.sync(
            lifetime.cancellation.as_ref(),
            Some(&snapshot.plugin_config),
        );
        if let Some(error) = sync.error {
            return Err(HomeLifecycleError::new(
                HomeLifecycleErrorKind::Stage,
                error.to_string(),
            ));
        }
        let task_work = plugins.stage_tasks(
            lifetime.cancellation.as_ref(),
            Some(&snapshot.plugin_config),
        )?;
        self.require_active(lifetime)?;
        let mut plugin = HomePluginFinalization {
            task_work,
            sync_key: sync.sync_key,
            mark_synced: sync.attempted || !sync.report.task.trim().is_empty(),
            ..HomePluginFinalization::default()
        };
        if !sync.report.task.trim().is_empty() && !snapshot.plugin_config.node_id.trim().is_empty()
        {
            plugin.status_work.push(HomePluginStatusWork {
                config: snapshot.plugin_config.clone(),
                report: sync.report,
            });
        }
        Ok(HomeOverlayWork {
            sequence,
            snapshot,
            plugin,
            commit: None,
            runtime_applied: false,
            published: false,
        })
    }

    pub fn commit_finalize_until_done(
        &self,
        lifetime: &HomeLifetime,
        work: &mut HomeOverlayWork,
    ) -> Result<(), HomeLifecycleError> {
        let _apply = lock(&self.apply);
        self.require_active(lifetime)?;
        if lock(&self.ownership)
            .active
            .as_ref()
            .is_some_and(|active| work.sequence <= active.completed_sequence)
        {
            work.published = true;
            return Ok(());
        }
        if work.commit.is_none() {
            let _commit = lock(&self.commit);
            self.require_active(lifetime)?;
            work.commit = Some(self.config.commit(&lifetime.cancellation, &work.snapshot)?);
        }
        if !work.runtime_applied {
            let _finalization = lock(&self.finalization);
            self.require_active(lifetime)?;
            self.config.apply_runtime(
                &lifetime.cancellation,
                work.commit.as_ref().expect("commit retained"),
            )?;
            self.require_active(lifetime)?;
            work.runtime_applied = true;
        }
        let mut attempt = 0;
        loop {
            self.require_active(lifetime)?;
            let result = {
                let _finalization = lock(&self.finalization);
                self.require_active(lifetime)?;
                HomePluginCoordinator::new(
                    self.plugin_control.as_ref(),
                    self.plugin_runtime.as_ref(),
                )
                .finalize(lifetime.cancellation.as_ref(), Some(&mut work.plugin))
            };
            match result {
                Ok(()) => break,
                Err(error) => {
                    attempt += 1;
                    if !self.retry.wait(&lifetime.cancellation, attempt) {
                        return if lifetime.cancellation.is_cancelled() {
                            Err(HomeLifecycleError::cancelled())
                        } else {
                            Err(error.into())
                        };
                    }
                }
            }
        }
        if !work.published {
            self.publish_owned(lifetime)?;
            work.published = true;
        }
        let mut ownership = lock(&self.ownership);
        let active = ownership
            .active
            .as_mut()
            .ok_or_else(HomeLifecycleError::cancelled)?;
        if active.lifetime.generation != lifetime.generation {
            return Err(HomeLifecycleError::cancelled());
        }
        active.completed_sequence = active.completed_sequence.max(work.sequence);
        Ok(())
    }

    fn publish_owned(&self, lifetime: &HomeLifetime) -> Result<(), HomeLifecycleError> {
        let mut ownership = lock(&self.ownership);
        let Some(active) = ownership.active.as_mut() else {
            return Err(HomeLifecycleError::cancelled());
        };
        if active.lifetime.generation != lifetime.generation
            || !Arc::ptr_eq(&active.lifetime.cancellation, &lifetime.cancellation)
            || lifetime.cancellation.is_cancelled()
        {
            return Err(HomeLifecycleError::cancelled());
        }
        if active.published.is_none() {
            active.published = Some(self.publisher.publish(HomePublisherLease {
                generation: lifetime.generation,
                registry: Arc::clone(&lifetime.registry),
                cancellation: Arc::clone(&lifetime.cancellation),
            })?);
        }
        Ok(())
    }

    fn require_active(&self, lifetime: &HomeLifetime) -> Result<(), HomeLifecycleError> {
        if lifetime.cancellation.is_cancelled() {
            return Err(HomeLifecycleError::cancelled());
        }
        let ownership = lock(&self.ownership);
        let Some(active) = ownership.active.as_ref() else {
            return Err(HomeLifecycleError::cancelled());
        };
        if active.lifetime.generation == lifetime.generation
            && Arc::ptr_eq(&active.lifetime.cancellation, &lifetime.cancellation)
            && Arc::ptr_eq(&active.lifetime.registry, &lifetime.registry)
        {
            Ok(())
        } else {
            Err(HomeLifecycleError::cancelled())
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[must_use]
pub fn sync_error_to_home(error: SyncError) -> HomeLifecycleError {
    HomeLifecycleError::new(HomeLifecycleErrorKind::Plugin, error.to_string())
}
