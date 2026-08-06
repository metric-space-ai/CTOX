// ref: sdk/cliproxy/home_plugins_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};

use super::home_plugins::*;
use super::service_home::*;
use crate::internal::homeplugins::{
    current_platform, OperationContext, PluginInstallStatus, SyncError, SyncOutcome, SyncReport,
};
use crate::sdk::pluginstore::{PluginSyncRequest, PluginSyncResponse};

#[derive(Default)]
struct Context(AtomicBool);

impl OperationContext for Context {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[derive(Default)]
pub(super) struct Control {
    fetch: Mutex<Option<Result<PluginSyncResponse, HomePluginControlError>>>,
    tasks: Mutex<Vec<HomePluginTask>>,
    pushes: AtomicUsize,
    fail_pushes: AtomicUsize,
    fetches: AtomicUsize,
}

impl HomePluginControl for Control {
    fn fetch_sync(
        &self,
        _request: &mut PluginSyncRequest,
    ) -> Result<PluginSyncResponse, HomePluginControlError> {
        self.fetches.fetch_add(1, Ordering::SeqCst);
        self.fetch
            .lock()
            .unwrap()
            .take()
            .unwrap_or_else(|| Ok(PluginSyncResponse::default()))
    }

    fn push_status(&self, _report: &SyncReport) -> Result<(), HomePluginControlError> {
        let attempt = self.pushes.fetch_add(1, Ordering::SeqCst);
        if attempt < self.fail_pushes.load(Ordering::SeqCst) {
            Err(HomePluginControlError::new(
                HomePluginControlErrorKind::Rejected,
                "blocked",
            ))
        } else {
            Ok(())
        }
    }

    fn plugin_tasks(&self) -> Result<Vec<HomePluginTask>, HomePluginControlError> {
        Ok(self.tasks.lock().unwrap().clone())
    }
}

#[derive(Default)]
pub(super) struct Runtime {
    resolved: AtomicUsize,
    fallback: AtomicUsize,
    deletes: AtomicUsize,
    fail_resolved: AtomicBool,
}

impl Runtime {
    fn report(ok: bool) -> SyncReport {
        let now = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        SyncReport {
            schema_version: 1,
            task_id: 0,
            task: "plugin-sync".to_owned(),
            node_id: String::new(),
            status: if ok { "success" } else { "failed" }.to_owned(),
            phase: "install".to_owned(),
            ok,
            started_at: now,
            finished_at: Some(now),
            updated_at: now,
            platform: current_platform(),
            plugins: Vec::<PluginInstallStatus>::new(),
            error: if ok {
                String::new()
            } else {
                "failed".to_owned()
            },
        }
    }
}

impl HomePluginRuntime for Runtime {
    fn empty_report(&self) -> SyncReport {
        let mut report = Self::report(true);
        report.task.clear();
        report
    }

    fn installed_versions(
        &self,
        _config: &HomePluginConfigSnapshot,
    ) -> Result<HashMap<String, String>, SyncError> {
        Ok(HashMap::from([("sample".to_owned(), "1.0.0".to_owned())]))
    }

    fn sync_resolved(
        &self,
        _context: &dyn OperationContext,
        _config: &HomePluginConfigSnapshot,
        _response: &mut PluginSyncResponse,
        _installed_versions: &HashMap<String, String>,
    ) -> SyncOutcome {
        self.resolved.fetch_add(1, Ordering::SeqCst);
        if self.fail_resolved.load(Ordering::SeqCst) {
            let error = SyncError::new("plugin runtime sync failed");
            return SyncOutcome {
                report: Self::report(false),
                error: Some(error),
            };
        }
        SyncOutcome {
            report: Self::report(true),
            error: None,
        }
    }

    fn sync_fallback(
        &self,
        _context: &dyn OperationContext,
        _config: &HomePluginConfigSnapshot,
    ) -> SyncOutcome {
        self.fallback.fetch_add(1, Ordering::SeqCst);
        SyncOutcome {
            report: Self::report(true),
            error: None,
        }
    }

    fn completed_report(&self, error: Option<SyncError>) -> SyncReport {
        Self::report(error.is_none())
    }

    fn delete_with_report(
        &self,
        _context: &dyn OperationContext,
        _config: &HomePluginConfigSnapshot,
        task: &HomePluginTask,
    ) -> SyncReport {
        self.deletes.fetch_add(1, Ordering::SeqCst);
        let mut report = Self::report(true);
        report.task_id = task.id;
        report
    }
}

pub(super) fn config() -> HomePluginConfigSnapshot {
    HomePluginConfigSnapshot {
        home_enabled: true,
        node_id: "node-1".to_owned(),
        plugins_enabled: true,
        plugins_dir: " plugins ".to_owned(),
        ..HomePluginConfigSnapshot::default()
    }
}

#[derive(Default)]
pub(super) struct ConfigAuthority {
    pub stages: AtomicUsize,
    pub commits: AtomicUsize,
    pub runtime_applies: AtomicUsize,
    pub fail_stage: AtomicBool,
    pub commit_revisions: Mutex<Vec<u64>>,
}

impl HomeConfigAuthority for ConfigAuthority {
    fn stage(
        &self,
        context: &HomeCancellation,
        input: &HomeOverlayInput,
    ) -> Result<HomeOverlaySnapshot, HomeLifecycleError> {
        self.stages.fetch_add(1, Ordering::SeqCst);
        if context.is_cancelled() {
            return Err(HomeLifecycleError::new(
                HomeLifecycleErrorKind::Cancelled,
                "cancelled",
            ));
        }
        if self.fail_stage.load(Ordering::SeqCst) {
            return Err(HomeLifecycleError::new(
                HomeLifecycleErrorKind::Stage,
                "stage failed",
            ));
        }
        Ok(HomeOverlaySnapshot {
            revision: input.revision,
            payload: input.payload.clone(),
            plugin_config: input.plugin_config.clone(),
            observation_barrier_revision: input.observation_barrier_revision,
            runtime_config: None,
        })
    }

    fn commit(
        &self,
        context: &HomeCancellation,
        snapshot: &HomeOverlaySnapshot,
    ) -> Result<HomeConfigCommit, HomeLifecycleError> {
        if context.is_cancelled() {
            return Err(HomeLifecycleError::new(
                HomeLifecycleErrorKind::Cancelled,
                "cancelled",
            ));
        }
        self.commits.fetch_add(1, Ordering::SeqCst);
        self.commit_revisions
            .lock()
            .unwrap()
            .push(snapshot.revision);
        Ok(HomeConfigCommit {
            revision: snapshot.revision,
            runtime_revision: snapshot.revision,
            runtime_config: snapshot.runtime_config.clone(),
        })
    }

    fn apply_runtime(
        &self,
        context: &HomeCancellation,
        _commit: &HomeConfigCommit,
    ) -> Result<(), HomeLifecycleError> {
        if context.is_cancelled() {
            return Err(HomeLifecycleError::new(
                HomeLifecycleErrorKind::Cancelled,
                "cancelled",
            ));
        }
        self.runtime_applies.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

pub(super) struct Retry {
    pub allow: AtomicBool,
    pub waits: AtomicUsize,
}

impl Default for Retry {
    fn default() -> Self {
        Self {
            allow: AtomicBool::new(false),
            waits: AtomicUsize::new(0),
        }
    }
}

impl HomeRetryPolicy for Retry {
    fn wait(&self, context: &HomeCancellation, _attempt: u32) -> bool {
        self.waits.fetch_add(1, Ordering::SeqCst);
        self.allow.load(Ordering::SeqCst) && !context.is_cancelled()
    }
}

#[derive(Default)]
pub(super) struct Publisher {
    pub publishes: AtomicUsize,
    pub stops: Arc<AtomicUsize>,
}

struct Published(Arc<AtomicUsize>);

impl HomePublishedLifetime for Published {
    fn stop_and_wait(
        &self,
        _budget: super::executionregistry::WaitBudget,
    ) -> Result<(), HomeLifecycleError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

impl HomePublisherAuthority for Publisher {
    fn publish(
        &self,
        _lease: HomePublisherLease,
    ) -> Result<Arc<dyn HomePublishedLifetime>, HomeLifecycleError> {
        self.publishes.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(Published(self.stops.clone())))
    }
}

pub(super) fn input(revision: u64) -> HomeOverlayInput {
    HomeOverlayInput {
        revision,
        payload: format!("revision={revision}").into_bytes(),
        plugin_config: config(),
        observation_barrier_revision: revision as i64,
    }
}

#[derive(Default)]
struct BlockingControl {
    entered: AtomicBool,
    gate: Mutex<bool>,
    changed: Condvar,
}

impl BlockingControl {
    fn wait_until_entered(&self) {
        let started = std::time::Instant::now();
        while !self.entered.load(Ordering::Acquire) {
            assert!(started.elapsed() < Duration::from_secs(2));
            std::thread::yield_now();
        }
    }

    fn release(&self) {
        *self.gate.lock().unwrap() = true;
        self.changed.notify_all();
    }
}

impl HomePluginControl for BlockingControl {
    fn fetch_sync(
        &self,
        _request: &mut PluginSyncRequest,
    ) -> Result<PluginSyncResponse, HomePluginControlError> {
        Ok(PluginSyncResponse::default())
    }

    fn push_status(&self, _report: &SyncReport) -> Result<(), HomePluginControlError> {
        self.entered.store(true, Ordering::Release);
        let mut released = self.gate.lock().unwrap();
        while !*released {
            released = self.changed.wait(released).unwrap();
        }
        Ok(())
    }

    fn plugin_tasks(&self) -> Result<Vec<HomePluginTask>, HomePluginControlError> {
        Ok(Vec::new())
    }
}

#[test]
fn sync_skips_unchanged_signature_only_after_finalization_marks_it() {
    let control = Arc::new(Control::default());
    let runtime = Arc::new(Runtime::default());
    let coordinator = HomePluginCoordinator::new(control.as_ref(), runtime.as_ref());
    let config = config();
    let first = coordinator.sync(&Context::default(), Some(&config));
    assert!(first.attempted);
    assert_eq!(control.fetches.load(Ordering::SeqCst), 1);
    assert_eq!(coordinator.synced_key(), "");
    let mut work = HomePluginFinalization {
        sync_key: first.sync_key.clone(),
        mark_synced: true,
        ..HomePluginFinalization::default()
    };
    coordinator
        .finalize(&Context::default(), Some(&mut work))
        .unwrap();
    let second = coordinator.sync(&Context::default(), Some(&config));
    assert!(!second.attempted);
    assert_eq!(second.report.task, "");
    assert_eq!(control.fetches.load(Ordering::SeqCst), 1);
}

#[test]
fn fetch_failure_returns_failure_report_without_marking_or_attempt_credit() {
    let control = Control::default();
    *control.fetch.lock().unwrap() = Some(Err(HomePluginControlError::new(
        HomePluginControlErrorKind::Unavailable,
        "plugin sync unavailable",
    )));
    let runtime = Runtime::default();
    let coordinator = HomePluginCoordinator::new(&control, &runtime);
    let result = coordinator.sync(&Context::default(), Some(&config()));
    assert!(!result.attempted);
    assert!(!result.report.ok);
    assert_eq!(result.error.unwrap().to_string(), "plugin sync unavailable");
    assert_eq!(coordinator.synced_key(), "");
}

#[test]
fn unsupported_control_protocol_uses_runtime_fallback() {
    let control = Control::default();
    *control.fetch.lock().unwrap() = Some(Err(HomePluginControlError::new(
        HomePluginControlErrorKind::SyncUnsupported,
        "unsupported",
    )));
    let runtime = Runtime::default();
    let coordinator = HomePluginCoordinator::new(&control, &runtime);
    let result = coordinator.sync(&Context::default(), Some(&config()));
    assert!(result.attempted);
    assert!(result.error.is_none());
    assert_eq!(runtime.fallback.load(Ordering::SeqCst), 1);
    assert_eq!(runtime.resolved.load(Ordering::SeqCst), 0);
}

#[test]
fn plugins_disabled_skips_fetch_and_still_has_stable_key() {
    let control = Control::default();
    let runtime = Runtime::default();
    let coordinator = HomePluginCoordinator::new(&control, &runtime);
    let mut config = config();
    config.plugins_enabled = false;
    let result = coordinator.sync(&Context::default(), Some(&config));
    assert!(!result.attempted);
    assert!(!result.sync_key.is_empty());
    assert_eq!(control.fetches.load(Ordering::SeqCst), 0);
}

#[test]
fn status_retry_does_not_advance_or_mark_until_success() {
    let control = Control::default();
    control.fail_pushes.store(1, Ordering::SeqCst);
    let runtime = Runtime::default();
    let coordinator = HomePluginCoordinator::new(&control, &runtime);
    let mut work = HomePluginFinalization {
        status_work: vec![HomePluginStatusWork {
            config: config(),
            report: Runtime::report(true),
        }],
        sync_key: "sync-key".to_owned(),
        mark_synced: true,
        ..HomePluginFinalization::default()
    };
    assert!(coordinator
        .finalize(&Context::default(), Some(&mut work))
        .is_err());
    assert_eq!(work.next_status, 0);
    assert!(work.mark_synced);
    assert_eq!(coordinator.synced_key(), "");
    coordinator
        .finalize(&Context::default(), Some(&mut work))
        .unwrap();
    assert_eq!(work.next_status, 1);
    assert!(!work.mark_synced);
    assert_eq!(coordinator.synced_key(), "sync-key");
}

#[test]
fn task_staging_filters_operations_and_defers_delete() {
    let control = Control::default();
    *control.tasks.lock().unwrap() = vec![
        HomePluginTask {
            id: 7,
            operation: " DELETE ".to_owned(),
            plugin_id: "plugin-a".to_owned(),
        },
        HomePluginTask {
            id: 8,
            operation: "install".to_owned(),
            plugin_id: "plugin-b".to_owned(),
        },
    ];
    let runtime = Runtime::default();
    let coordinator = HomePluginCoordinator::new(&control, &runtime);
    let tasks = coordinator
        .stage_tasks(&Context::default(), Some(&config()))
        .unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(runtime.deletes.load(Ordering::SeqCst), 0);
}

#[test]
fn failed_task_report_retry_never_repeats_delete() {
    let control = Control::default();
    control.fail_pushes.store(1, Ordering::SeqCst);
    let runtime = Runtime::default();
    let coordinator = HomePluginCoordinator::new(&control, &runtime);
    let mut work = HomePluginFinalization {
        task_work: vec![HomePluginTaskWork {
            config: config(),
            task: HomePluginTask {
                id: 8,
                operation: "delete".to_owned(),
                plugin_id: "plugin-b".to_owned(),
            },
            report: None,
        }],
        ..HomePluginFinalization::default()
    };
    assert!(coordinator
        .finalize(&Context::default(), Some(&mut work))
        .is_err());
    assert_eq!(runtime.deletes.load(Ordering::SeqCst), 1);
    assert!(work.task_work[0].report.is_some());
    coordinator
        .finalize(&Context::default(), Some(&mut work))
        .unwrap();
    assert_eq!(runtime.deletes.load(Ordering::SeqCst), 1);
    assert_eq!(work.next_task, 1);
}

#[test]
fn cancellation_prevents_finalization_and_sync_marking() {
    let control = Control::default();
    let runtime = Runtime::default();
    let coordinator = HomePluginCoordinator::new(&control, &runtime);
    let context = Context::default();
    context.0.store(true, Ordering::SeqCst);
    let mut work = HomePluginFinalization {
        sync_key: "sync-key".to_owned(),
        mark_synced: true,
        ..HomePluginFinalization::default()
    };
    assert!(coordinator.finalize(&context, Some(&mut work)).is_err());
    assert!(work.mark_synced);
    assert_eq!(coordinator.synced_key(), "");
}

#[test]
fn sync_key_is_ordered_and_includes_credential_revision_and_raw_config() {
    let mut config = config();
    config.plugins.insert(
        " z ".to_owned(),
        HomePluginInstanceSnapshot {
            enabled: Some(true),
            priority: 2,
            raw_yaml: b"key: value\n".to_vec(),
        },
    );
    config.plugins.insert(
        "a".to_owned(),
        HomePluginInstanceSnapshot {
            enabled: None,
            priority: 1,
            raw_yaml: Vec::new(),
        },
    );
    let first = home_plugin_sync_key(Some(&config));
    let same = home_plugin_sync_key(Some(&config.clone()));
    assert_eq!(first, same);
    config.auth_revision = 2;
    assert_ne!(first, home_plugin_sync_key(Some(&config)));
    config.auth_revision = 0;
    config.plugins.get_mut(" z ").unwrap().raw_yaml.push(b'!');
    assert_ne!(first, home_plugin_sync_key(Some(&config)));
}

#[test]
fn disabled_plugin_report_is_skipped_only_after_successful_finalization() {
    let control = Control::default();
    let runtime = Runtime::default();
    let coordinator = HomePluginCoordinator::new(&control, &runtime);
    let mut disabled = config();
    disabled.plugins_enabled = false;
    let first = coordinator.sync(&Context::default(), Some(&disabled));
    assert!(!first.sync_key.is_empty());
    let mut work = HomePluginFinalization {
        sync_key: first.sync_key,
        mark_synced: true,
        ..HomePluginFinalization::default()
    };
    coordinator
        .finalize(&Context::default(), Some(&mut work))
        .unwrap();
    let second = coordinator.sync(&Context::default(), Some(&disabled));
    assert!(second.report.task.is_empty());
    assert_eq!(control.fetches.load(Ordering::SeqCst), 0);
}

#[test]
fn overlay_plugin_stage_failure_never_commits_or_applies_config() {
    let control = Arc::new(Control::default());
    let runtime = Arc::new(Runtime::default());
    runtime.fail_resolved.store(true, Ordering::SeqCst);
    let config_authority = Arc::new(ConfigAuthority::default());
    let publisher = Arc::new(Publisher::default());
    let retry = Arc::new(Retry::default());
    let coordinator = HomeLifecycleCoordinator::new(
        config_authority.clone(),
        control.clone(),
        runtime.clone(),
        publisher.clone(),
        retry.clone(),
    );
    let lifetime = coordinator
        .start_lifetime(
            HomeReplacementMode::Drain,
            super::executionregistry::WaitBudget::unbounded(),
        )
        .unwrap();
    assert!(coordinator
        .stage_until_ready(&lifetime, 1, &input(1))
        .is_err());
    assert_eq!(config_authority.commits.load(Ordering::SeqCst), 0);
    assert_eq!(config_authority.runtime_applies.load(Ordering::SeqCst), 0);
    assert_eq!(publisher.publishes.load(Ordering::SeqCst), 0);
}

#[test]
fn initial_overlay_stages_without_publishing_plugin_or_config_writes() {
    let control = Arc::new(Control::default());
    let runtime = Arc::new(Runtime::default());
    let config_authority = Arc::new(ConfigAuthority::default());
    let publisher = Arc::new(Publisher::default());
    let retry = Arc::new(Retry::default());
    let coordinator = HomeLifecycleCoordinator::new(
        config_authority.clone(),
        control.clone(),
        runtime.clone(),
        publisher.clone(),
        retry.clone(),
    );
    let lifetime = coordinator
        .start_lifetime(
            HomeReplacementMode::Drain,
            super::executionregistry::WaitBudget::unbounded(),
        )
        .unwrap();
    let work = coordinator
        .stage_until_ready(&lifetime, 1, &input(1))
        .unwrap();
    assert!(work.commit.is_none());
    assert_eq!(config_authority.commits.load(Ordering::SeqCst), 0);
    assert_eq!(control.pushes.load(Ordering::SeqCst), 0);
    assert_eq!(publisher.publishes.load(Ordering::SeqCst), 0);
}

#[test]
fn acknowledged_lifetime_retries_status_then_commits_and_publishes_once() {
    let control = Arc::new(Control::default());
    control.fail_pushes.store(1, Ordering::SeqCst);
    let runtime = Arc::new(Runtime::default());
    let config_authority = Arc::new(ConfigAuthority::default());
    let publisher = Arc::new(Publisher::default());
    let retry = Arc::new(Retry::default());
    retry.allow.store(true, Ordering::SeqCst);
    let coordinator = HomeLifecycleCoordinator::new(
        config_authority.clone(),
        control.clone(),
        runtime.clone(),
        publisher.clone(),
        retry.clone(),
    );
    let lifetime = coordinator
        .start_lifetime(
            HomeReplacementMode::Drain,
            super::executionregistry::WaitBudget::unbounded(),
        )
        .unwrap();
    let mut work = coordinator
        .stage_until_ready(&lifetime, 1, &input(1))
        .unwrap();
    coordinator
        .commit_finalize_until_done(&lifetime, &mut work)
        .unwrap();
    coordinator
        .commit_finalize_until_done(&lifetime, &mut work)
        .unwrap();
    assert_eq!(config_authority.commits.load(Ordering::SeqCst), 1);
    assert_eq!(config_authority.runtime_applies.load(Ordering::SeqCst), 1);
    assert_eq!(control.pushes.load(Ordering::SeqCst), 2);
    assert_eq!(publisher.publishes.load(Ordering::SeqCst), 1);
    assert!(work.published);
}

#[test]
fn replacement_waits_for_plugin_finalization_ownership() {
    let control = Arc::new(BlockingControl::default());
    let runtime = Arc::new(Runtime::default());
    let config_authority = Arc::new(ConfigAuthority::default());
    let publisher = Arc::new(Publisher::default());
    let retry = Arc::new(Retry::default());
    let coordinator = HomeLifecycleCoordinator::new(
        config_authority.clone(),
        control.clone(),
        runtime.clone(),
        publisher.clone(),
        retry.clone(),
    );
    let lifetime = coordinator
        .start_lifetime(
            HomeReplacementMode::Drain,
            super::executionregistry::WaitBudget::unbounded(),
        )
        .unwrap();
    let mut work = coordinator
        .stage_until_ready(&lifetime, 1, &input(1))
        .unwrap();
    std::thread::scope(|thread| {
        let (worker_tx, worker_rx) = std::sync::mpsc::channel();
        let coordinator_ref = &coordinator;
        let lifetime_ref = &lifetime;
        let work_ref = &mut work;
        thread.spawn(move || {
            worker_tx
                .send(coordinator_ref.commit_finalize_until_done(lifetime_ref, work_ref))
                .unwrap();
        });
        control.wait_until_entered();
        let (replace_tx, replace_rx) = std::sync::mpsc::channel();
        let coordinator_ref = &coordinator;
        thread.spawn(move || {
            replace_tx
                .send(coordinator_ref.start_lifetime(
                    HomeReplacementMode::PreserveInFlight,
                    super::executionregistry::WaitBudget::unbounded(),
                ))
                .unwrap();
        });
        while !lifetime.cancellation().is_cancelled() {
            std::thread::yield_now();
        }
        assert!(replace_rx.try_recv().is_err());
        control.release();
        assert!(worker_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .is_err());
        assert!(replace_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .is_ok());
    });
}

#[test]
fn shutdown_cancels_and_waits_for_blocked_plugin_finalization() {
    let control = Arc::new(BlockingControl::default());
    let runtime = Arc::new(Runtime::default());
    let config_authority = Arc::new(ConfigAuthority::default());
    let publisher = Arc::new(Publisher::default());
    let retry = Arc::new(Retry::default());
    let coordinator = HomeLifecycleCoordinator::new(
        config_authority.clone(),
        control.clone(),
        runtime.clone(),
        publisher.clone(),
        retry.clone(),
    );
    let lifetime = coordinator
        .start_lifetime(
            HomeReplacementMode::Drain,
            super::executionregistry::WaitBudget::unbounded(),
        )
        .unwrap();
    let mut work = coordinator
        .stage_until_ready(&lifetime, 1, &input(1))
        .unwrap();
    std::thread::scope(|thread| {
        let coordinator_ref = &coordinator;
        let lifetime_ref = &lifetime;
        let work_ref = &mut work;
        thread.spawn(move || {
            let _ = coordinator_ref.commit_finalize_until_done(lifetime_ref, work_ref);
        });
        control.wait_until_entered();
        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel();
        let coordinator_ref = &coordinator;
        thread.spawn(move || {
            shutdown_tx
                .send(coordinator_ref.shutdown(super::executionregistry::WaitBudget::unbounded()))
                .unwrap();
        });
        while !lifetime.cancellation().is_cancelled() {
            std::thread::yield_now();
        }
        assert!(shutdown_rx.try_recv().is_err());
        control.release();
        assert!(shutdown_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .is_ok());
    });
    assert_eq!(coordinator.active_generation(), None);
}

// Upstream's `forceHomeRuntimeConfig` clears a raw `Plugins.StoreAuth` field.
// CTOX's `HomeOverlayInput` cannot carry credential material at all; secret
// authority remains outside this coordinator, so that literal mutation test is
// a reasoned exclusion rather than a synthetic field added for parity.
