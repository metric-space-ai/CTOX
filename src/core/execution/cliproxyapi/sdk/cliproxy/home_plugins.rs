// ref: sdk/cliproxy/home_plugins.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Home plugin synchronization coordinator.
//!
//! Upstream couples this state machine to its Redis-backed Home client. CTOX
//! already owns durable control-plane state, so this port keeps the ordering,
//! retry, fallback and deduplication contract behind injected boundaries. It
//! deliberately does not recreate Home configuration or a Redis authority.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use crate::internal::homeplugins::{
    current_platform, OperationContext, SyncError, SyncOutcome, SyncReport,
};
use crate::sdk::pluginstore::{PluginSyncRequest, PluginSyncResponse, PLUGIN_SYNC_SCHEMA_VERSION};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HomePluginInstanceSnapshot {
    pub enabled: Option<bool>,
    pub priority: i32,
    /// The configuration boundary supplies upstream-compatible YAML bytes.
    /// Keeping encoding outside the coordinator avoids a second config parser.
    pub raw_yaml: Vec<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HomePluginConfigSnapshot {
    pub home_enabled: bool,
    pub node_id: String,
    pub plugins_enabled: bool,
    pub plugins_dir: String,
    pub auth_revision: u64,
    pub plugins: BTreeMap<String, HomePluginInstanceSnapshot>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HomePluginTask {
    pub id: u64,
    pub operation: String,
    pub plugin_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HomePluginControlErrorKind {
    Unavailable,
    SyncUnsupported,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HomePluginControlError {
    pub kind: HomePluginControlErrorKind,
    pub message: String,
}

impl HomePluginControlError {
    pub fn new(kind: HomePluginControlErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for HomePluginControlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for HomePluginControlError {}

pub trait HomePluginControl: Send + Sync {
    fn fetch_sync(
        &self,
        request: &mut PluginSyncRequest,
    ) -> Result<PluginSyncResponse, HomePluginControlError>;

    fn push_status(&self, report: &SyncReport) -> Result<(), HomePluginControlError>;

    fn plugin_tasks(&self) -> Result<Vec<HomePluginTask>, HomePluginControlError>;
}

/// Injected adapter to the already-ported plugin installation implementation.
pub trait HomePluginRuntime: Send + Sync {
    /// Go's zero report used when an already-finalized signature is skipped.
    fn empty_report(&self) -> SyncReport;

    fn installed_versions(
        &self,
        config: &HomePluginConfigSnapshot,
    ) -> Result<HashMap<String, String>, SyncError>;

    fn sync_resolved(
        &self,
        context: &dyn OperationContext,
        config: &HomePluginConfigSnapshot,
        response: &mut PluginSyncResponse,
        installed_versions: &HashMap<String, String>,
    ) -> SyncOutcome;

    fn sync_fallback(
        &self,
        context: &dyn OperationContext,
        config: &HomePluginConfigSnapshot,
    ) -> SyncOutcome;

    fn completed_report(&self, error: Option<SyncError>) -> SyncReport;

    fn delete_with_report(
        &self,
        context: &dyn OperationContext,
        config: &HomePluginConfigSnapshot,
        task: &HomePluginTask,
    ) -> SyncReport;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HomePluginSyncResult {
    pub report: SyncReport,
    pub sync_key: String,
    pub attempted: bool,
    pub error: Option<SyncError>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HomePluginStatusWork {
    pub config: HomePluginConfigSnapshot,
    pub report: SyncReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HomePluginTaskWork {
    pub config: HomePluginConfigSnapshot,
    pub task: HomePluginTask,
    pub report: Option<SyncReport>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HomePluginFinalization {
    pub status_work: Vec<HomePluginStatusWork>,
    pub next_status: usize,
    pub task_work: Vec<HomePluginTaskWork>,
    pub next_task: usize,
    pub sync_key: String,
    pub mark_synced: bool,
}

pub struct HomePluginCoordinator<'a> {
    control: &'a dyn HomePluginControl,
    runtime: &'a dyn HomePluginRuntime,
    synced_key: Mutex<String>,
}

impl<'a> HomePluginCoordinator<'a> {
    pub fn new(control: &'a dyn HomePluginControl, runtime: &'a dyn HomePluginRuntime) -> Self {
        Self {
            control,
            runtime,
            synced_key: Mutex::new(String::new()),
        }
    }

    pub fn synced_key(&self) -> String {
        self.synced_key
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn sync(
        &self,
        context: &dyn OperationContext,
        config: Option<&HomePluginConfigSnapshot>,
    ) -> HomePluginSyncResult {
        let Some(config) = config.filter(|config| config.home_enabled) else {
            return HomePluginSyncResult {
                report: self.runtime.empty_report(),
                sync_key: String::new(),
                attempted: false,
                error: None,
            };
        };
        let sync_key = home_plugin_sync_key(Some(config));
        if !sync_key.is_empty() && self.synced_key() == sync_key {
            return HomePluginSyncResult {
                report: self.runtime.empty_report(),
                sync_key,
                attempted: false,
                error: None,
            };
        }
        if !config.plugins_enabled {
            return HomePluginSyncResult {
                report: self.runtime.completed_report(None),
                sync_key,
                attempted: false,
                error: None,
            };
        }
        if context.is_cancelled() {
            let error = SyncError::new("home plugin sync cancelled");
            return HomePluginSyncResult {
                report: self.runtime.completed_report(Some(error.clone())),
                sync_key,
                attempted: false,
                error: Some(error),
            };
        }
        let installed_versions = match self.runtime.installed_versions(config) {
            Ok(versions) => versions,
            Err(error) => {
                return HomePluginSyncResult {
                    report: self.runtime.completed_report(Some(error.clone())),
                    sync_key,
                    attempted: false,
                    error: Some(error),
                };
            }
        };
        let platform = current_platform();
        let mut request = PluginSyncRequest {
            schema_version: PLUGIN_SYNC_SCHEMA_VERSION,
            goos: platform.goos,
            goarch: platform.goarch,
            installed_versions,
        };
        let fetched = self.control.fetch_sync(&mut request);
        match fetched {
            Ok(mut response) => {
                let outcome = self.runtime.sync_resolved(
                    context,
                    config,
                    &mut response,
                    &request.installed_versions,
                );
                response.clear();
                request.clear();
                outcome_to_result(outcome, sync_key, true)
            }
            Err(error) if error.kind == HomePluginControlErrorKind::SyncUnsupported => {
                request.clear();
                outcome_to_result(self.runtime.sync_fallback(context, config), sync_key, true)
            }
            Err(error) => {
                request.clear();
                let error = SyncError::new(error.to_string());
                HomePluginSyncResult {
                    report: self.runtime.completed_report(Some(error.clone())),
                    sync_key,
                    attempted: false,
                    error: Some(error),
                }
            }
        }
    }

    pub fn mark_synced(&self, sync_key: &str) {
        if sync_key.trim().is_empty() {
            return;
        }
        *self
            .synced_key
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = sync_key.to_owned();
    }

    pub fn stage_tasks(
        &self,
        context: &dyn OperationContext,
        config: Option<&HomePluginConfigSnapshot>,
    ) -> Result<Vec<HomePluginTaskWork>, HomePluginControlError> {
        let Some(config) = config.filter(|config| config.home_enabled) else {
            return Ok(Vec::new());
        };
        if context.is_cancelled() {
            return Err(HomePluginControlError::new(
                HomePluginControlErrorKind::Rejected,
                "home plugin task staging cancelled",
            ));
        }
        Ok(self
            .control
            .plugin_tasks()?
            .into_iter()
            .filter(|task| task.operation.trim().eq_ignore_ascii_case("delete"))
            .map(|task| HomePluginTaskWork {
                config: config.clone(),
                task,
                report: None,
            })
            .collect())
    }

    /// Advances only after a status write succeeds. A delete result is retained
    /// before reporting, so retrying a failed report never repeats the delete.
    pub fn finalize(
        &self,
        context: &dyn OperationContext,
        work: Option<&mut HomePluginFinalization>,
    ) -> Result<(), HomePluginControlError> {
        let Some(work) = work else {
            return Ok(());
        };
        reject_cancelled(context)?;
        while work.next_status < work.status_work.len() {
            reject_cancelled(context)?;
            let status = &work.status_work[work.next_status];
            let mut report = status.report.clone();
            report.node_id = status.config.node_id.trim().to_owned();
            if report.node_id.is_empty() {
                return Err(HomePluginControlError::new(
                    HomePluginControlErrorKind::Rejected,
                    "home node id is empty",
                ));
            }
            self.control.push_status(&report)?;
            work.next_status += 1;
        }
        while work.next_task < work.task_work.len() {
            reject_cancelled(context)?;
            let task = &mut work.task_work[work.next_task];
            if task.report.is_none() {
                task.report = Some(self.runtime.delete_with_report(
                    context,
                    &task.config,
                    &task.task,
                ));
            }
            let mut report = task.report.clone().expect("delete report retained");
            report.node_id = task.config.node_id.trim().to_owned();
            if report.node_id.is_empty() {
                return Err(HomePluginControlError::new(
                    HomePluginControlErrorKind::Rejected,
                    "home node id is empty",
                ));
            }
            self.control.push_status(&report)?;
            work.next_task += 1;
        }
        if work.mark_synced {
            reject_cancelled(context)?;
            self.mark_synced(&work.sync_key);
            work.mark_synced = false;
        }
        Ok(())
    }
}

fn reject_cancelled(context: &dyn OperationContext) -> Result<(), HomePluginControlError> {
    if context.is_cancelled() {
        Err(HomePluginControlError::new(
            HomePluginControlErrorKind::Rejected,
            "home plugin finalization cancelled",
        ))
    } else {
        Ok(())
    }
}

fn outcome_to_result(
    outcome: SyncOutcome,
    sync_key: String,
    attempted: bool,
) -> HomePluginSyncResult {
    HomePluginSyncResult {
        report: outcome.report,
        sync_key,
        attempted,
        error: outcome.error,
    }
}

pub fn home_plugin_sync_key(config: Option<&HomePluginConfigSnapshot>) -> String {
    let Some(config) = config.filter(|config| config.home_enabled) else {
        return String::new();
    };
    let mut hash = Sha256::new();
    hash.update(
        format!(
            "enabled={}\ndir={}\nauth-revision={}\n",
            config.plugins_enabled,
            config.plugins_dir.trim(),
            config.auth_revision
        )
        .as_bytes(),
    );
    for (id, item) in &config.plugins {
        hash.update(
            format!(
                "plugin={}\nenabled={}\npriority={}\n",
                id.trim(),
                item.enabled.unwrap_or(false),
                item.priority
            )
            .as_bytes(),
        );
        hash.update(&item.raw_yaml);
        hash.update(b"\n");
    }
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
