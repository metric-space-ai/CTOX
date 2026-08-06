// ref: internal/homeplugins/sync.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Home-managed plugin synchronization.
//!
//! Upstream obtains configuration, HTTP clients, the wall clock, and home-directory
//! expansion through package globals. CTOX keeps the same planning/reporting and
//! filesystem semantics behind instance-owned, injected boundaries. The store
//! implementation can therefore be supplied by the eventual `sdk::pluginstore`
//! port without introducing ambient credentials or mutable process globals.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::internal::pluginstore::update_available;
pub use crate::sdk::pluginstore::{
    Artifact, InstallPlan, InstallResult, Manifest, Platform, PluginSyncItem, ResolvedAuthConfig,
};
use crate::sdk::pluginstore::{Client as PluginStoreClient, PluginStoreIo};

const PLUGIN_TASK_NAME: &str = "plugin-sync";
const PLUGIN_DELETE_TASK_NAME: &str = "plugin-delete";
const PLUGIN_TASK_STATUS_OK: &str = "success";
const PLUGIN_TASK_STATUS_ERROR: &str = "failed";
const PLUGIN_TASK_PHASE_INSTALL: &str = "install";
const PLUGIN_TASK_PHASE_LOAD: &str = "load";
const PLUGIN_TASK_PHASE_DELETE: &str = "delete";

const PLUGIN_INSTALL_STATUS_INSTALLED: &str = "installed";
const PLUGIN_INSTALL_STATUS_SKIPPED: &str = "skipped";
const PLUGIN_INSTALL_STATUS_FAILED: &str = "failed";
const PLUGIN_INSTALL_STATUS_DELETED: &str = "deleted";
const PLUGIN_INSTALL_STATUS_MISSING: &str = "missing";
const PLUGIN_LOAD_STATUS_LOADED: &str = "loaded";
const PLUGIN_LOAD_STATUS_FAILED: &str = "failed";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncError(String);

impl SyncError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for SyncError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for SyncError {}

impl From<std::io::Error> for SyncError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncReport {
    pub schema_version: i32,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub task_id: u64,
    pub task: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub node_id: String,
    pub status: String,
    pub phase: String,
    pub ok: bool,
    pub started_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
    pub platform: Platform,
    pub plugins: Vec<PluginInstallStatus>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
}

/// Rust representation of Go's `(SyncReport, error)` return pair. Keeping the
/// report on failure is load-bearing because Home reports partial per-plugin
/// progress even when the aggregate operation fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncOutcome {
    pub report: SyncReport,
    pub error: Option<SyncError>,
}

impl SyncOutcome {
    pub fn into_result(self) -> Result<SyncReport, SyncError> {
        self.error.map_or(Ok(self.report), Err)
    }
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginInstallStatus {
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub version: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub release_tag: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub repository: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub install_type: String,
    pub install_status: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub load_status: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub path: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub skipped: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub overwritten: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PluginInstanceConfig {
    pub enabled: Option<bool>,
    pub store: Option<Manifest>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SyncConfig {
    pub home_enabled: bool,
    pub plugins_enabled: bool,
    pub plugins_dir: String,
    pub plugins: BTreeMap<String, PluginInstanceConfig>,
}

pub struct ResolvedSyncRequest<'a> {
    pub config: Option<&'a SyncConfig>,
    pub items: &'a mut [PluginSyncItem],
    pub expires_at: DateTime<Utc>,
    pub installed_versions: &'a HashMap<String, String>,
    pub runtime: Option<&'a dyn PluginRuntime>,
}

pub trait OperationContext: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

#[derive(Debug, Default)]
pub struct ActiveContext;

impl OperationContext for ActiveContext {
    fn is_cancelled(&self) -> bool {
        false
    }
}

pub trait PluginRuntime: Send + Sync {
    fn plugin_busy(&self, id: &str) -> bool;
    fn unload_plugin(&self, context: &dyn OperationContext, id: &str) -> bool;
}

pub trait PluginLoadInspector {
    fn plugin_registered(&self, id: &str) -> bool;
}

pub struct InstallRequest<'a> {
    pub manifest: &'a Manifest,
    pub plugins_dir: &'a Path,
    pub platform: &'a Platform,
    pub plugin_loaded: bool,
    pub resolved_auth: &'a [ResolvedAuthConfig],
    pub expires_at: Option<DateTime<Utc>>,
}

struct InstallInvocation<'a> {
    manifest: &'a Manifest,
    auth: &'a [ResolvedAuthConfig],
    expires_at: Option<DateTime<Utc>>,
    root: &'a Path,
    platform: &'a Platform,
    runtime: Option<&'a dyn PluginRuntime>,
}

pub trait PluginStore: Send + Sync {
    fn install_manifest(
        &self,
        context: &dyn OperationContext,
        request: InstallRequest<'_>,
    ) -> Result<InstallResult, SyncError>;
}

/// Bridges Home's bounded sync request to the public SDK store facade without
/// duplicating SDK DTOs or retaining per-response credentials.
pub struct SdkPluginStoreAdapter {
    io: Arc<dyn PluginStoreIo>,
    registry_url: String,
}

impl SdkPluginStoreAdapter {
    pub fn new(io: Arc<dyn PluginStoreIo>, registry_url: impl Into<String>) -> Self {
        Self {
            io,
            registry_url: registry_url.into().trim().to_owned(),
        }
    }
}

impl PluginStore for SdkPluginStoreAdapter {
    fn install_manifest(
        &self,
        context: &dyn OperationContext,
        request: InstallRequest<'_>,
    ) -> Result<InstallResult, SyncError> {
        if context.is_cancelled() {
            return Err(SyncError::new("context canceled"));
        }
        let client = PluginStoreClient::with_resolved_auth_expiry(
            Arc::clone(&self.io),
            &self.registry_url,
            request.resolved_auth.to_vec(),
            request.expires_at,
        );
        let loaded = request.plugin_loaded;
        let options = crate::sdk::pluginstore::InstallOptions {
            plugins_dir: request.plugins_dir.to_path_buf(),
            goos: request.platform.goos.clone(),
            goarch: request.platform.goarch.clone(),
            plugin_loaded: Some(Arc::new(move || loaded)),
            before_write: None,
        };
        client
            .install_manifest(request.manifest, &options)
            .map_err(|error| SyncError::new(error.to_string()))
    }
}

/// Injected CTOX control-plane boundary. Implementations may resolve paths from
/// typed runtime configuration; the default deliberately refuses ambient `~`
/// expansion instead of consulting HOME/USERPROFILE.
pub trait HomePluginBoundary: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
    fn resolve_plugins_dir(&self, configured: &str) -> Result<PathBuf, SyncError>;
}

#[derive(Debug, Default)]
pub struct StrictHostBoundary;

impl HomePluginBoundary for StrictHostBoundary {
    fn now(&self) -> DateTime<Utc> {
        DateTime::<Utc>::from(std::time::SystemTime::now())
    }

    fn resolve_plugins_dir(&self, configured: &str) -> Result<PathBuf, SyncError> {
        let configured = configured.trim();
        if configured.starts_with('~') {
            return Err(SyncError::new(
                "resolve plugins directory: home directory is unavailable",
            ));
        }
        Ok(if configured.is_empty() {
            PathBuf::from("plugins")
        } else {
            PathBuf::from(configured)
        })
    }
}

pub struct HomePluginSync<'a> {
    store: &'a dyn PluginStore,
    boundary: &'a dyn HomePluginBoundary,
}

impl<'a> HomePluginSync<'a> {
    pub fn new(store: &'a dyn PluginStore, boundary: &'a dyn HomePluginBoundary) -> Self {
        Self { store, boundary }
    }

    pub fn sync_platform(
        &self,
        context: &dyn OperationContext,
        config: Option<&SyncConfig>,
        runtime: Option<&dyn PluginRuntime>,
        platform: Platform,
    ) -> Result<(), SyncError> {
        self.sync_platform_with_report(context, config, runtime, platform)
            .into_result()
            .map(|_| ())
    }

    pub fn sync_platform_with_report(
        &self,
        context: &dyn OperationContext,
        config: Option<&SyncConfig>,
        runtime: Option<&dyn PluginRuntime>,
        platform: Platform,
    ) -> SyncOutcome {
        let Some(config) = enabled_config(config) else {
            return SyncOutcome {
                report: self.new_sync_report(platform),
                error: None,
            };
        };
        let platform = normalize_platform(platform);
        let mut report = self.new_sync_report(platform.clone());
        if platform.goos.is_empty() {
            return self.fail(report, SyncError::new("home plugins: goos is required"));
        }
        if platform.goarch.is_empty() {
            return self.fail(report, SyncError::new("home plugins: goarch is required"));
        }
        let root = match self.boundary.resolve_plugins_dir(&config.plugins_dir) {
            Ok(root) => root,
            Err(error) => return self.fail(report, prefix_error("home plugins", error)),
        };
        let mut errors = Vec::new();
        for (id, item) in &config.plugins {
            if !plugin_config_enabled(item) {
                continue;
            }
            let manifest = match store_manifest_from_plugin_config(id, item) {
                Ok(Some(manifest)) => manifest,
                Ok(None) => continue,
                Err(error) => {
                    report.plugins.push(PluginInstallStatus {
                        id: id.trim().to_owned(),
                        install_status: PLUGIN_INSTALL_STATUS_FAILED.to_owned(),
                        error: error.to_string(),
                        ..PluginInstallStatus::default()
                    });
                    errors.push(error);
                    continue;
                }
            };
            let mut status = plugin_status_from_manifest(&manifest);
            match self.install_manifest(
                context,
                InstallInvocation {
                    manifest: &manifest,
                    auth: &[],
                    expires_at: None,
                    root: &root,
                    platform: &platform,
                    runtime,
                },
            ) {
                Ok(result) => apply_install_result(&mut status, result),
                Err(error) => {
                    status.install_status = PLUGIN_INSTALL_STATUS_FAILED.to_owned();
                    status.error = error.to_string();
                    errors.push(error);
                }
            }
            report.plugins.push(status);
        }
        self.finish_report(&mut report, join_errors(&errors).as_ref());
        SyncOutcome {
            report,
            error: join_errors(&errors),
        }
    }

    pub fn sync_resolved_with_report(
        &self,
        context: &dyn OperationContext,
        request: ResolvedSyncRequest<'_>,
    ) -> SyncOutcome {
        let ResolvedSyncRequest {
            config,
            items,
            expires_at,
            installed_versions,
            runtime,
        } = request;
        let platform = normalize_platform(current_platform());
        let mut report = self.new_sync_report(platform.clone());
        let result = if let Some(config) = enabled_config(config) {
            match self.boundary.resolve_plugins_dir(&config.plugins_dir) {
                Err(error) => Err(prefix_error("home plugins", error)),
                Ok(root) => {
                    add_installed_version_statuses(&mut report, config, &root, installed_versions);
                    let mut errors = Vec::new();
                    for item in items.iter_mut() {
                        if self.boundary.now() >= expires_at {
                            errors
                                .push(SyncError::new("home plugins: plugin sync response expired"));
                            break;
                        }
                        let manifest = item.manifest.clone();
                        let mut status = plugin_status_from_manifest(&manifest);
                        match self.install_manifest(
                            context,
                            InstallInvocation {
                                manifest: &manifest,
                                auth: &item.auth,
                                expires_at: Some(expires_at),
                                root: &root,
                                platform: &platform,
                                runtime,
                            },
                        ) {
                            Ok(install) => apply_install_result(&mut status, install),
                            Err(error) => {
                                status.install_status = PLUGIN_INSTALL_STATUS_FAILED.to_owned();
                                status.error = error.to_string();
                                errors.push(error);
                            }
                        }
                        upsert_plugin_install_status(&mut report, status);
                        item.clear();
                    }
                    if let Some(error) = join_errors(&errors) {
                        Err(error)
                    } else {
                        Ok(())
                    }
                }
            }
        } else {
            Ok(())
        };
        for item in items.iter_mut() {
            item.clear();
        }
        match result {
            Ok(()) => {
                self.finish_report(&mut report, None);
                SyncOutcome {
                    report,
                    error: None,
                }
            }
            Err(error) => self.fail(report, error),
        }
    }

    pub fn installed_versions(
        &self,
        config: Option<&SyncConfig>,
    ) -> Result<HashMap<String, String>, SyncError> {
        let Some(config) = config else {
            return Ok(HashMap::new());
        };
        let root = self
            .boundary
            .resolve_plugins_dir(&config.plugins_dir)
            .map_err(|error| prefix_error("home plugins", error))?;
        let mut versions = HashMap::new();
        for id in config.plugins.keys() {
            let files = plugin_file_infos(&root, id).map_err(|error| {
                SyncError::new(format!(
                    "home plugins: discover installed plugin {id}: {error}"
                ))
            })?;
            if let Some(file) = files.first() {
                if !file.version.trim().is_empty() {
                    versions.insert(id.trim().to_owned(), file.version.trim().to_owned());
                }
            }
        }
        Ok(versions)
    }

    pub fn delete_with_report(
        &self,
        context: &dyn OperationContext,
        config: Option<&SyncConfig>,
        runtime: Option<&dyn PluginRuntime>,
        task_id: u64,
        plugin_id: &str,
    ) -> SyncReport {
        let mut report = self.new_sync_report(current_platform());
        report.task_id = task_id;
        report.task = PLUGIN_DELETE_TASK_NAME.to_owned();
        report.phase = PLUGIN_TASK_PHASE_DELETE.to_owned();
        let mut status = PluginInstallStatus {
            id: plugin_id.trim().to_owned(),
            ..PluginInstallStatus::default()
        };
        let outcome = if context.is_cancelled() {
            Err(SyncError::new("context canceled"))
        } else if let Some(config) = config {
            self.boundary
                .resolve_plugins_dir(&config.plugins_dir)
                .map_err(|error| prefix_error("home plugins", error))
                .map(|root| delete_plugin_artifact(context, &root, plugin_id, runtime))
        } else {
            Err(SyncError::new("home plugins: config is nil"))
        };
        let task_error = match outcome {
            Ok(outcome) => {
                status.path = outcome.path.to_string_lossy().trim().to_owned();
                if let Some(error) = outcome.error {
                    status.install_status = PLUGIN_INSTALL_STATUS_FAILED.to_owned();
                    status.error = error.to_string();
                    Some(error)
                } else {
                    status.install_status = if outcome.deleted {
                        PLUGIN_INSTALL_STATUS_DELETED
                    } else {
                        PLUGIN_INSTALL_STATUS_MISSING
                    }
                    .to_owned();
                    None
                }
            }
            Err(error) => {
                status.install_status = PLUGIN_INSTALL_STATUS_FAILED.to_owned();
                status.error = error.to_string();
                Some(error)
            }
        };
        report.plugins.push(status);
        self.finish_report(&mut report, task_error.as_ref());
        report
    }

    fn install_manifest(
        &self,
        context: &dyn OperationContext,
        invocation: InstallInvocation<'_>,
    ) -> Result<InstallResult, SyncError> {
        let InstallInvocation {
            manifest,
            auth,
            expires_at,
            root,
            platform,
            runtime,
        } = invocation;
        let id = manifest.id.trim();
        if id.is_empty() {
            return Err(SyncError::new("home plugins: manifest plugin id is empty"));
        }
        manifest
            .validate()
            .map_err(|error| SyncError::new(format!("home plugins: install {id}: {error}")))?;
        self.store
            .install_manifest(
                context,
                InstallRequest {
                    manifest,
                    plugins_dir: root,
                    platform,
                    plugin_loaded: runtime.is_some_and(|runtime| runtime.plugin_busy(id)),
                    resolved_auth: auth,
                    expires_at,
                },
            )
            .map_err(|error| SyncError::new(format!("home plugins: install {id}: {error}")))
    }

    fn new_sync_report(&self, platform: Platform) -> SyncReport {
        let now = self.boundary.now();
        SyncReport {
            schema_version: 1,
            task_id: 0,
            task: PLUGIN_TASK_NAME.to_owned(),
            node_id: String::new(),
            status: PLUGIN_TASK_STATUS_OK.to_owned(),
            phase: PLUGIN_TASK_PHASE_INSTALL.to_owned(),
            ok: true,
            started_at: now,
            finished_at: None,
            updated_at: now,
            platform: normalize_platform(platform),
            plugins: Vec::new(),
            error: String::new(),
        }
    }

    fn finish_report(&self, report: &mut SyncReport, error: Option<&SyncError>) {
        let now = self.boundary.now();
        report.finished_at = Some(now);
        report.updated_at = now;
        report.ok = error.is_none();
        if let Some(error) = error {
            report.status = PLUGIN_TASK_STATUS_ERROR.to_owned();
            report.error = error.to_string();
        } else {
            report.status = PLUGIN_TASK_STATUS_OK.to_owned();
            report.error.clear();
        }
    }

    fn fail(&self, mut report: SyncReport, error: SyncError) -> SyncOutcome {
        self.finish_report(&mut report, Some(&error));
        SyncOutcome {
            report,
            error: Some(error),
        }
    }

    pub fn completed_sync_report(
        &self,
        platform: Platform,
        error: Option<SyncError>,
    ) -> SyncReport {
        let mut report = self.new_sync_report(platform);
        self.finish_report(&mut report, error.as_ref());
        report
    }
}

pub fn current_platform() -> Platform {
    Platform {
        goos: match std::env::consts::OS {
            "macos" => "darwin",
            os => os,
        }
        .to_owned(),
        goarch: match std::env::consts::ARCH {
            "x86_64" => "amd64",
            "aarch64" => "arm64",
            arch => arch,
        }
        .to_owned(),
    }
}

pub fn normalize_platform(platform: Platform) -> Platform {
    let goos = match platform.goos.trim().to_ascii_lowercase().as_str() {
        "mac" | "macos" | "osx" => "darwin".to_owned(),
        value => value.to_owned(),
    };
    let goarch = match platform.goarch.trim().to_ascii_lowercase().as_str() {
        "x64" | "x86_64" => "amd64".to_owned(),
        "aarch64" => "arm64".to_owned(),
        value => value.to_owned(),
    };
    Platform { goos, goarch }
}

pub fn mark_load_results(
    report: Option<&mut SyncReport>,
    inspector: Option<&dyn PluginLoadInspector>,
    now: DateTime<Utc>,
) -> Result<(), SyncError> {
    let Some(report) = report else {
        return Ok(());
    };
    report.phase = PLUGIN_TASK_PHASE_LOAD.to_owned();
    let preserve_sync_error = !report.ok && !report.error.trim().is_empty();
    let mut errors = Vec::new();
    if preserve_sync_error {
        errors.push(SyncError::new(report.error.clone()));
    }
    for status in &mut report.plugins {
        if status.install_status == PLUGIN_INSTALL_STATUS_FAILED {
            if status.load_status.is_empty() {
                status.load_status = PLUGIN_INSTALL_STATUS_SKIPPED.to_owned();
            }
            if !preserve_sync_error {
                errors.push(if status.error.trim().is_empty() {
                    SyncError::new(format!("home plugins: plugin {} install failed", status.id))
                } else {
                    SyncError::new(status.error.clone())
                });
            }
        } else if inspector.is_some_and(|inspector| inspector.plugin_registered(&status.id)) {
            status.load_status = PLUGIN_LOAD_STATUS_LOADED.to_owned();
        } else {
            status.load_status = PLUGIN_LOAD_STATUS_FAILED.to_owned();
            let error = SyncError::new(format!(
                "home plugins: plugin {} installed but not loaded",
                status.id
            ));
            if status.error.trim().is_empty() {
                status.error = error.to_string();
            }
            errors.push(error);
        }
    }
    report.finished_at = Some(now);
    report.updated_at = now;
    if let Some(error) = join_errors(&errors) {
        report.ok = false;
        report.status = PLUGIN_TASK_STATUS_ERROR.to_owned();
        report.error = error.to_string();
        Err(error)
    } else {
        report.ok = true;
        report.status = PLUGIN_TASK_STATUS_OK.to_owned();
        report.error.clear();
        Ok(())
    }
}

fn enabled_config(config: Option<&SyncConfig>) -> Option<&SyncConfig> {
    config.filter(|config| config.home_enabled && config.plugins_enabled)
}

fn plugin_config_enabled(item: &PluginInstanceConfig) -> bool {
    item.enabled == Some(true)
}

fn store_manifest_from_plugin_config(
    id: &str,
    item: &PluginInstanceConfig,
) -> Result<Option<Manifest>, SyncError> {
    let Some(mut manifest) = item.store.clone() else {
        return Ok(None);
    };
    if manifest.id.trim().is_empty() {
        manifest.id = id.trim().to_owned();
    }
    manifest.validate().map_err(|error| {
        SyncError::new(format!(
            "home plugins: invalid store manifest for {id}: {error}"
        ))
    })?;
    Ok(Some(manifest))
}

fn plugin_status_from_manifest(manifest: &Manifest) -> PluginInstallStatus {
    PluginInstallStatus {
        id: manifest.id.trim().to_owned(),
        version: manifest.version.trim().to_owned(),
        release_tag: manifest.release_tag.trim().to_owned(),
        repository: manifest.repository.trim().to_owned(),
        install_type: manifest.install_type().to_owned(),
        install_status: PLUGIN_INSTALL_STATUS_FAILED.to_owned(),
        ..PluginInstallStatus::default()
    }
}

fn apply_install_result(status: &mut PluginInstallStatus, result: InstallResult) {
    status.path = result.path.to_string_lossy().trim().to_owned();
    status.skipped = result.skipped;
    status.overwritten = result.overwritten;
    status.install_status = if result.skipped {
        PLUGIN_INSTALL_STATUS_SKIPPED
    } else {
        PLUGIN_INSTALL_STATUS_INSTALLED
    }
    .to_owned();
}

fn add_installed_version_statuses(
    report: &mut SyncReport,
    config: &SyncConfig,
    root: &Path,
    installed_versions: &HashMap<String, String>,
) {
    for (id, item) in &config.plugins {
        if !plugin_config_enabled(item) {
            continue;
        }
        let Some(version) = installed_versions.get(id.trim()) else {
            continue;
        };
        let mut status = PluginInstallStatus {
            id: id.trim().to_owned(),
            version: version.trim().to_owned(),
            install_status: PLUGIN_INSTALL_STATUS_SKIPPED.to_owned(),
            skipped: true,
            ..PluginInstallStatus::default()
        };
        if let Ok(files) = plugin_file_infos(root, id) {
            if let Some(file) = files
                .iter()
                .find(|file| file.version.trim() == status.version)
            {
                status.path = file.path.to_string_lossy().trim().to_owned();
            }
        }
        if let Ok(Some(manifest)) = store_manifest_from_plugin_config(id, item) {
            if plugin_versions_equal(&status.version, &manifest.version) {
                status.release_tag = manifest.release_tag.trim().to_owned();
                status.repository = manifest.repository.trim().to_owned();
                status.install_type = manifest.install_type().to_owned();
            }
        }
        report.plugins.push(status);
    }
}

fn plugin_versions_equal(left: &str, right: &str) -> bool {
    !left.trim().is_empty()
        && !right.trim().is_empty()
        && !update_available(left, right)
        && !update_available(right, left)
}

fn upsert_plugin_install_status(report: &mut SyncReport, status: PluginInstallStatus) {
    if let Some(existing) = report
        .plugins
        .iter_mut()
        .find(|existing| existing.id.trim() == status.id.trim())
    {
        *existing = status;
    } else {
        report.plugins.push(status);
    }
}

struct DeleteArtifactOutcome {
    path: PathBuf,
    deleted: bool,
    error: Option<SyncError>,
}

fn delete_plugin_artifact(
    context: &dyn OperationContext,
    root: &Path,
    id: &str,
    runtime: Option<&dyn PluginRuntime>,
) -> DeleteArtifactOutcome {
    if let Err(error) = check_context(context) {
        return DeleteArtifactOutcome {
            path: PathBuf::new(),
            deleted: false,
            error: Some(error),
        };
    }
    let id = id.trim();
    if !valid_plugin_file_id(id) {
        return DeleteArtifactOutcome {
            path: PathBuf::new(),
            deleted: false,
            error: Some(SyncError::new(format!("invalid plugin id {id:?}"))),
        };
    }
    let paths = match plugin_file_paths(root, id) {
        Ok(paths) => paths,
        Err(error) => {
            return DeleteArtifactOutcome {
                path: PathBuf::new(),
                deleted: false,
                error: Some(error),
            };
        }
    };
    if let Err(error) = check_context(context) {
        return DeleteArtifactOutcome {
            path: PathBuf::new(),
            deleted: false,
            error: Some(error),
        };
    }
    let Some(first) = paths.first().cloned() else {
        return DeleteArtifactOutcome {
            path: PathBuf::new(),
            deleted: false,
            error: None,
        };
    };
    if let Some(runtime) = runtime.filter(|runtime| runtime.plugin_busy(id)) {
        if let Err(error) = check_context(context) {
            return DeleteArtifactOutcome {
                path: first,
                deleted: false,
                error: Some(error),
            };
        }
        if !runtime.unload_plugin(context, id) && runtime.plugin_busy(id) {
            return DeleteArtifactOutcome {
                path: first,
                deleted: false,
                error: Some(SyncError::new(
                    "loaded plugin library cannot be overwritten while the server is running",
                )),
            };
        }
    }
    let mut deleted = false;
    for path in paths {
        if let Err(error) = check_context(context) {
            return DeleteArtifactOutcome {
                path: first,
                deleted,
                error: Some(error),
            };
        }
        match fs::remove_file(&path) {
            Ok(()) => deleted = true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return DeleteArtifactOutcome {
                    path: first,
                    deleted,
                    error: Some(error.into()),
                };
            }
        }
        if let Err(error) = check_context(context) {
            return DeleteArtifactOutcome {
                path: first,
                deleted,
                error: Some(error),
            };
        }
    }
    DeleteArtifactOutcome {
        path: first,
        deleted,
        error: None,
    }
}

fn check_context(context: &dyn OperationContext) -> Result<(), SyncError> {
    if context.is_cancelled() {
        Err(SyncError::new("context canceled"))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct PluginFileInfo {
    id: String,
    path: PathBuf,
    version: String,
}

fn plugin_file_paths(root: &Path, id: &str) -> Result<Vec<PathBuf>, SyncError> {
    Ok(plugin_file_infos(root, id)?
        .into_iter()
        .map(|file| file.path)
        .collect())
}

fn plugin_file_infos(root: &Path, id: &str) -> Result<Vec<PluginFileInfo>, SyncError> {
    let root = if root.as_os_str().is_empty() {
        Path::new("plugins")
    } else {
        root
    };
    let id = id.trim();
    let platform = current_platform();
    let extension = plugin_extension(&platform.goos);
    let mut candidates = Vec::new();
    for directory in plugin_candidate_dirs(root, &platform.goos, &platform.goarch) {
        let entries = match fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        let mut paths = entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                entry
                    .file_type()
                    .ok()
                    .filter(|kind| kind.is_file())
                    .map(|_| entry.path())
            })
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.to_ascii_lowercase().ends_with(extension))
            })
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            if let Some(file) = plugin_file_from_path(&path, Some(extension)) {
                if file.id == id {
                    candidates.push(file);
                }
            }
        }
    }
    if candidates.len() > 1 {
        let mut best = 0;
        for index in 1..candidates.len() {
            if plugin_file_preferred(&candidates[index], &candidates[best]) {
                best = index;
            }
        }
        if best != 0 {
            let preferred = candidates.remove(best);
            candidates.insert(0, preferred);
        }
    }
    Ok(candidates)
}

fn plugin_candidate_dirs(root: &Path, goos: &str, goarch: &str) -> [PathBuf; 2] {
    [root.join(goos).join(goarch), root.to_path_buf()]
}

fn plugin_file_from_path(path: &Path, required_extension: Option<&str>) -> Option<PluginFileInfo> {
    let base = path.file_name()?.to_str()?;
    let lower = base.to_ascii_lowercase();
    let extension = if let Some(required) = required_extension {
        if !lower.ends_with(&required.to_ascii_lowercase()) {
            return None;
        }
        required
    } else {
        [".so", ".dylib", ".dll"]
            .into_iter()
            .find(|extension| lower.ends_with(extension))?
    };
    let name = &base[..base.len() - extension.len()];
    let (id, version) = name
        .rfind("-v")
        .filter(|index| *index > 0)
        .map(|index| (&name[..index], &name[index + 2..]))
        .filter(|(id, version)| valid_plugin_file_id(id) && valid_plugin_file_version(version))
        .unwrap_or((name, ""));
    valid_plugin_file_id(id).then(|| PluginFileInfo {
        id: id.to_owned(),
        path: path.to_path_buf(),
        version: version.to_owned(),
    })
}

fn plugin_file_preferred(candidate: &PluginFileInfo, current: &PluginFileInfo) -> bool {
    if current.path.as_os_str().is_empty() {
        true
    } else if candidate.version.is_empty() {
        false
    } else if current.version.is_empty() {
        true
    } else {
        update_available(&current.version, &candidate.version)
    }
}

pub fn plugin_extension(goos: &str) -> &'static str {
    match goos.trim().to_ascii_lowercase().as_str() {
        "darwin" | "mac" | "macos" | "osx" => ".dylib",
        "windows" => ".dll",
        _ => ".so",
    }
}

fn valid_plugin_file_id(id: &str) -> bool {
    let id = id.trim();
    !id.is_empty()
        && id != "."
        && id != ".."
        && !id.contains(['/', '\\'])
        && id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "-_.".contains(character))
}

fn valid_plugin_file_version(version: &str) -> bool {
    let version = version.trim();
    !version.is_empty() && !version.starts_with('v') && version.as_bytes()[0].is_ascii_digit()
}

fn prefix_error(prefix: &str, error: SyncError) -> SyncError {
    SyncError::new(format!("{prefix}: {error}"))
}

fn join_errors(errors: &[SyncError]) -> Option<SyncError> {
    (!errors.is_empty()).then(|| {
        SyncError::new(
            errors
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        )
    })
}
