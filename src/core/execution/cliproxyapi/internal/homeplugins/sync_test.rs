// ref: internal/homeplugins/sync_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, TimeZone, Utc};
use tempfile::TempDir;

use super::*;

struct FixedBoundary {
    now: DateTime<Utc>,
    reject_tilde: bool,
}

impl FixedBoundary {
    fn new() -> Self {
        Self {
            now: Utc.with_ymd_and_hms(2026, 8, 4, 10, 0, 0).unwrap(),
            reject_tilde: true,
        }
    }
}

impl HomePluginBoundary for FixedBoundary {
    fn now(&self) -> DateTime<Utc> {
        self.now
    }

    fn resolve_plugins_dir(&self, configured: &str) -> Result<PathBuf, SyncError> {
        if self.reject_tilde && configured.trim().starts_with('~') {
            Err(SyncError::new(
                "resolve plugins directory: home directory is unavailable",
            ))
        } else {
            Ok(PathBuf::from(configured))
        }
    }
}

#[derive(Default)]
struct FakeStore {
    fail: Mutex<Option<String>>,
    skipped: AtomicBool,
    loaded_seen: AtomicBool,
    auth_seen: Mutex<Vec<u8>>,
}

#[derive(Default)]
struct FakeSdkIo {
    auth_seen: Mutex<Vec<u8>>,
    loaded_seen: AtomicBool,
    calls: Mutex<usize>,
}

impl crate::sdk::pluginstore::PluginStoreIo for FakeSdkIo {
    fn fetch_registry(
        &self,
        _client: &crate::sdk::pluginstore::Client,
    ) -> crate::sdk::pluginstore::Result<crate::sdk::pluginstore::Registry> {
        unreachable!("sync install must not fetch the registry")
    }

    fn fetch_latest_release(
        &self,
        _client: &crate::sdk::pluginstore::Client,
        _plugin: &crate::sdk::pluginstore::Plugin,
    ) -> crate::sdk::pluginstore::Result<crate::sdk::pluginstore::Release> {
        unreachable!("sync install must not resolve latest")
    }

    fn fetch_release_by_tag(
        &self,
        _client: &crate::sdk::pluginstore::Client,
        _plugin: &crate::sdk::pluginstore::Plugin,
        _tag: &str,
    ) -> crate::sdk::pluginstore::Result<crate::sdk::pluginstore::Release> {
        unreachable!("sync install must not resolve a tag in this fixture")
    }

    fn install(
        &self,
        _client: &crate::sdk::pluginstore::Client,
        _plugin: &crate::sdk::pluginstore::Plugin,
        _options: &crate::sdk::pluginstore::InstallOptions,
    ) -> crate::sdk::pluginstore::Result<InstallResult> {
        unreachable!("sync uses pinned manifest installation")
    }

    fn install_version(
        &self,
        _client: &crate::sdk::pluginstore::Client,
        _plugin: &crate::sdk::pluginstore::Plugin,
        _release_tag: &str,
        _version: &str,
        _options: &crate::sdk::pluginstore::InstallOptions,
    ) -> crate::sdk::pluginstore::Result<InstallResult> {
        unreachable!("sync uses pinned manifest installation")
    }

    fn install_manifest(
        &self,
        client: &crate::sdk::pluginstore::Client,
        manifest: &Manifest,
        options: &crate::sdk::pluginstore::InstallOptions,
    ) -> crate::sdk::pluginstore::Result<InstallResult> {
        *self.calls.lock().unwrap() += 1;
        *self.auth_seen.lock().unwrap() = client.resolved_auth()[0].token.expose().to_vec();
        self.loaded_seen.store(
            options
                .plugin_loaded
                .as_ref()
                .is_some_and(|loaded| loaded()),
            Ordering::SeqCst,
        );
        let path = plugin_path(
            &options.plugins_dir,
            &options.goos,
            &options.goarch,
            &manifest.id,
            &manifest.version,
        );
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"sdk-store-io").unwrap();
        Ok(InstallResult {
            id: manifest.id.clone(),
            version: manifest.version.clone(),
            path,
            ..InstallResult::default()
        })
    }
}

impl PluginStore for FakeStore {
    fn install_manifest(
        &self,
        context: &dyn OperationContext,
        request: InstallRequest<'_>,
    ) -> Result<InstallResult, SyncError> {
        if context.is_cancelled() {
            return Err(SyncError::new("context canceled"));
        }
        if let Some(message) = self.fail.lock().unwrap().clone() {
            return Err(SyncError::new(message));
        }
        self.loaded_seen
            .store(request.plugin_loaded, Ordering::SeqCst);
        if let Some(auth) = request.resolved_auth.first() {
            *self.auth_seen.lock().unwrap() = auth.token.expose().to_vec();
        }
        let path = plugin_path(
            request.plugins_dir,
            &request.platform.goos,
            &request.platform.goarch,
            &request.manifest.id,
            &request.manifest.version,
        );
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let skipped = self.skipped.load(Ordering::SeqCst);
        if !skipped {
            fs::write(&path, b"library-data").unwrap();
        }
        let overwritten = path.exists() && !skipped;
        Ok(InstallResult {
            path,
            skipped,
            overwritten,
            ..InstallResult::default()
        })
    }
}

#[derive(Default)]
struct FakeRuntime {
    busy: AtomicBool,
    unloads: Mutex<Vec<String>>,
    context_cancelled_at_unload: AtomicBool,
}

impl PluginRuntime for FakeRuntime {
    fn plugin_busy(&self, _id: &str) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    fn unload_plugin(&self, context: &dyn OperationContext, id: &str) -> bool {
        self.context_cancelled_at_unload
            .store(context.is_cancelled(), Ordering::SeqCst);
        self.unloads.lock().unwrap().push(id.to_owned());
        self.busy.store(false, Ordering::SeqCst);
        true
    }
}

struct CancelContext(AtomicBool);

impl CancelContext {
    fn cancelled() -> Self {
        Self(AtomicBool::new(true))
    }
}

impl OperationContext for CancelContext {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

struct Inspector(HashMap<String, bool>);

impl PluginLoadInspector for Inspector {
    fn plugin_registered(&self, id: &str) -> bool {
        self.0.get(id).copied().unwrap_or(false)
    }
}

fn manifest(id: &str, version: &str) -> Manifest {
    Manifest {
        id: id.to_owned(),
        name: "Sample".to_owned(),
        description: "Adds sample support.".to_owned(),
        author: "owner".to_owned(),
        version: version.to_owned(),
        release_tag: format!("v{version}"),
        repository: "https://github.com/owner/sample-plugin".to_owned(),
        ..Manifest::default()
    }
}

fn config(root: &Path) -> SyncConfig {
    SyncConfig {
        home_enabled: true,
        plugins_enabled: true,
        plugins_dir: root.to_string_lossy().into_owned(),
        plugins: BTreeMap::from([(
            "sample".to_owned(),
            PluginInstanceConfig {
                enabled: Some(true),
                store: Some(manifest("sample", "0.2.0")),
            },
        )]),
    }
}

fn plugin_path(root: &Path, goos: &str, goarch: &str, id: &str, version: &str) -> PathBuf {
    let name = if version.trim().is_empty() {
        id.trim().to_owned()
    } else {
        format!("{}-v{}", id.trim(), version.trim())
    };
    root.join(goos)
        .join(goarch)
        .join(format!("{name}{}", plugin_extension(goos)))
}

#[test]
fn sync_platform_installs_manifest_artifact_and_reports_success() {
    let root = TempDir::new().unwrap();
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let sync = HomePluginSync::new(&store, &boundary);
    let platform = Platform {
        goos: "windows".to_owned(),
        goarch: "amd64".to_owned(),
    };

    let outcome =
        sync.sync_platform_with_report(&ActiveContext, Some(&config(root.path())), None, platform);

    assert_eq!(outcome.error, None);
    assert!(outcome.report.ok);
    assert_eq!(outcome.report.status, "success");
    assert_eq!(outcome.report.phase, "install");
    assert_eq!(outcome.report.plugins.len(), 1);
    let status = &outcome.report.plugins[0];
    assert_eq!(status.id, "sample");
    assert_eq!(status.version, "0.2.0");
    assert_eq!(status.install_status, "installed");
    assert_eq!(fs::read(&status.path).unwrap(), b"library-data");
}

#[test]
fn sync_platform_records_skipped_identical_artifact_and_busy_state() {
    let root = TempDir::new().unwrap();
    let store = FakeStore::default();
    store.skipped.store(true, Ordering::SeqCst);
    let runtime = FakeRuntime::default();
    runtime.busy.store(true, Ordering::SeqCst);
    let boundary = FixedBoundary::new();
    let sync = HomePluginSync::new(&store, &boundary);

    let outcome = sync.sync_platform_with_report(
        &ActiveContext,
        Some(&config(root.path())),
        Some(&runtime),
        Platform {
            goos: "windows".into(),
            goarch: "amd64".into(),
        },
    );

    assert!(outcome.report.ok);
    assert_eq!(outcome.report.plugins[0].install_status, "skipped");
    assert!(outcome.report.plugins[0].skipped);
    assert!(store.loaded_seen.load(Ordering::SeqCst));
    assert!(runtime.unloads.lock().unwrap().is_empty());
}

#[test]
fn sync_platform_skips_disabled_and_manifestless_configs_in_sorted_order() {
    let root = TempDir::new().unwrap();
    let mut config = config(root.path());
    config.plugins.insert(
        "aaa-disabled".into(),
        PluginInstanceConfig {
            enabled: Some(false),
            store: Some(manifest("aaa-disabled", "1.0.0")),
        },
    );
    config.plugins.insert(
        "bbb-manifestless".into(),
        PluginInstanceConfig {
            enabled: Some(true),
            store: None,
        },
    );
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let outcome = HomePluginSync::new(&store, &boundary).sync_platform_with_report(
        &ActiveContext,
        Some(&config),
        None,
        Platform {
            goos: "linux".into(),
            goarch: "amd64".into(),
        },
    );
    assert!(outcome.report.ok);
    assert_eq!(outcome.report.plugins.len(), 1);
    assert_eq!(outcome.report.plugins[0].id, "sample");
}

#[test]
fn sync_platform_preserves_failed_report_for_invalid_manifest() {
    let root = TempDir::new().unwrap();
    let mut config = config(root.path());
    config.plugins.get_mut("sample").unwrap().store = Some(Manifest {
        id: "sample".into(),
        ..Manifest::default()
    });
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();

    let outcome = HomePluginSync::new(&store, &boundary).sync_platform_with_report(
        &ActiveContext,
        Some(&config),
        None,
        Platform {
            goos: "linux".into(),
            goarch: "amd64".into(),
        },
    );

    assert!(outcome.error.is_some());
    assert!(!outcome.report.ok);
    assert_eq!(outcome.report.status, "failed");
    assert_eq!(outcome.report.plugins[0].install_status, "failed");
    assert!(outcome.report.plugins[0]
        .error
        .contains("invalid store manifest"));
}

#[test]
fn sync_platform_rejects_empty_normalized_platform_parts() {
    let root = TempDir::new().unwrap();
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let sync = HomePluginSync::new(&store, &boundary);
    let goos = sync.sync_platform_with_report(
        &ActiveContext,
        Some(&config(root.path())),
        None,
        Platform {
            goos: " ".into(),
            goarch: "amd64".into(),
        },
    );
    assert_eq!(
        goos.error.unwrap().to_string(),
        "home plugins: goos is required"
    );
    let goarch = sync.sync_platform_with_report(
        &ActiveContext,
        Some(&config(root.path())),
        None,
        Platform {
            goos: "linux".into(),
            goarch: " ".into(),
        },
    );
    assert_eq!(
        goarch.error.unwrap().to_string(),
        "home plugins: goarch is required"
    );
}

#[test]
fn resolved_sync_uses_auth_clears_it_and_upserts_installed_status() {
    let root = TempDir::new().unwrap();
    let installed = plugin_path(
        root.path(),
        &current_platform().goos,
        &current_platform().goarch,
        "sample",
        "0.9.0",
    );
    fs::create_dir_all(installed.parent().unwrap()).unwrap();
    fs::write(&installed, b"old").unwrap();
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let mut items = vec![PluginSyncItem {
        manifest: manifest("sample", "1.0.0"),
        auth: vec![ResolvedAuthConfig {
            token: crate::sdk::pluginstore::Secret::new(b"temporary-token".to_vec()),
            ..ResolvedAuthConfig::default()
        }],
    }];
    let outcome = HomePluginSync::new(&store, &boundary).sync_resolved_with_report(
        &ActiveContext,
        ResolvedSyncRequest {
            config: Some(&config(root.path())),
            items: &mut items,
            expires_at: boundary.now + chrono::Duration::minutes(1),
            installed_versions: &HashMap::from([("sample".into(), "0.9.0".into())]),
            runtime: None,
        },
    );
    assert_eq!(outcome.error, None);
    assert_eq!(&*store.auth_seen.lock().unwrap(), b"temporary-token");
    assert!(items[0].auth.is_empty());
    assert_eq!(items[0].manifest, Manifest::default());
    assert_eq!(outcome.report.plugins.len(), 1);
    assert_eq!(outcome.report.plugins[0].version, "1.0.0");
    assert_eq!(outcome.report.plugins[0].install_status, "installed");
}

#[test]
fn resolved_sync_reaches_sdk_store_io_with_owned_types_and_scoped_auth() {
    let root = TempDir::new().unwrap();
    let io = Arc::new(FakeSdkIo::default());
    let adapter = SdkPluginStoreAdapter::new(
        Arc::clone(&io) as Arc<dyn crate::sdk::pluginstore::PluginStoreIo>,
        "https://plugins.example/registry.json",
    );
    let runtime = FakeRuntime::default();
    runtime.busy.store(true, Ordering::SeqCst);
    let boundary = FixedBoundary::new();
    let mut items = vec![PluginSyncItem {
        manifest: manifest("sample", "1.4.0"),
        auth: vec![ResolvedAuthConfig {
            match_url: "https://api.github.com/repos/owner/sample-plugin".into(),
            apply_to: vec![crate::sdk::pluginstore::REQUEST_KIND_ARTIFACT.into()],
            auth_type: crate::sdk::pluginstore::AUTH_TYPE_BEARER.into(),
            token: crate::sdk::pluginstore::Secret::new(b"scoped-token".to_vec()),
            ..ResolvedAuthConfig::default()
        }],
    }];

    let outcome = HomePluginSync::new(&adapter, &boundary).sync_resolved_with_report(
        &ActiveContext,
        ResolvedSyncRequest {
            config: Some(&config(root.path())),
            items: &mut items,
            expires_at: boundary.now + chrono::Duration::minutes(1),
            installed_versions: &HashMap::new(),
            runtime: Some(&runtime),
        },
    );

    assert_eq!(outcome.error, None);
    assert_eq!(*io.calls.lock().unwrap(), 1);
    assert_eq!(&*io.auth_seen.lock().unwrap(), b"scoped-token");
    assert!(io.loaded_seen.load(Ordering::SeqCst));
    assert!(items[0].auth.is_empty());
    assert_eq!(items[0].manifest, Manifest::default());
    assert_eq!(
        fs::read(&outcome.report.plugins[0].path).unwrap(),
        b"sdk-store-io"
    );
}

#[test]
fn resolved_sync_stops_on_expiry_and_clears_all_items() {
    let root = TempDir::new().unwrap();
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let mut items = vec![
        PluginSyncItem {
            manifest: manifest("sample", "1.0.0"),
            auth: vec![ResolvedAuthConfig {
                token: crate::sdk::pluginstore::Secret::new(b"secret-a".to_vec()),
                ..ResolvedAuthConfig::default()
            }],
        },
        PluginSyncItem {
            manifest: manifest("other", "1.0.0"),
            auth: vec![ResolvedAuthConfig {
                token: crate::sdk::pluginstore::Secret::new(b"secret-b".to_vec()),
                ..ResolvedAuthConfig::default()
            }],
        },
    ];
    let outcome = HomePluginSync::new(&store, &boundary).sync_resolved_with_report(
        &ActiveContext,
        ResolvedSyncRequest {
            config: Some(&config(root.path())),
            items: &mut items,
            expires_at: boundary.now,
            installed_versions: &HashMap::new(),
            runtime: None,
        },
    );
    assert!(outcome.error.unwrap().to_string().contains("expired"));
    assert!(items.iter().all(|item| item.auth.is_empty()));
    assert!(!outcome.report.ok);
}

#[test]
fn resolved_sync_reports_unchanged_installed_metadata_only_for_equal_version() {
    let root = TempDir::new().unwrap();
    let platform = current_platform();
    let installed = plugin_path(
        root.path(),
        &platform.goos,
        &platform.goarch,
        "sample",
        "1.0.0",
    );
    fs::create_dir_all(installed.parent().unwrap()).unwrap();
    fs::write(&installed, b"plugin").unwrap();
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let sync = HomePluginSync::new(&store, &boundary);
    let mut no_items = [];
    let mut equal_config = config(root.path());
    equal_config.plugins.get_mut("sample").unwrap().store = Some(manifest("sample", "1.0.0"));
    let equal = sync.sync_resolved_with_report(
        &ActiveContext,
        ResolvedSyncRequest {
            config: Some(&equal_config),
            items: &mut no_items,
            expires_at: boundary.now + chrono::Duration::minutes(1),
            installed_versions: &HashMap::from([("sample".into(), "1.0.0".into())]),
            runtime: None,
        },
    );
    let equal_status = &equal.report.plugins[0];
    assert_eq!(equal_status.path, installed.to_string_lossy());
    assert_eq!(equal_status.release_tag, "v1.0.0");
    assert_eq!(equal_status.install_type, "github-release");

    let mut changed = config(root.path());
    changed.plugins.get_mut("sample").unwrap().store = Some(manifest("sample", "2.0.0"));
    let different = sync.sync_resolved_with_report(
        &ActiveContext,
        ResolvedSyncRequest {
            config: Some(&changed),
            items: &mut no_items,
            expires_at: boundary.now + chrono::Duration::minutes(1),
            installed_versions: &HashMap::from([("sample".into(), "1.0.0".into())]),
            runtime: None,
        },
    );
    let different_status = &different.report.plugins[0];
    assert_eq!(different_status.version, "1.0.0");
    assert!(different_status.release_tag.is_empty());
    assert!(different_status.repository.is_empty());
    assert!(different_status.install_type.is_empty());
}

#[test]
fn installed_versions_prefers_newest_current_platform_artifact() {
    let root = TempDir::new().unwrap();
    let platform = current_platform();
    for version in ["1.0.0", "2.3.4"] {
        let path = plugin_path(
            root.path(),
            &platform.goos,
            &platform.goarch,
            "sample",
            version,
        );
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"plugin").unwrap();
    }
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let versions = HomePluginSync::new(&store, &boundary)
        .installed_versions(Some(&config(root.path())))
        .unwrap();
    assert_eq!(versions.get("sample").map(String::as_str), Some("2.3.4"));
}

#[test]
fn mark_load_results_fails_unloaded_plugin() {
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let mut report =
        HomePluginSync::new(&store, &boundary).completed_sync_report(current_platform(), None);
    report.finished_at = None;
    report.plugins.push(PluginInstallStatus {
        id: "sample".into(),
        install_status: "installed".into(),
        ..PluginInstallStatus::default()
    });
    let error = mark_load_results(
        Some(&mut report),
        Some(&Inspector(HashMap::new())),
        boundary.now,
    )
    .unwrap_err();
    assert!(error.to_string().contains("installed but not loaded"));
    assert_eq!(report.phase, "load");
    assert_eq!(report.plugins[0].load_status, "failed");
    assert!(!report.ok);
}

#[test]
fn mark_load_results_preserves_install_and_global_sync_failures() {
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let sync = HomePluginSync::new(&store, &boundary);
    let mut install =
        sync.completed_sync_report(current_platform(), Some(SyncError::new("install boom")));
    install.plugins.push(PluginInstallStatus {
        id: "sample".into(),
        install_status: "failed".into(),
        error: "install boom".into(),
        ..PluginInstallStatus::default()
    });
    assert!(mark_load_results(
        Some(&mut install),
        Some(&Inspector(HashMap::from([("sample".into(), true)]))),
        boundary.now,
    )
    .is_err());
    assert_eq!(install.plugins[0].load_status, "skipped");
    assert_eq!(install.error, "install boom");

    let mut global = sync.completed_sync_report(
        current_platform(),
        Some(SyncError::new("home plugins: plugin sync response expired")),
    );
    global.plugins.push(PluginInstallStatus {
        id: "sample".into(),
        install_status: "installed".into(),
        ..PluginInstallStatus::default()
    });
    assert!(mark_load_results(
        Some(&mut global),
        Some(&Inspector(HashMap::from([("sample".into(), true)]))),
        boundary.now,
    )
    .is_err());
    assert_eq!(global.plugins[0].load_status, "loaded");
    assert!(global.error.contains("expired"));
}

#[test]
fn completed_sync_report_has_terminal_success_and_failure_shapes() {
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let sync = HomePluginSync::new(&store, &boundary);
    let success = sync.completed_sync_report(current_platform(), None);
    assert!(success.ok);
    assert_eq!(success.task, "plugin-sync");
    assert_eq!(success.finished_at, Some(boundary.now));
    let failure =
        sync.completed_sync_report(current_platform(), Some(SyncError::new("access denied")));
    assert!(!failure.ok);
    assert_eq!(failure.status, "failed");
    assert_eq!(failure.error, "access denied");
}

#[test]
fn delete_rejects_unresolved_plugins_directory_without_touching_literal_tilde() {
    let workspace = TempDir::new().unwrap();
    let literal = workspace
        .path()
        .join("~/.cli-proxy-api/plugins")
        .join(current_platform().goos)
        .join(current_platform().goarch)
        .join(format!(
            "sample{}",
            plugin_extension(&current_platform().goos)
        ));
    fs::create_dir_all(literal.parent().unwrap()).unwrap();
    fs::write(&literal, b"plugin").unwrap();
    let cfg = SyncConfig {
        plugins_dir: "~/.cli-proxy-api/plugins".into(),
        ..SyncConfig::default()
    };
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let report = HomePluginSync::new(&store, &boundary).delete_with_report(
        &ActiveContext,
        Some(&cfg),
        None,
        41,
        "sample",
    );
    assert!(!report.ok);
    assert!(report.plugins[0]
        .error
        .contains("resolve plugins directory"));
    assert!(literal.exists());
}

#[test]
fn delete_removes_all_versions_unloads_once_and_retains_other_plugins() {
    let root = TempDir::new().unwrap();
    let platform = current_platform();
    let older = plugin_path(
        root.path(),
        &platform.goos,
        &platform.goarch,
        "sample",
        "0.2.0",
    );
    let newer = plugin_path(
        root.path(),
        &platform.goos,
        &platform.goarch,
        "sample",
        "0.3.0",
    );
    let other = plugin_path(
        root.path(),
        &platform.goos,
        &platform.goarch,
        "other",
        "0.3.0",
    );
    fs::create_dir_all(older.parent().unwrap()).unwrap();
    for path in [&older, &newer, &other] {
        fs::write(path, b"plugin").unwrap();
    }
    let runtime = FakeRuntime::default();
    runtime.busy.store(true, Ordering::SeqCst);
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let report = HomePluginSync::new(&store, &boundary).delete_with_report(
        &ActiveContext,
        Some(&config(root.path())),
        Some(&runtime),
        43,
        "sample",
    );
    assert!(report.ok);
    assert_eq!(report.task_id, 43);
    assert_eq!(report.task, "plugin-delete");
    assert_eq!(report.phase, "delete");
    assert_eq!(report.plugins[0].install_status, "deleted");
    assert_eq!(report.plugins[0].path, newer.to_string_lossy());
    assert_eq!(&*runtime.unloads.lock().unwrap(), &["sample"]);
    assert!(!older.exists() && !newer.exists());
    assert!(other.exists());
}

#[test]
fn delete_stops_before_unload_when_context_is_cancelled() {
    let root = TempDir::new().unwrap();
    let platform = current_platform();
    let path = plugin_path(
        root.path(),
        &platform.goos,
        &platform.goarch,
        "sample",
        "1.0.0",
    );
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, b"plugin").unwrap();
    let runtime = FakeRuntime::default();
    runtime.busy.store(true, Ordering::SeqCst);
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let report = HomePluginSync::new(&store, &boundary).delete_with_report(
        &CancelContext::cancelled(),
        Some(&config(root.path())),
        Some(&runtime),
        44,
        "sample",
    );
    assert!(!report.ok);
    assert!(report.error.contains("context canceled"));
    assert!(runtime.unloads.lock().unwrap().is_empty());
    assert!(path.exists());
}

#[test]
fn delete_missing_plugin_is_success_and_nil_config_is_failure() {
    let root = TempDir::new().unwrap();
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let sync = HomePluginSync::new(&store, &boundary);
    let missing = sync.delete_with_report(
        &ActiveContext,
        Some(&config(root.path())),
        None,
        7,
        "missing",
    );
    assert!(missing.ok);
    assert_eq!(missing.plugins[0].install_status, "missing");
    let nil = sync.delete_with_report(&ActiveContext, None, None, 8, "sample");
    assert!(!nil.ok);
    assert_eq!(nil.plugins[0].error, "home plugins: config is nil");
}

#[test]
fn platform_normalization_and_extension_match_go() {
    assert_eq!(
        normalize_platform(Platform {
            goos: " MacOS ".into(),
            goarch: " x86_64 ".into(),
        }),
        Platform {
            goos: "darwin".into(),
            goarch: "amd64".into(),
        }
    );
    assert_eq!(plugin_extension("darwin"), ".dylib");
    assert_eq!(plugin_extension("windows"), ".dll");
    assert_eq!(plugin_extension("linux"), ".so");
}

#[test]
fn reports_serialize_with_upstream_wire_names_and_omit_empty_fields() {
    let store = FakeStore::default();
    let boundary = FixedBoundary::new();
    let report =
        HomePluginSync::new(&store, &boundary).completed_sync_report(current_platform(), None);
    let value = serde_json::to_value(report).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["task"], "plugin-sync");
    assert!(value.get("task_id").is_none());
    assert!(value.get("error").is_none());
    assert!(value.get("finished_at").is_some());
    assert!(value["platform"].get("goos").is_some());
}
