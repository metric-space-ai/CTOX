// ref: internal/watcher/watcher_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::config_reload::{JsonConfigDecoder, WatcherConfig};
use super::dispatcher::{AuthUpdateDispatcher, AuthUpdateSink};
use super::events::{match_provider, ChannelEventSource, FsEvent, FsEventKind};
use super::{
    ConfigReloadSink, NativeWatchFilesystem, NoopPersistenceSink, SystemWatchClock, Watcher,
    WatcherDependencies, WatcherError,
};
use std::io;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime};

#[derive(Default)]
struct CapturingUpdateSink(Mutex<Vec<super::AuthUpdate>>);
impl AuthUpdateSink for CapturingUpdateSink {
    fn send_batch(&self, updates: &[super::AuthUpdate]) -> io::Result<()> {
        self.0.lock().unwrap().extend_from_slice(updates);
        Ok(())
    }
}
#[derive(Default)]
struct CapturingReloadSink(Mutex<Vec<WatcherConfig>>);
impl ConfigReloadSink for CapturingReloadSink {
    fn on_reload(&self, config: &WatcherConfig) {
        self.0.lock().unwrap().push(config.clone());
    }
}

fn dependencies(
    events: Arc<ChannelEventSource>,
    updates: Arc<CapturingUpdateSink>,
    reloads: Arc<CapturingReloadSink>,
) -> WatcherDependencies {
    WatcherDependencies {
        filesystem: Arc::new(NativeWatchFilesystem),
        clock: Arc::new(SystemWatchClock),
        events,
        config_decoder: Arc::new(JsonConfigDecoder),
        reload_sink: reloads,
        persistence_sink: Arc::new(NoopPersistenceSink),
        dispatcher: Arc::new(AuthUpdateDispatcher::start(8, updates)),
        plugin_parser: None,
        coalesce_window: Duration::ZERO,
        remove_debounce: Duration::from_millis(50),
    }
}

#[test]
fn start_validates_paths_and_stop_is_owned_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing.json");
    let events = ChannelEventSource::new();
    let watcher = Watcher::new(
        missing.clone(),
        dir.path().to_path_buf(),
        dependencies(
            events,
            Arc::new(CapturingUpdateSink::default()),
            Arc::new(CapturingReloadSink::default()),
        ),
    );
    assert!(matches!(watcher.start(), Err(WatcherError::MissingConfig(path)) if path == missing));
    watcher.stop();
    watcher.stop();
}

#[test]
fn lifecycle_discovers_credentials_coalesces_events_and_dispatches_changes() {
    let dir = tempfile::tempdir().unwrap();
    let auth_dir = dir.path().join("auth");
    std::fs::create_dir(&auth_dir).unwrap();
    let config_path = dir.path().join("config.json");
    std::fs::write(&config_path, b"{}").unwrap();
    let auth_path = auth_dir.join("claude.json");
    std::fs::write(&auth_path, br#"{"type":"claude","access_token":"one"}"#).unwrap();
    let events = ChannelEventSource::new();
    let updates = Arc::new(CapturingUpdateSink::default());
    let reloads = Arc::new(CapturingReloadSink::default());
    let watcher = Watcher::new(
        config_path.clone(),
        auth_dir.clone(),
        dependencies(events.clone(), updates.clone(), reloads.clone()),
    );
    watcher.start().unwrap();
    for _ in 0..100 {
        if !updates.0.lock().unwrap().is_empty() {
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(watcher.snapshot_auths().len(), 1);
    assert_eq!(reloads.0.lock().unwrap().len(), 1);
    std::fs::write(&auth_path, br#"{"type":"claude","access_token":"two"}"#).unwrap();
    let now = SystemTime::now();
    events.emit(FsEvent {
        path: auth_path.clone(),
        kind: FsEventKind::Write,
        observed_at: now,
    });
    events.emit(FsEvent {
        path: auth_path,
        kind: FsEventKind::Write,
        observed_at: now,
    });
    for _ in 0..100 {
        if updates.0.lock().unwrap().len() >= 2 {
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(
        updates.0.lock().unwrap().len(),
        2,
        "initial add plus one coalesced modify"
    );
    watcher.stop();
    watcher.stop();
}

#[test]
fn config_reload_is_hash_guarded_and_provider_matching_is_case_insensitive() {
    let dir = tempfile::tempdir().unwrap();
    let auth_dir = dir.path().join("auth");
    std::fs::create_dir(&auth_dir).unwrap();
    let config_path = dir.path().join("config.json");
    std::fs::write(&config_path, b"{}").unwrap();
    let events = ChannelEventSource::new();
    let reloads = Arc::new(CapturingReloadSink::default());
    let watcher = Watcher::new(
        config_path.clone(),
        auth_dir,
        dependencies(
            events.clone(),
            Arc::new(CapturingUpdateSink::default()),
            reloads.clone(),
        ),
    );
    watcher.start().unwrap();
    events.emit(FsEvent {
        path: config_path.clone(),
        kind: FsEventKind::Write,
        observed_at: SystemTime::now(),
    });
    thread::sleep(Duration::from_millis(20));
    assert_eq!(reloads.0.lock().unwrap().len(), 1);
    std::fs::write(&config_path, br#"{"settings":{"retry":2}}"#).unwrap();
    events.emit(FsEvent {
        path: config_path,
        kind: FsEventKind::Write,
        observed_at: SystemTime::now(),
    });
    for _ in 0..100 {
        if reloads.0.lock().unwrap().len() == 2 {
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(reloads.0.lock().unwrap().len(), 2);
    assert_eq!(
        match_provider("Claude", &["codex".into(), "claude".into()]),
        Some("claude".into())
    );
    watcher.stop();
}
