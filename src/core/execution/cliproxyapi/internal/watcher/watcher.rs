// ref: internal/watcher/watcher.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::clients::reload_clients;
use super::config_reload::{reload_config_if_changed, ConfigDecoder, WatcherConfig};
use super::dispatcher::AuthUpdateDispatcher;
use super::events::{self, FsEvent, WatchEventSource};
use super::synthesizer::context::PluginAuthParser;
use super::synthesizer::context::SynthesizedAuth;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthUpdateAction {
    Add,
    Modify,
    Delete,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthUpdate {
    pub action: AuthUpdateAction,
    pub auth: SynthesizedAuth,
}

pub trait WatchFilesystem: Send + Sync {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>>;
    fn list_files(&self, path: &Path) -> io::Result<Vec<PathBuf>>;
    fn exists(&self, path: &Path) -> bool;
}

#[derive(Debug, Default)]
pub struct NativeWatchFilesystem;
impl WatchFilesystem for NativeWatchFilesystem {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        std::fs::read(path)
    }
    fn list_files(&self, path: &Path) -> io::Result<Vec<PathBuf>> {
        let entries = std::fs::read_dir(path)?;
        Ok(entries
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_type().ok()?.is_file().then(|| entry.path()))
            .collect())
    }
    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }
}

pub trait WatchClock: Send + Sync {
    fn now(&self) -> SystemTime;
}
#[derive(Debug, Default)]
pub struct SystemWatchClock;
impl WatchClock for SystemWatchClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

pub trait ConfigReloadSink: Send + Sync {
    fn on_reload(&self, config: &WatcherConfig);
}
pub trait WatchPersistenceSink: Send + Sync {
    fn persist_config(&self) -> io::Result<()> {
        Ok(())
    }
    fn persist_auth(&self, _paths: &[PathBuf]) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Default)]
pub struct NoopConfigReloadSink;
impl ConfigReloadSink for NoopConfigReloadSink {
    fn on_reload(&self, _config: &WatcherConfig) {}
}
#[derive(Default)]
pub struct NoopPersistenceSink;
impl WatchPersistenceSink for NoopPersistenceSink {}

pub struct WatcherDependencies {
    pub filesystem: Arc<dyn WatchFilesystem>,
    pub clock: Arc<dyn WatchClock>,
    pub events: Arc<dyn WatchEventSource>,
    pub config_decoder: Arc<dyn ConfigDecoder>,
    pub reload_sink: Arc<dyn ConfigReloadSink>,
    pub persistence_sink: Arc<dyn WatchPersistenceSink>,
    pub dispatcher: Arc<AuthUpdateDispatcher>,
    pub plugin_parser: Option<Arc<dyn PluginAuthParser>>,
    pub coalesce_window: Duration,
    pub remove_debounce: Duration,
}

pub(super) struct WatcherState {
    pub(super) config: WatcherConfig,
    pub(super) config_hash: Option<String>,
    pub(super) auth_hashes: HashMap<PathBuf, String>,
    pub(super) auths: BTreeMap<String, SynthesizedAuth>,
    pub(super) pending: BTreeMap<PathBuf, FsEvent>,
    pub(super) last_remove: HashMap<PathBuf, SystemTime>,
}

struct Lifecycle {
    stop: Arc<AtomicBool>,
    worker: JoinHandle<()>,
}

pub struct Watcher {
    config_path: PathBuf,
    auth_dir: PathBuf,
    dependencies: Arc<WatcherDependencies>,
    state: Arc<Mutex<WatcherState>>,
    lifecycle: Mutex<Option<Lifecycle>>,
}

#[derive(Debug)]
pub enum WatcherError {
    MissingConfig(PathBuf),
    MissingAuthDir(PathBuf),
    Io(io::Error),
    AlreadyStarted,
}
impl fmt::Display for WatcherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingConfig(path) => {
                write!(f, "watcher config does not exist: {}", path.display())
            }
            Self::MissingAuthDir(path) => write!(
                f,
                "watcher auth directory does not exist: {}",
                path.display()
            ),
            Self::Io(error) => write!(f, "watcher I/O error: {error}"),
            Self::AlreadyStarted => f.write_str("watcher already started"),
        }
    }
}
impl std::error::Error for WatcherError {}
impl From<io::Error> for WatcherError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl Watcher {
    pub fn new(config_path: PathBuf, auth_dir: PathBuf, dependencies: WatcherDependencies) -> Self {
        Self {
            config_path,
            auth_dir,
            dependencies: Arc::new(dependencies),
            state: Arc::new(Mutex::new(WatcherState {
                config: WatcherConfig::default(),
                config_hash: None,
                auth_hashes: HashMap::new(),
                auths: BTreeMap::new(),
                pending: BTreeMap::new(),
                last_remove: HashMap::new(),
            })),
            lifecycle: Mutex::new(None),
        }
    }

    pub fn start(&self) -> Result<(), WatcherError> {
        if !self.dependencies.filesystem.exists(&self.config_path) {
            return Err(WatcherError::MissingConfig(self.config_path.clone()));
        }
        if !self.dependencies.filesystem.exists(&self.auth_dir) {
            return Err(WatcherError::MissingAuthDir(self.auth_dir.clone()));
        }
        let mut lifecycle = self.lifecycle.lock().unwrap_or_else(|p| p.into_inner());
        if lifecycle.is_some() {
            return Err(WatcherError::AlreadyStarted);
        }
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            let _ =
                reload_config_if_changed(self.dependencies.as_ref(), &self.config_path, &mut state);
            reload_clients(self.dependencies.as_ref(), &self.auth_dir, &mut state);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let config_path = self.config_path.clone();
        let auth_dir = self.auth_dir.clone();
        let dependencies = Arc::clone(&self.dependencies);
        let state = Arc::clone(&self.state);
        let worker = thread::Builder::new()
            .name("cliproxy-watcher".into())
            .spawn(move || {
                events::run_event_loop(worker_stop, dependencies, config_path, auth_dir, state);
            })?;
        *lifecycle = Some(Lifecycle { stop, worker });
        Ok(())
    }

    pub fn stop(&self) {
        let lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        if let Some(lifecycle) = lifecycle {
            lifecycle.stop.store(true, Ordering::Release);
            self.dependencies.events.close();
            let _ = lifecycle.worker.join();
            self.dependencies.dispatcher.stop();
        }
    }

    pub fn snapshot_auths(&self) -> Vec<SynthesizedAuth> {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .auths
            .values()
            .cloned()
            .collect()
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        self.stop();
    }
}
