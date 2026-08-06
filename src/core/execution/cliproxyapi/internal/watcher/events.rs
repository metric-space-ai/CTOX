// ref: internal/watcher/events.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::clients::reload_clients;
use super::config_reload::reload_config_if_changed;
use super::runtime::{WatchFilesystem, WatcherDependencies, WatcherState};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsEventKind {
    Create,
    Write,
    Remove,
    Rename,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsEvent {
    pub path: PathBuf,
    pub kind: FsEventKind,
    pub observed_at: SystemTime,
}

pub trait WatchEventSource: Send + Sync {
    fn recv(&self, timeout: Duration) -> io::Result<Option<FsEvent>>;
    fn close(&self);
}

pub struct ChannelEventSource {
    receiver: Mutex<mpsc::Receiver<FsEvent>>,
    sender: Mutex<Option<mpsc::Sender<FsEvent>>>,
}
impl ChannelEventSource {
    pub fn new() -> Arc<Self> {
        let (sender, receiver) = mpsc::channel();
        Arc::new(Self {
            receiver: Mutex::new(receiver),
            sender: Mutex::new(Some(sender)),
        })
    }
    pub fn emit(&self, event: FsEvent) -> bool {
        self.sender
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .is_some_and(|sender| sender.send(event).is_ok())
    }
}
impl WatchEventSource for ChannelEventSource {
    fn recv(&self, timeout: Duration) -> io::Result<Option<FsEvent>> {
        match self
            .receiver
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .recv_timeout(timeout)
        {
            Ok(event) => Ok(Some(event)),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => Ok(None),
        }
    }
    fn close(&self) {
        self.sender.lock().unwrap_or_else(|p| p.into_inner()).take();
    }
}

/// Supervised polling source for hosts without a native notification adapter.
pub struct PollingEventSource {
    filesystem: Arc<dyn WatchFilesystem>,
    clock: Arc<dyn super::runtime::WatchClock>,
    roots: Vec<PathBuf>,
    snapshots: Mutex<HashMap<PathBuf, String>>,
    pending: Mutex<Vec<FsEvent>>,
    closed: AtomicBool,
}
impl PollingEventSource {
    pub fn new(
        filesystem: Arc<dyn WatchFilesystem>,
        clock: Arc<dyn super::runtime::WatchClock>,
        roots: Vec<PathBuf>,
    ) -> Self {
        Self {
            filesystem,
            clock,
            roots,
            snapshots: Mutex::new(HashMap::new()),
            pending: Mutex::new(Vec::new()),
            closed: AtomicBool::new(false),
        }
    }
    fn scan(&self) {
        let mut current = HashMap::new();
        for root in &self.roots {
            let files = if self.filesystem.exists(root) && root.extension().is_some() {
                vec![root.clone()]
            } else {
                self.filesystem.list_files(root).unwrap_or_default()
            };
            for path in files {
                if let Ok(bytes) = self.filesystem.read(&path) {
                    current.insert(path, format!("{:x}", Sha256::digest(bytes)));
                }
            }
        }
        let mut previous = self.snapshots.lock().unwrap_or_else(|p| p.into_inner());
        let now = self.clock.now();
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        for (path, hash) in &current {
            match previous.get(path) {
                None => pending.push(FsEvent {
                    path: path.clone(),
                    kind: FsEventKind::Create,
                    observed_at: now,
                }),
                Some(old) if old != hash => pending.push(FsEvent {
                    path: path.clone(),
                    kind: FsEventKind::Write,
                    observed_at: now,
                }),
                _ => {}
            }
        }
        for path in previous.keys().filter(|path| !current.contains_key(*path)) {
            pending.push(FsEvent {
                path: path.clone(),
                kind: FsEventKind::Remove,
                observed_at: now,
            });
        }
        *previous = current;
    }
}
impl WatchEventSource for PollingEventSource {
    fn recv(&self, timeout: Duration) -> io::Result<Option<FsEvent>> {
        if self.closed.load(Ordering::Acquire) {
            return Ok(None);
        }
        if let Some(event) = self.pending.lock().unwrap_or_else(|p| p.into_inner()).pop() {
            return Ok(Some(event));
        }
        thread::sleep(timeout.min(Duration::from_millis(100)));
        self.scan();
        Ok(self.pending.lock().unwrap_or_else(|p| p.into_inner()).pop())
    }
    fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }
}

pub(super) fn run_event_loop(
    stop: Arc<AtomicBool>,
    dependencies: Arc<WatcherDependencies>,
    config_path: PathBuf,
    auth_dir: PathBuf,
    state: Arc<Mutex<WatcherState>>,
) {
    while !stop.load(Ordering::Acquire) {
        if let Ok(Some(event)) = dependencies.events.recv(Duration::from_millis(50)) {
            let mut state = state.lock().unwrap_or_else(|p| p.into_inner());
            let path = normalize_path(&event.path);
            if event.kind == FsEventKind::Remove {
                if state
                    .last_remove
                    .get(&path)
                    .and_then(|last| event.observed_at.duration_since(*last).ok())
                    .is_some_and(|elapsed| elapsed < dependencies.remove_debounce)
                {
                    continue;
                }
                state.last_remove.insert(path.clone(), event.observed_at);
            }
            state.pending.insert(path, event);
        }
        let now = dependencies.clock.now();
        let due = {
            let mut state = state.lock().unwrap_or_else(|p| p.into_inner());
            let keys = state
                .pending
                .iter()
                .filter(|(_, event)| {
                    now.duration_since(event.observed_at).unwrap_or_default()
                        >= dependencies.coalesce_window
                })
                .map(|(path, _)| path.clone())
                .collect::<Vec<_>>();
            keys.into_iter()
                .filter_map(|path| state.pending.remove(&path))
                .collect::<Vec<_>>()
        };
        if due.is_empty() {
            continue;
        }
        let config_changed = due
            .iter()
            .any(|event| normalize_path(&event.path) == normalize_path(&config_path));
        let auth_changed = due
            .iter()
            .any(|event| event.path.starts_with(&auth_dir) && is_auth_file(&event.path));
        let mut state = state.lock().unwrap_or_else(|p| p.into_inner());
        if config_changed {
            let _ = reload_config_if_changed(dependencies.as_ref(), &config_path, &mut state);
        }
        if config_changed || auth_changed {
            reload_clients(dependencies.as_ref(), &auth_dir, &mut state);
        }
    }
}

pub fn match_provider(provider: &str, targets: &[String]) -> Option<String> {
    targets
        .iter()
        .find(|target| target.eq_ignore_ascii_case(provider.trim()))
        .cloned()
}
pub fn normalize_path(path: &Path) -> PathBuf {
    path.components().collect()
}
pub fn is_auth_file(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
}
