// ref: internal/logging/home_app_log_forwarder.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::global_logger::{LogEntry, LogFormatter};
use serde::{Deserialize, Serialize};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

pub trait HomeAppLogSink: Send + Sync {
    fn heartbeat_ok(&self) -> bool;
    fn push_app_log(&self, payload: &[u8]) -> io::Result<()>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HomeAppLogPayload {
    pub line: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub level: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub timestamp: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub request_id: String,
}

struct QueuedPayload {
    payload: HomeAppLogPayload,
    owner: Arc<dyn HomeAppLogSink>,
}

enum WorkerMessage {
    Payload(QueuedPayload),
    Stop,
}

pub struct HomeAppLogForwarder {
    formatter: LogFormatter,
    sender: mpsc::SyncSender<WorkerMessage>,
    worker: Mutex<Option<JoinHandle<()>>>,
    owner: Arc<Mutex<Option<Arc<dyn HomeAppLogSink>>>>,
    enabled: Arc<AtomicBool>,
    stopped: AtomicBool,
    dropped: AtomicU64,
}

impl HomeAppLogForwarder {
    pub fn start(queue_size: usize) -> Self {
        let (sender, receiver) = mpsc::sync_channel(queue_size.max(1));
        let owner = Arc::new(Mutex::new(None::<Arc<dyn HomeAppLogSink>>));
        let enabled = Arc::new(AtomicBool::new(true));
        let worker_owner = Arc::clone(&owner);
        let worker_enabled = Arc::clone(&enabled);
        let worker = thread::Builder::new()
            .name("cliproxy-home-app-log".to_owned())
            .spawn(move || {
                while let Ok(message) = receiver.recv() {
                    let WorkerMessage::Payload(queued) = message else {
                        break;
                    };
                    if !worker_enabled.load(Ordering::Acquire) {
                        continue;
                    }
                    let still_owner = worker_owner
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .as_ref()
                        .is_some_and(|owner| Arc::ptr_eq(owner, &queued.owner));
                    if !still_owner || !queued.owner.heartbeat_ok() {
                        continue;
                    }
                    let Ok(raw) = serde_json::to_vec(&queued.payload) else {
                        continue;
                    };
                    if let Err(error) = queued.owner.push_app_log(&raw) {
                        if is_home_app_log_unsupported(&error) {
                            let current = worker_owner
                                .lock()
                                .unwrap_or_else(|poison| poison.into_inner());
                            if current
                                .as_ref()
                                .is_some_and(|owner| Arc::ptr_eq(owner, &queued.owner))
                            {
                                worker_enabled.store(false, Ordering::Release);
                            }
                        }
                    }
                }
            })
            .expect("home app log worker must start");
        Self {
            formatter: LogFormatter,
            sender,
            worker: Mutex::new(Some(worker)),
            owner,
            enabled,
            stopped: AtomicBool::new(false),
            dropped: AtomicU64::new(0),
        }
    }

    pub fn bind(&self, owner: Arc<dyn HomeAppLogSink>) {
        if self.stopped.load(Ordering::Acquire) {
            return;
        }
        *self
            .owner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(owner);
        self.enabled.store(true, Ordering::Release);
    }

    pub fn deactivate(&self, owner: &Arc<dyn HomeAppLogSink>) {
        let mut current = self
            .owner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if current
            .as_ref()
            .is_some_and(|value| Arc::ptr_eq(value, owner))
        {
            *current = None;
        }
    }

    pub fn fire(&self, entry: &LogEntry) {
        if self.stopped.load(Ordering::Acquire) || !self.enabled.load(Ordering::Acquire) {
            return;
        }
        let owner = self
            .owner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        let Some(owner) = owner.filter(|owner| owner.heartbeat_ok()) else {
            return;
        };
        let request_id = entry
            .fields
            .get("request_id")
            .map(|value| value.trim())
            .filter(|value| !value.is_empty() && *value != "--------")
            .unwrap_or_default()
            .to_owned();
        let timestamp: chrono::DateTime<chrono::Utc> = entry.timestamp.into();
        let payload = QueuedPayload {
            payload: HomeAppLogPayload {
                line: self.formatter.format(entry),
                level: entry.level.to_string(),
                timestamp: timestamp.to_rfc3339_opts(chrono::SecondsFormat::Nanos, true),
                request_id,
            },
            owner,
        };
        if self
            .sender
            .try_send(WorkerMessage::Payload(payload))
            .is_err()
        {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn stop(&self) {
        if self.stopped.swap(true, Ordering::AcqRel) {
            return;
        }
        self.enabled.store(false, Ordering::Release);
        self.owner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
        let _ = self.sender.send(WorkerMessage::Stop);
        if let Some(worker) = self
            .worker
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take()
        {
            let _ = worker.join();
        }
    }
}

impl Drop for HomeAppLogForwarder {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn is_home_app_log_unsupported(error: &io::Error) -> bool {
    let message = error.to_string().trim().to_ascii_lowercase();
    ["unsupported key", "unknown command", "unsupported command"]
        .iter()
        .any(|needle| message.contains(needle))
}
