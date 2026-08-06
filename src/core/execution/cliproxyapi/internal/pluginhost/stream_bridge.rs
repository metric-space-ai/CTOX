// ref: internal/pluginhost/stream_bridge.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: bounded process-stream registry replaces in-process callback channels
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use tokio::sync::mpsc;

use crate::sdk::pluginapi::ExecutorStreamChunk;

pub const STREAM_BRIDGE_BUFFER_SIZE: usize = 16;

struct StreamEntry {
    owner_plugin_id: String,
    sender: mpsc::Sender<ExecutorStreamChunk>,
    cancelled: Arc<AtomicBool>,
}

struct BridgeInner {
    next: AtomicU64,
    streams: Mutex<BTreeMap<String, StreamEntry>>,
}

#[derive(Clone)]
pub struct StreamBridge {
    inner: Arc<BridgeInner>,
}

impl Default for StreamBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamBridge {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(BridgeInner {
                next: AtomicU64::new(0),
                streams: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    pub fn open(
        &self,
        owner_plugin_id: &str,
    ) -> (String, mpsc::Receiver<ExecutorStreamChunk>, StreamLease) {
        let id = self
            .inner
            .next
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1)
            .to_string();
        let (sender, receiver) = mpsc::channel(STREAM_BRIDGE_BUFFER_SIZE);
        let cancelled = Arc::new(AtomicBool::new(false));
        self.inner
            .streams
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                id.clone(),
                StreamEntry {
                    owner_plugin_id: owner_plugin_id.trim().to_owned(),
                    sender,
                    cancelled: cancelled.clone(),
                },
            );
        let lease = StreamLease {
            id: id.clone(),
            bridge: Arc::downgrade(&self.inner),
            cancelled,
            closed: false,
        };
        (id, receiver, lease)
    }

    pub async fn emit(
        &self,
        owner_plugin_id: &str,
        id: &str,
        payload: Vec<u8>,
    ) -> Result<(), StreamBridgeError> {
        self.emit_chunk(owner_plugin_id, id, payload, None).await
    }

    pub async fn emit_chunk(
        &self,
        owner_plugin_id: &str,
        id: &str,
        payload: Vec<u8>,
        error_message: Option<String>,
    ) -> Result<(), StreamBridgeError> {
        let sender = self.sender(owner_plugin_id, id)?;
        sender
            .send(ExecutorStreamChunk {
                payload,
                error: error_message
                    .filter(|message| !message.trim().is_empty())
                    .map(|message| Arc::new(StreamTerminalError(message)) as _),
            })
            .await
            .map_err(|_| StreamBridgeError::Closed)
    }

    pub async fn close(
        &self,
        owner_plugin_id: &str,
        id: &str,
        error_message: Option<String>,
    ) -> Result<(), StreamBridgeError> {
        let entry = {
            let mut streams = self
                .inner
                .streams
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(entry) = streams.get(id) else {
                return Err(StreamBridgeError::NotOpen);
            };
            if entry.owner_plugin_id != owner_plugin_id.trim() {
                return Err(StreamBridgeError::WrongOwner);
            }
            streams.remove(id).expect("stream checked above")
        };
        entry.cancelled.store(true, Ordering::Release);
        if let Some(message) = error_message.filter(|message| !message.trim().is_empty()) {
            entry
                .sender
                .send(ExecutorStreamChunk {
                    payload: Vec::new(),
                    error: Some(Arc::new(StreamTerminalError(message))),
                })
                .await
                .map_err(|_| StreamBridgeError::Closed)?;
        }
        Ok(())
    }

    pub fn is_open(&self, id: &str) -> bool {
        self.inner
            .streams
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(id)
    }

    fn sender(
        &self,
        owner_plugin_id: &str,
        id: &str,
    ) -> Result<mpsc::Sender<ExecutorStreamChunk>, StreamBridgeError> {
        let streams = self
            .inner
            .streams
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = streams.get(id).ok_or(StreamBridgeError::NotOpen)?;
        if entry.owner_plugin_id != owner_plugin_id.trim() {
            return Err(StreamBridgeError::WrongOwner);
        }
        Ok(entry.sender.clone())
    }
}

pub struct StreamLease {
    id: String,
    bridge: Weak<BridgeInner>,
    cancelled: Arc<AtomicBool>,
    closed: bool,
}

impl StreamLease {
    pub fn cancel(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.cancelled.store(true, Ordering::Release);
        if let Some(bridge) = self.bridge.upgrade() {
            bridge
                .streams
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&self.id);
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

impl Drop for StreamLease {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamBridgeError {
    NotOpen,
    Closed,
    WrongOwner,
}

impl fmt::Display for StreamBridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NotOpen => "stream is not open",
            Self::Closed => "stream receiver is closed",
            Self::WrongOwner => "stream belongs to another plugin",
        })
    }
}

impl std::error::Error for StreamBridgeError {}

#[derive(Debug)]
struct StreamTerminalError(String);

impl fmt::Display for StreamTerminalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for StreamTerminalError {}
