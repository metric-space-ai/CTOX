// ref: internal/pluginhost/http_stream_bridge.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: HTTP stream handles are instance-local and bounded by process lifecycle
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{mpsc, Mutex as AsyncMutex};

use crate::sdk::pluginapi::{Headers, HttpStreamChunk, HttpStreamResponse};

struct HttpStreamEntry {
    owner_plugin_id: String,
    chunks: AsyncMutex<mpsc::Receiver<HttpStreamChunk>>,
}

#[derive(Default)]
pub struct HttpStreamBridge {
    next: AtomicU64,
    streams: Mutex<BTreeMap<String, Arc<HttpStreamEntry>>>,
}

impl HttpStreamBridge {
    pub fn open(&self, owner_plugin_id: &str, response: HttpStreamResponse) -> HttpStreamHandle {
        let stream_id = self
            .next
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1)
            .to_string();
        self.streams
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                stream_id.clone(),
                Arc::new(HttpStreamEntry {
                    owner_plugin_id: owner_plugin_id.trim().to_owned(),
                    chunks: AsyncMutex::new(response.chunks),
                }),
            );
        HttpStreamHandle {
            status_code: response.status_code,
            headers: response.headers,
            stream_id,
        }
    }

    pub async fn read(
        &self,
        owner_plugin_id: &str,
        stream_id: &str,
    ) -> Result<HttpStreamRead, HttpStreamBridgeError> {
        let entry = self
            .streams
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(stream_id)
            .cloned()
            .ok_or(HttpStreamBridgeError::NotOpen)?;
        if entry.owner_plugin_id != owner_plugin_id.trim() {
            return Err(HttpStreamBridgeError::WrongOwner);
        }
        let chunk = entry.chunks.lock().await.recv().await;
        match chunk {
            Some(chunk) => Ok(HttpStreamRead {
                payload: chunk.payload,
                error: chunk.error.map(|error| error.to_string()),
                done: false,
            }),
            None => {
                self.close(owner_plugin_id, stream_id)?;
                Ok(HttpStreamRead {
                    payload: Vec::new(),
                    error: None,
                    done: true,
                })
            }
        }
    }

    pub fn close(
        &self,
        owner_plugin_id: &str,
        stream_id: &str,
    ) -> Result<bool, HttpStreamBridgeError> {
        let mut streams = self
            .streams
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(entry) = streams.get(stream_id) else {
            return Ok(false);
        };
        if entry.owner_plugin_id != owner_plugin_id.trim() {
            return Err(HttpStreamBridgeError::WrongOwner);
        }
        streams.remove(stream_id);
        Ok(true)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpStreamHandle {
    pub status_code: u16,
    pub headers: Headers,
    pub stream_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpStreamRead {
    pub payload: Vec<u8>,
    pub error: Option<String>,
    pub done: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpStreamBridgeError {
    NotOpen,
    WrongOwner,
}

impl std::fmt::Display for HttpStreamBridgeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NotOpen => "HTTP stream is not open",
            Self::WrongOwner => "HTTP stream belongs to another plugin",
        })
    }
}

impl std::error::Error for HttpStreamBridgeError {}
