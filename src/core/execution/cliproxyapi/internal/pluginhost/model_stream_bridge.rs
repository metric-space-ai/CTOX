// ref: internal/pluginhost/model_stream_bridge.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: model stream handles are owner-bound to isolated callback contexts
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{mpsc, Mutex as AsyncMutex};

use crate::sdk::pluginapi::{
    Headers, HostModelStreamReadResponse, HostModelStreamResponse, PluginExecutionError,
};

pub struct ModelExecutionChunk {
    pub payload: Vec<u8>,
    pub error: Option<PluginExecutionError>,
}

pub struct ModelExecutionStream {
    pub status_code: u16,
    pub headers: Headers,
    pub chunks: mpsc::Receiver<ModelExecutionChunk>,
}

struct ModelStreamEntry {
    owner_plugin_id: String,
    chunks: AsyncMutex<mpsc::Receiver<ModelExecutionChunk>>,
}

#[derive(Default)]
pub struct ModelStreamBridge {
    next: AtomicU64,
    streams: Mutex<BTreeMap<String, Arc<ModelStreamEntry>>>,
}

impl ModelStreamBridge {
    pub fn open(
        &self,
        owner_plugin_id: &str,
        stream: ModelExecutionStream,
    ) -> HostModelStreamResponse {
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
                Arc::new(ModelStreamEntry {
                    owner_plugin_id: owner_plugin_id.to_owned(),
                    chunks: AsyncMutex::new(stream.chunks),
                }),
            );
        HostModelStreamResponse {
            status_code: stream.status_code,
            headers: stream.headers,
            stream_id,
        }
    }

    pub async fn read(
        &self,
        owner_plugin_id: &str,
        stream_id: &str,
    ) -> Result<HostModelStreamReadResponse, ModelStreamBridgeError> {
        let entry = self.entry(owner_plugin_id, stream_id)?;
        let chunk = {
            let mut chunks = entry.chunks.lock().await;
            chunks.recv().await
        };
        match chunk {
            Some(chunk) => Ok(HostModelStreamReadResponse {
                payload: chunk.payload,
                error: chunk
                    .error
                    .map(|error| error.to_string())
                    .unwrap_or_default(),
                done: false,
            }),
            None => {
                self.close(owner_plugin_id, stream_id)?;
                Ok(HostModelStreamReadResponse {
                    done: true,
                    ..HostModelStreamReadResponse::default()
                })
            }
        }
    }

    pub fn close(
        &self,
        owner_plugin_id: &str,
        stream_id: &str,
    ) -> Result<(), ModelStreamBridgeError> {
        self.entry(owner_plugin_id, stream_id)?;
        self.streams
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(stream_id);
        Ok(())
    }

    fn entry(
        &self,
        owner_plugin_id: &str,
        stream_id: &str,
    ) -> Result<Arc<ModelStreamEntry>, ModelStreamBridgeError> {
        let entry = self
            .streams
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(stream_id)
            .cloned()
            .ok_or(ModelStreamBridgeError::NotOpen)?;
        if entry.owner_plugin_id != owner_plugin_id {
            return Err(ModelStreamBridgeError::WrongOwner);
        }
        Ok(entry)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelStreamBridgeError {
    NotOpen,
    WrongOwner,
}

impl std::fmt::Display for ModelStreamBridgeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NotOpen => "model stream is not open",
            Self::WrongOwner => "model stream belongs to another plugin",
        })
    }
}

impl std::error::Error for ModelStreamBridgeError {}
