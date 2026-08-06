// ref: internal/pluginhost/adapters_executors.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: executor calls and callback streams cross owner-bound process RPC
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::sdk::pluginabi::{
    METHOD_EXECUTOR_COUNT_TOKENS, METHOD_EXECUTOR_EXECUTE, METHOD_EXECUTOR_EXECUTE_STREAM,
    METHOD_EXECUTOR_HTTP_REQUEST,
};
use crate::sdk::pluginapi::{
    ExecutorHttpRequest, ExecutorHttpResponse, ExecutorRequest, ExecutorResponse,
    ExecutorStreamChunk, ExecutorStreamResponse, Headers, PluginFuture, ProviderExecutor,
};

use super::adapters::RpcCapabilityClient;
use super::callback_contexts::CallbackAuthority;
use super::stream_bridge::StreamBridge;

#[derive(Clone)]
pub struct RpcProviderExecutor {
    identifier: String,
    client: RpcCapabilityClient,
    streams: StreamBridge,
}

impl std::fmt::Debug for RpcProviderExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RpcProviderExecutor")
            .field("identifier", &self.identifier)
            .field("client", &self.client)
            .finish_non_exhaustive()
    }
}

impl RpcProviderExecutor {
    pub fn new(
        identifier: &str,
        client: RpcCapabilityClient,
        streams: StreamBridge,
    ) -> Result<Self, InvalidIdentifier> {
        let identifier = identifier.trim().to_ascii_lowercase();
        if identifier.is_empty() || identifier.len() > 128 {
            return Err(InvalidIdentifier);
        }
        Ok(Self {
            identifier,
            client,
            streams,
        })
    }
}

impl ProviderExecutor for RpcProviderExecutor {
    fn identifier(&self) -> &str {
        &self.identifier
    }

    fn execute<'a>(&'a self, request: ExecutorRequest) -> PluginFuture<'a, ExecutorResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_EXECUTOR_EXECUTE, &request, None)
                .await
                .map_err(plugin_error)
        })
    }

    fn execute_stream<'a>(
        &'a self,
        request: ExecutorRequest,
    ) -> PluginFuture<'a, ExecutorStreamResponse> {
        Box::pin(async move {
            let callback_lease = self
                .client
                .contexts()
                .open(CallbackAuthority::new(self.client.plugin_id(), None));
            let (stream_id, mut bridge_chunks, stream_lease) =
                self.streams.open(self.client.plugin_id());
            let wire = RpcExecutorStreamRequest {
                request: &request,
                stream_id: &stream_id,
                host_callback_id: callback_lease.id(),
            };
            let response: RpcExecutorStreamResponse = self
                .client
                .client()
                .call(METHOD_EXECUTOR_EXECUTE_STREAM, &wire, None)
                .await
                .map_err(plugin_error)?;
            let RpcExecutorStreamResponse {
                headers,
                chunks: inline_chunks,
            } = response;
            let capacity = inline_chunks.len().clamp(1, 32);
            let (sender, receiver) = mpsc::channel(capacity);
            tokio::spawn(async move {
                let _callback_lease = callback_lease;
                let _stream_lease = stream_lease;
                if !inline_chunks.is_empty() {
                    for chunk in inline_chunks {
                        if sender.send(chunk.into_chunk()).await.is_err() {
                            break;
                        }
                    }
                    return;
                }
                while let Some(chunk) = bridge_chunks.recv().await {
                    if sender.send(chunk).await.is_err() {
                        break;
                    }
                }
            });
            Ok(ExecutorStreamResponse {
                headers,
                chunks: receiver,
            })
        })
    }

    fn count_tokens<'a>(&'a self, request: ExecutorRequest) -> PluginFuture<'a, ExecutorResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_EXECUTOR_COUNT_TOKENS, &request, None)
                .await
                .map_err(plugin_error)
        })
    }

    fn http_request<'a>(
        &'a self,
        request: ExecutorHttpRequest,
    ) -> PluginFuture<'a, ExecutorHttpResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_EXECUTOR_HTTP_REQUEST, &request, None)
                .await
                .map_err(plugin_error)
        })
    }
}

#[derive(Serialize)]
struct RpcExecutorStreamRequest<'a> {
    #[serde(flatten)]
    request: &'a ExecutorRequest,
    stream_id: &'a str,
    host_callback_id: &'a str,
}

#[derive(Default, Deserialize)]
#[serde(default, rename_all = "PascalCase")]
struct RpcExecutorStreamResponse {
    headers: Headers,
    chunks: Vec<RpcExecutorChunk>,
}

#[derive(Default, Deserialize)]
#[serde(default, rename_all = "PascalCase")]
struct RpcExecutorChunk {
    #[serde(with = "wire_bytes")]
    payload: Vec<u8>,
    error: String,
}

impl RpcExecutorChunk {
    fn into_chunk(self) -> ExecutorStreamChunk {
        ExecutorStreamChunk {
            payload: self.payload,
            error: if self.error.trim().is_empty() {
                None
            } else {
                Some(Arc::new(ExecutorRemoteError(self.error)))
            },
        }
    }
}

fn plugin_error(
    error: super::abi::PluginClientError,
) -> crate::sdk::pluginapi::PluginExecutionError {
    Arc::new(error)
}

#[derive(Debug)]
struct ExecutorRemoteError(String);

impl std::fmt::Display for ExecutorRemoteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ExecutorRemoteError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidIdentifier;

impl std::fmt::Display for InvalidIdentifier {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("executor identifier is invalid")
    }
}

impl std::error::Error for InvalidIdentifier {}

mod wire_bytes {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use serde::{Deserialize, Deserializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        STANDARD
            .decode(String::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}
