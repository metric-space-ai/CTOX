// ref: internal/pluginhost/host_model_stream_callbacks.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: model callbacks use injected execution authority and owner-bound stream handles
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use serde::Serialize;
use serde_json::{value::to_raw_value, value::RawValue};

use crate::sdk::pluginabi::{
    Envelope, METHOD_HOST_MODEL_EXECUTE, METHOD_HOST_MODEL_EXECUTE_STREAM,
    METHOD_HOST_MODEL_STREAM_CLOSE, METHOD_HOST_MODEL_STREAM_READ,
};
use crate::sdk::pluginapi::{
    HostModelExecutionRequest, HostModelExecutionResponse, HostModelStreamCloseRequest,
    HostModelStreamReadRequest, PluginFuture,
};

use super::abi::PluginClientError;
use super::callback_contexts::CallbackAuthority;
use super::host_callbacks::{HostCallbackFuture, HostCallbackHandler, HostCallbackRouter};
use super::model_stream_bridge::{ModelExecutionStream, ModelStreamBridge};
use super::rpc_schema::RpcEmptyResponse;

pub trait HostModelExecutionAuthority: Send + Sync {
    fn execute<'a>(
        &'a self,
        caller_plugin_id: &'a str,
        request: HostModelExecutionRequest,
    ) -> PluginFuture<'a, HostModelExecutionResponse>;

    fn execute_stream<'a>(
        &'a self,
        caller_plugin_id: &'a str,
        request: HostModelExecutionRequest,
    ) -> PluginFuture<'a, ModelExecutionStream>;
}

#[derive(Clone)]
struct ModelCallbackHandler {
    authority: Arc<dyn HostModelExecutionAuthority>,
    streams: Arc<ModelStreamBridge>,
    operation: ModelOperation,
}

#[derive(Clone, Copy)]
enum ModelOperation {
    Execute,
    ExecuteStream,
    Read,
    Close,
}

impl HostCallbackHandler for ModelCallbackHandler {
    fn call<'a>(
        &'a self,
        callback: &'a CallbackAuthority,
        payload: &'a RawValue,
    ) -> HostCallbackFuture<'a> {
        Box::pin(async move {
            match self.operation {
                ModelOperation::Execute => {
                    let request = decode(payload)?;
                    let response = self
                        .authority
                        .execute(callback.plugin_id(), request)
                        .await
                        .map_err(plugin_error)?;
                    envelope(&response)
                }
                ModelOperation::ExecuteStream => {
                    let request = decode(payload)?;
                    let stream = self
                        .authority
                        .execute_stream(callback.plugin_id(), request)
                        .await
                        .map_err(plugin_error)?;
                    envelope(&self.streams.open(callback.plugin_id(), stream))
                }
                ModelOperation::Read => {
                    let request: HostModelStreamReadRequest = decode(payload)?;
                    let response = self
                        .streams
                        .read(callback.plugin_id(), &request.stream_id)
                        .await
                        .map_err(|error| PluginClientError::Transport(error.to_string()))?;
                    envelope(&response)
                }
                ModelOperation::Close => {
                    let request: HostModelStreamCloseRequest = decode(payload)?;
                    self.streams
                        .close(callback.plugin_id(), &request.stream_id)
                        .map_err(|error| PluginClientError::Transport(error.to_string()))?;
                    envelope(&RpcEmptyResponse {})
                }
            }
        })
    }
}

pub fn install_host_model_callbacks(
    router: &mut HostCallbackRouter,
    authority: Arc<dyn HostModelExecutionAuthority>,
    streams: Arc<ModelStreamBridge>,
) -> Result<(), super::host_callbacks::HostCallbackRouteError> {
    for (method, operation) in [
        (METHOD_HOST_MODEL_EXECUTE, ModelOperation::Execute),
        (
            METHOD_HOST_MODEL_EXECUTE_STREAM,
            ModelOperation::ExecuteStream,
        ),
        (METHOD_HOST_MODEL_STREAM_READ, ModelOperation::Read),
        (METHOD_HOST_MODEL_STREAM_CLOSE, ModelOperation::Close),
    ] {
        router.register(
            method,
            Arc::new(ModelCallbackHandler {
                authority: authority.clone(),
                streams: streams.clone(),
                operation,
            }),
        )?;
    }
    Ok(())
}

fn decode<T: serde::de::DeserializeOwned>(payload: &RawValue) -> Result<T, PluginClientError> {
    serde_json::from_str(payload.get()).map_err(|_| PluginClientError::InvalidRequest)
}

fn envelope(value: &impl Serialize) -> Result<Envelope, PluginClientError> {
    Ok(Envelope::success(Some(
        to_raw_value(value).map_err(|_| PluginClientError::InvalidResponse)?,
    )))
}

fn plugin_error(error: crate::sdk::pluginapi::PluginExecutionError) -> PluginClientError {
    PluginClientError::Transport(error.to_string())
}
