// ref: internal/pluginhost/rpc_client.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: upstream RPC envelope semantics over the isolated process client
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use serde::{de::DeserializeOwned, Serialize};
use serde_json::{value::to_raw_value, Value};

use crate::sdk::pluginabi::{Envelope, SCHEMA_VERSION};

use super::abi::{PluginCall, PluginClient, PluginClientError};
use super::rpc_schema::{RpcLifecycleRequest, RpcRegistration};

#[derive(Clone)]
pub struct RpcPluginClient {
    client: Arc<dyn PluginClient>,
}

impl std::fmt::Debug for RpcPluginClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RpcPluginClient")
            .finish_non_exhaustive()
    }
}

impl RpcPluginClient {
    pub fn new(client: Arc<dyn PluginClient>) -> Self {
        Self { client }
    }

    pub async fn register(
        &self,
        method: &str,
        config_yaml: Vec<u8>,
        deadline_unix_ms: Option<u64>,
    ) -> Result<RpcRegistration, PluginClientError> {
        let registration: RpcRegistration = self
            .call(
                method,
                &RpcLifecycleRequest {
                    config_yaml,
                    schema_version: SCHEMA_VERSION,
                },
                deadline_unix_ms,
            )
            .await?;
        if registration.schema_version > SCHEMA_VERSION
            || registration.metadata.name.trim().is_empty()
        {
            return Err(PluginClientError::InvalidResponse);
        }
        Ok(registration)
    }

    pub async fn call<Request, Response>(
        &self,
        method: &str,
        request: &Request,
        deadline_unix_ms: Option<u64>,
    ) -> Result<Response, PluginClientError>
    where
        Request: Serialize + ?Sized,
        Response: DeserializeOwned,
    {
        let request =
            serde_json::to_value(request).map_err(|_| PluginClientError::InvalidRequest)?;
        let sanitized = sanitize_plugin_value(request);
        let payload = to_raw_value(&sanitized).map_err(|_| PluginClientError::InvalidRequest)?;
        let envelope = self
            .client
            .call(PluginCall {
                method: method.to_owned(),
                payload,
                deadline_unix_ms,
            })
            .await?;
        decode_envelope(envelope)
    }

    pub async fn shutdown(&self) -> Result<(), PluginClientError> {
        self.client.shutdown().await
    }

    pub(crate) fn transport(&self) -> &Arc<dyn PluginClient> {
        &self.client
    }
}

pub fn decode_envelope<T: DeserializeOwned>(envelope: Envelope) -> Result<T, PluginClientError> {
    if !envelope.ok {
        let error = envelope.error.ok_or(PluginClientError::InvalidResponse)?;
        return Err(PluginClientError::Plugin {
            code: error.code,
            message: error.message,
            retryable: error.retryable,
            http_status: error.http_status,
        });
    }
    if envelope.error.is_some() {
        return Err(PluginClientError::InvalidResponse);
    }
    let result = envelope.result.ok_or(PluginClientError::InvalidResponse)?;
    serde_json::from_str(result.get()).map_err(|_| PluginClientError::InvalidResponse)
}

/// JSON is the process ABI. Values that cannot cross that ABI never enter this
/// function; recursively normalizing objects gives deterministic key ordering
/// and prevents host-only Rust handles from being smuggled through extensions.
pub fn sanitize_plugin_value(value: Value) -> Value {
    match value {
        Value::Array(values) => {
            Value::Array(values.into_iter().map(sanitize_plugin_value).collect())
        }
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .filter(|(key, _)| !key.starts_with("__ctox_host_"))
                .map(|(key, value)| (key, sanitize_plugin_value(value)))
                .collect(),
        ),
        scalar => scalar,
    }
}
