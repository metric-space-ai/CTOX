// ref: internal/pluginhost/auth_callbacks.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: auth persistence is delegated to an injected typed CTOX authority
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{value::to_raw_value, value::RawValue, Value};

use crate::sdk::pluginabi::{
    Envelope, METHOD_HOST_AUTH_GET, METHOD_HOST_AUTH_GET_RUNTIME, METHOD_HOST_AUTH_LIST,
    METHOD_HOST_AUTH_SAVE,
};
use crate::sdk::pluginapi::{
    HostAuthFileEntry, HostAuthGetRequest, HostAuthGetResponse, HostAuthGetRuntimeResponse,
    HostAuthSaveRequest, HostAuthSaveResponse, PluginFuture,
};

use super::abi::PluginClientError;
use super::callback_contexts::CallbackAuthority;
use super::host_callbacks::{HostCallbackFuture, HostCallbackHandler, HostCallbackRouter};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HostAuthListResponse {
    pub files: Vec<HostAuthFileEntry>,
}

/// The plugin host never receives filesystem or secret-store authority. CTOX
/// supplies this narrow interface after applying its own policy and scoping.
pub trait HostAuthAuthority: Send + Sync {
    fn list<'a>(&'a self, caller_plugin_id: &'a str) -> PluginFuture<'a, Vec<HostAuthFileEntry>>;
    fn get<'a>(
        &'a self,
        caller_plugin_id: &'a str,
        auth_index: &'a str,
    ) -> PluginFuture<'a, HostAuthGetResponse>;
    fn get_runtime<'a>(
        &'a self,
        caller_plugin_id: &'a str,
        auth_index: &'a str,
    ) -> PluginFuture<'a, HostAuthGetRuntimeResponse>;
    fn save<'a>(
        &'a self,
        caller_plugin_id: &'a str,
        name: &'a str,
        json: Value,
    ) -> PluginFuture<'a, HostAuthSaveResponse>;
}

#[derive(Clone, Copy)]
enum AuthOperation {
    List,
    Get,
    GetRuntime,
    Save,
}

#[derive(Clone)]
struct AuthCallbackHandler {
    authority: Arc<dyn HostAuthAuthority>,
    operation: AuthOperation,
}

impl HostCallbackHandler for AuthCallbackHandler {
    fn call<'a>(
        &'a self,
        callback: &'a CallbackAuthority,
        payload: &'a RawValue,
    ) -> HostCallbackFuture<'a> {
        Box::pin(async move {
            match self.operation {
                AuthOperation::List => {
                    decode_optional_object(payload)?;
                    let files = self
                        .authority
                        .list(callback.plugin_id())
                        .await
                        .map_err(plugin_error)?;
                    envelope(&HostAuthListResponse { files })
                }
                AuthOperation::Get => {
                    let request: HostAuthGetRequest = decode(payload)?;
                    let auth_index = required(&request.auth_index)?;
                    let response = self
                        .authority
                        .get(callback.plugin_id(), auth_index)
                        .await
                        .map_err(plugin_error)?;
                    envelope(&response)
                }
                AuthOperation::GetRuntime => {
                    let request: HostAuthGetRequest = decode(payload)?;
                    let auth_index = required(&request.auth_index)?;
                    let response = self
                        .authority
                        .get_runtime(callback.plugin_id(), auth_index)
                        .await
                        .map_err(plugin_error)?;
                    envelope(&response)
                }
                AuthOperation::Save => {
                    let request: HostAuthSaveRequest = decode(payload)?;
                    let name = validate_auth_file_name(&request.name)?;
                    if !request.json.is_object() {
                        return Err(PluginClientError::InvalidRequest);
                    }
                    let response = self
                        .authority
                        .save(callback.plugin_id(), name, request.json)
                        .await
                        .map_err(plugin_error)?;
                    envelope(&response)
                }
            }
        })
    }
}

pub fn install_host_auth_callbacks(
    router: &mut HostCallbackRouter,
    authority: Arc<dyn HostAuthAuthority>,
) -> Result<(), super::host_callbacks::HostCallbackRouteError> {
    for (method, operation) in [
        (METHOD_HOST_AUTH_LIST, AuthOperation::List),
        (METHOD_HOST_AUTH_GET, AuthOperation::Get),
        (METHOD_HOST_AUTH_GET_RUNTIME, AuthOperation::GetRuntime),
        (METHOD_HOST_AUTH_SAVE, AuthOperation::Save),
    ] {
        router.register(
            method,
            Arc::new(AuthCallbackHandler {
                authority: authority.clone(),
                operation,
            }),
        )?;
    }
    Ok(())
}

fn validate_auth_file_name(name: &str) -> Result<&str, PluginClientError> {
    let name = required(name)?;
    if !name.to_ascii_lowercase().ends_with(".json")
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.chars().any(char::is_control)
    {
        return Err(PluginClientError::InvalidRequest);
    }
    Ok(name)
}

fn required(value: &str) -> Result<&str, PluginClientError> {
    let value = value.trim();
    if value.is_empty() {
        Err(PluginClientError::InvalidRequest)
    } else {
        Ok(value)
    }
}

fn decode<T: serde::de::DeserializeOwned>(payload: &RawValue) -> Result<T, PluginClientError> {
    serde_json::from_str(payload.get()).map_err(|_| PluginClientError::InvalidRequest)
}

fn decode_optional_object(payload: &RawValue) -> Result<(), PluginClientError> {
    let value: Value = decode(payload)?;
    if value.is_null() || value.is_object() {
        Ok(())
    } else {
        Err(PluginClientError::InvalidRequest)
    }
}

fn envelope(value: &impl Serialize) -> Result<Envelope, PluginClientError> {
    Ok(Envelope::success(Some(
        to_raw_value(value).map_err(|_| PluginClientError::InvalidResponse)?,
    )))
}

fn plugin_error(error: crate::sdk::pluginapi::PluginExecutionError) -> PluginClientError {
    PluginClientError::Transport(error.to_string())
}
