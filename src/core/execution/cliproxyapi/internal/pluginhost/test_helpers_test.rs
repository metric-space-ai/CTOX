// ref: internal/pluginhost/test_helpers_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: process fixture exercises lifecycle and management callback wrappers
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use serde_json::value::to_raw_value;
use tokio::sync::mpsc;

use crate::sdk::pluginabi::{
    Envelope, METHOD_MANAGEMENT_HANDLE, METHOD_MANAGEMENT_REGISTER, METHOD_REQUEST_COMPLETE,
};
use crate::sdk::pluginapi::{
    ManagementApi, ManagementRegistrationRequest, ManagementRequest, RequestCompletion,
    RequestLifecyclePlugin,
};

use super::abi::{PluginCall, PluginClient, PluginClientError, PluginFuture, PluginStream};
use super::adapters::{RpcCapabilityClient, RpcManagementApi};
use super::adapters_interceptors::RpcRequestLifecyclePlugin;
use super::callback_contexts::CallbackContextRegistry;
use super::rpc_client::RpcPluginClient;

#[derive(Default)]
struct FixtureClient {
    calls: Mutex<Vec<PluginCall>>,
}

impl PluginClient for FixtureClient {
    fn call<'a>(&'a self, call: PluginCall) -> PluginFuture<'a, Envelope> {
        Box::pin(async move {
            let response = match call.method.as_str() {
                METHOD_REQUEST_COMPLETE => serde_json::json!({}),
                METHOD_MANAGEMENT_REGISTER => serde_json::json!({
                    "routes": [{
                        "method": "POST",
                        "path": "/rotate",
                        "menu": "Runtime",
                        "description": "Rotate bounded state"
                    }]
                }),
                METHOD_MANAGEMENT_HANDLE => serde_json::json!({
                    "StatusCode": 202,
                    "Body": "YWNjZXB0ZWQ="
                }),
                _ => return Err(PluginClientError::UnsupportedCapability),
            };
            self.calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(call);
            Ok(Envelope::success(Some(to_raw_value(&response).unwrap())))
        })
    }

    fn call_stream<'a>(&'a self, _call: PluginCall) -> PluginFuture<'a, PluginStream> {
        Box::pin(async {
            let (_sender, receiver) = mpsc::channel(1);
            Ok(PluginStream { chunks: receiver })
        })
    }

    fn shutdown<'a>(&'a self) -> PluginFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}

fn fixture() -> (RpcCapabilityClient, Arc<FixtureClient>) {
    let transport = Arc::new(FixtureClient::default());
    let client = RpcCapabilityClient::new(
        "fixture-plugin",
        RpcPluginClient::new(transport.clone()),
        CallbackContextRegistry::new(),
    )
    .unwrap();
    (client, transport)
}

#[tokio::test]
async fn lifecycle_fixture_includes_nonempty_callback_id() {
    let (client, transport) = fixture();
    RpcRequestLifecyclePlugin::new(client)
        .handle_request_complete(RequestCompletion {
            request_id: "request-1".to_owned(),
            ..RequestCompletion::default()
        })
        .await
        .unwrap();
    let calls = transport
        .calls
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let payload: serde_json::Value = serde_json::from_str(calls[0].payload.get()).unwrap();
    assert_eq!(payload["RequestID"], "request-1");
    assert!(!payload["host_callback_id"].as_str().unwrap().is_empty());
}

#[tokio::test]
async fn management_fixture_binds_descriptor_to_rpc_handler() {
    let (client, transport) = fixture();
    let response = RpcManagementApi::new(client)
        .register_management(ManagementRegistrationRequest::default())
        .await
        .unwrap();
    assert_eq!(response.routes.len(), 1);
    let response = response.routes[0]
        .handler
        .handle_management(ManagementRequest {
            method: "POST".to_owned(),
            path: "/rotate".to_owned(),
            ..ManagementRequest::default()
        })
        .await
        .unwrap();
    assert_eq!(response.status_code, 202);
    assert_eq!(response.body, b"accepted");
    let calls = transport
        .calls
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(calls[1].method, METHOD_MANAGEMENT_HANDLE);
    let payload: serde_json::Value = serde_json::from_str(calls[1].payload.get()).unwrap();
    assert!(!payload["host_callback_id"].as_str().unwrap().is_empty());
}
