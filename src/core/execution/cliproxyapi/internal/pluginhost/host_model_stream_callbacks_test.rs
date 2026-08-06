// ref: internal/pluginhost/host_model_stream_callbacks_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: owner-bound model callback stream lifecycle over process RPC
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use serde_json::value::to_raw_value;
use tokio::sync::mpsc;

use crate::sdk::pluginabi::{METHOD_HOST_MODEL_EXECUTE_STREAM, METHOD_HOST_MODEL_STREAM_READ};
use crate::sdk::pluginapi::{
    HostModelExecutionRequest, HostModelExecutionResponse, HostModelStreamReadRequest,
    HostModelStreamReadResponse, HostModelStreamResponse, PluginFuture,
};

use super::callback_contexts::{CallbackAuthority, CallbackContextRegistry};
use super::host_callbacks::{HostCallbackRouteError, HostCallbackRouter};
use super::host_model_stream_callbacks::{
    install_host_model_callbacks, HostModelExecutionAuthority,
};
use super::model_stream_bridge::{ModelExecutionChunk, ModelExecutionStream, ModelStreamBridge};

#[derive(Default)]
struct Authority {
    callers: Mutex<Vec<String>>,
}

impl HostModelExecutionAuthority for Authority {
    fn execute<'a>(
        &'a self,
        caller_plugin_id: &'a str,
        _request: HostModelExecutionRequest,
    ) -> PluginFuture<'a, HostModelExecutionResponse> {
        Box::pin(async move {
            self.callers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(caller_plugin_id.to_owned());
            Ok(HostModelExecutionResponse::default())
        })
    }

    fn execute_stream<'a>(
        &'a self,
        caller_plugin_id: &'a str,
        _request: HostModelExecutionRequest,
    ) -> PluginFuture<'a, ModelExecutionStream> {
        Box::pin(async move {
            self.callers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(caller_plugin_id.to_owned());
            let (sender, receiver) = mpsc::channel(1);
            sender
                .send(ModelExecutionChunk {
                    payload: b"model-chunk".to_vec(),
                    error: None,
                })
                .await
                .unwrap();
            drop(sender);
            Ok(ModelExecutionStream {
                status_code: 200,
                headers: Default::default(),
                chunks: receiver,
            })
        })
    }
}

#[tokio::test]
async fn model_stream_callbacks_bind_owner_and_reach_terminal_done() {
    let contexts = CallbackContextRegistry::new();
    let mut owner_lease = contexts.open(CallbackAuthority::new("plugin-a", None));
    let other_lease = contexts.open(CallbackAuthority::new("plugin-b", None));
    let mut router = HostCallbackRouter::new(contexts);
    let authority = Arc::new(Authority::default());
    install_host_model_callbacks(
        &mut router,
        authority.clone(),
        Arc::new(ModelStreamBridge::default()),
    )
    .unwrap();

    let request = to_raw_value(&HostModelExecutionRequest {
        model: "target".to_owned(),
        stream: true,
        ..HostModelExecutionRequest::default()
    })
    .unwrap();
    let response = router
        .dispatch(
            "plugin-a",
            owner_lease.id(),
            METHOD_HOST_MODEL_EXECUTE_STREAM,
            &request,
            0,
        )
        .await
        .unwrap();
    let handle: HostModelStreamResponse =
        serde_json::from_str(response.result.unwrap().get()).unwrap();
    assert_eq!(handle.status_code, 200);
    assert_eq!(
        authority
            .callers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_slice(),
        &["plugin-a"]
    );

    let read = to_raw_value(&HostModelStreamReadRequest {
        stream_id: handle.stream_id.clone(),
    })
    .unwrap();
    let wrong_owner = router
        .dispatch(
            "plugin-b",
            other_lease.id(),
            METHOD_HOST_MODEL_STREAM_READ,
            &read,
            0,
        )
        .await
        .err()
        .unwrap();
    assert!(matches!(wrong_owner, HostCallbackRouteError::Handler(_)));
    let first = router
        .dispatch(
            "plugin-a",
            owner_lease.id(),
            METHOD_HOST_MODEL_STREAM_READ,
            &read,
            0,
        )
        .await
        .unwrap();
    let first: HostModelStreamReadResponse =
        serde_json::from_str(first.result.unwrap().get()).unwrap();
    assert_eq!(first.payload, b"model-chunk");
    assert!(!first.done);
    let done = router
        .dispatch(
            "plugin-a",
            owner_lease.id(),
            METHOD_HOST_MODEL_STREAM_READ,
            &read,
            0,
        )
        .await
        .unwrap();
    let done: HostModelStreamReadResponse =
        serde_json::from_str(done.result.unwrap().get()).unwrap();
    assert!(done.done);
    owner_lease.close();
}
