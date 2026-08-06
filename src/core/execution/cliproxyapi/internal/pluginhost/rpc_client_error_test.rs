// ref: internal/pluginhost/rpc_client_error_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: upstream RPC error semantics over the isolated process client
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use serde_json::{json, value::to_raw_value, Value};
use tokio::sync::mpsc;

use crate::sdk::pluginabi::{Envelope, Error};

use super::abi::{PluginCall, PluginClient, PluginClientError, PluginFuture, PluginStream};
use super::rpc_client::{sanitize_plugin_value, RpcPluginClient};

struct EnvelopeClient(Mutex<Option<Envelope>>);

impl PluginClient for EnvelopeClient {
    fn call<'a>(&'a self, _call: PluginCall) -> PluginFuture<'a, Envelope> {
        Box::pin(async move {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .ok_or(PluginClientError::Closed)
        })
    }

    fn call_stream<'a>(&'a self, _call: PluginCall) -> PluginFuture<'a, PluginStream> {
        Box::pin(async move {
            let (_sender, receiver) = mpsc::channel(1);
            Ok(PluginStream { chunks: receiver })
        })
    }

    fn shutdown<'a>(&'a self) -> PluginFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn plugin_error_envelope_preserves_typed_retry_and_http_status() {
    let client = RpcPluginClient::new(Arc::new(EnvelopeClient(Mutex::new(Some(
        Envelope::failure(Error {
            code: "quota".to_owned(),
            message: "try later".to_owned(),
            retryable: true,
            http_status: 429,
        }),
    )))));
    let error = client
        .call::<_, Value>("executor.execute", &json!({}), None)
        .await
        .unwrap_err();
    assert_eq!(
        error,
        PluginClientError::Plugin {
            code: "quota".to_owned(),
            message: "try later".to_owned(),
            retryable: true,
            http_status: 429,
        }
    );
}

#[tokio::test]
async fn malformed_success_envelope_fails_closed() {
    let client = RpcPluginClient::new(Arc::new(EnvelopeClient(Mutex::new(Some(Envelope {
        ok: true,
        result: Some(to_raw_value(&"not an integer").unwrap()),
        error: None,
    })))));
    assert_eq!(
        client
            .call::<_, u64>("executor.count_tokens", &json!({}), None)
            .await,
        Err(PluginClientError::InvalidResponse)
    );
}

#[test]
fn sanitization_removes_only_reserved_host_handles_recursively() {
    let sanitized = sanitize_plugin_value(json!({
        "safe": 1,
        "nested": {"__ctox_host_client": "opaque", "kept": true},
        "__ctox_host_authority": "opaque"
    }));
    assert_eq!(sanitized, json!({"safe": 1, "nested": {"kept": true}}));
}
