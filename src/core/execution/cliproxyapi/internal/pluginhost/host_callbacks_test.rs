// ref: internal/pluginhost/host_callbacks_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: all callback methods are identity-bound to injected authorities
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use serde_json::{value::to_raw_value, value::RawValue, Value};

use crate::sdk::pluginabi::{Envelope, METHOD_HOST_HTTP_DO, METHOD_HOST_LOG};
use crate::sdk::pluginapi::{
    HostHttpClient, HttpRequest, HttpResponse, HttpStreamChunk, HttpStreamResponse, PluginFuture,
};
use tokio::sync::mpsc;

use super::abi::PluginClientError;
use super::callback_contexts::{CallbackAuthority, CallbackContextRegistry};
use super::host_callbacks::{
    install_standard_host_callbacks, HostCallbackFuture, HostCallbackHandler,
    HostCallbackRouteError, HostCallbackRouter, HostLogAuthority,
};
use super::http_bridge::HostHttpBridge;
use super::http_stream_bridge::{HttpStreamBridge, HttpStreamBridgeError};
use super::stream_bridge::StreamBridge;

struct Echo;

impl HostCallbackHandler for Echo {
    fn call<'a>(
        &'a self,
        _authority: &'a CallbackAuthority,
        payload: &'a RawValue,
    ) -> HostCallbackFuture<'a> {
        Box::pin(async move {
            Ok(Envelope::success(Some(
                serde_json::value::RawValue::from_string(payload.get().to_owned())
                    .map_err(|_| PluginClientError::InvalidRequest)?,
            )))
        })
    }
}

#[tokio::test]
async fn callback_context_binds_plugin_deadline_and_cleanup() {
    let contexts = CallbackContextRegistry::new();
    let authority = CallbackAuthority::new("plugin-a", Some(200));
    let mut lease = contexts.open(authority.clone());
    let callback_id = lease.id().to_owned();
    let mut router = HostCallbackRouter::new(contexts.clone());
    router.register("host.log", Arc::new(Echo)).unwrap();

    let payload = to_raw_value(&serde_json::json!({"message": "hello"})).unwrap();
    let envelope = router
        .dispatch("plugin-a", &callback_id, "host.log", &payload, 100)
        .await
        .unwrap();
    assert!(envelope.ok);
    assert_eq!(envelope.result.unwrap().get(), payload.get());
    assert_eq!(
        router
            .dispatch("plugin-b", &callback_id, "host.log", &payload, 100)
            .await
            .err()
            .unwrap(),
        HostCallbackRouteError::WrongPlugin
    );
    assert_eq!(
        router
            .dispatch("plugin-a", &callback_id, "host.log", &payload, 200)
            .await
            .err()
            .unwrap(),
        HostCallbackRouteError::DeadlineExceeded
    );

    lease.close();
    assert!(authority.is_cancelled());
    assert!(contexts.is_empty());
    assert_eq!(
        router
            .dispatch("plugin-a", &callback_id, "host.log", &payload, 100)
            .await
            .err()
            .unwrap(),
        HostCallbackRouteError::UnknownContext
    );
}

#[test]
fn duplicate_and_non_host_methods_fail_closed() {
    let contexts = CallbackContextRegistry::new();
    let mut router = HostCallbackRouter::new(contexts);
    assert_eq!(
        router.register("executor.execute", Arc::new(Echo)),
        Err(HostCallbackRouteError::InvalidMethod)
    );
    router.register("host.log", Arc::new(Echo)).unwrap();
    assert_eq!(
        router.register("host.log", Arc::new(Echo)),
        Err(HostCallbackRouteError::DuplicateMethod)
    );
}

struct HttpAuthority {
    seen: Mutex<Vec<HttpRequest>>,
}

impl HostHttpClient for HttpAuthority {
    fn execute<'a>(&'a self, request: HttpRequest) -> PluginFuture<'a, HttpResponse> {
        Box::pin(async move {
            self.seen
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(request);
            Ok(HttpResponse {
                status_code: 204,
                ..HttpResponse::default()
            })
        })
    }

    fn execute_stream<'a>(&'a self, request: HttpRequest) -> PluginFuture<'a, HttpStreamResponse> {
        Box::pin(async move {
            self.seen
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(request);
            let (sender, receiver) = mpsc::channel(2);
            sender
                .send(HttpStreamChunk {
                    payload: b"chunk".to_vec(),
                    error: None,
                })
                .await
                .unwrap();
            drop(sender);
            Ok(HttpStreamResponse {
                status_code: 200,
                headers: Default::default(),
                chunks: receiver,
            })
        })
    }
}

#[tokio::test]
async fn http_bridge_uses_injected_authority_and_rejects_credential_urls() {
    let authority = Arc::new(HttpAuthority {
        seen: Mutex::new(Vec::new()),
    });
    let bridge = HostHttpBridge::new(authority.clone());
    let response = bridge
        .execute(HttpRequest {
            method: "POST".to_owned(),
            url: "https://provider.example/v1".to_owned(),
            body: b"request".to_vec(),
            ..HttpRequest::default()
        })
        .await
        .unwrap();
    assert_eq!(response.status_code, 204);
    assert_eq!(
        authority
            .seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len(),
        1
    );
    assert!(bridge
        .execute(HttpRequest {
            method: "GET".to_owned(),
            url: "https://secret@provider.example/v1".to_owned(),
            ..HttpRequest::default()
        })
        .await
        .is_err());
}

#[tokio::test]
async fn http_stream_handles_are_terminal_and_explicitly_closeable() {
    let authority = Arc::new(HttpAuthority {
        seen: Mutex::new(Vec::new()),
    });
    let response = HostHttpBridge::new(authority)
        .execute_stream(HttpRequest {
            method: "GET".to_owned(),
            url: "https://provider.example/stream".to_owned(),
            ..HttpRequest::default()
        })
        .await
        .unwrap();
    let streams = HttpStreamBridge::default();
    let handle = streams.open("plugin-a", response);
    assert_eq!(
        streams
            .read("plugin-a", &handle.stream_id)
            .await
            .unwrap()
            .payload,
        b"chunk"
    );
    assert!(
        streams
            .read("plugin-a", &handle.stream_id)
            .await
            .unwrap()
            .done
    );
    assert_eq!(
        streams.read("plugin-a", &handle.stream_id).await,
        Err(HttpStreamBridgeError::NotOpen)
    );
}

#[derive(Default)]
struct LogAuthority {
    calls: Mutex<Vec<(String, String, String)>>,
}

impl HostLogAuthority for LogAuthority {
    fn log<'a>(
        &'a self,
        caller_plugin_id: &'a str,
        level: &'a str,
        message: &'a str,
        _fields: std::collections::BTreeMap<String, Value>,
    ) -> PluginFuture<'a, ()> {
        Box::pin(async move {
            self.calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((
                    caller_plugin_id.to_owned(),
                    level.to_owned(),
                    message.to_owned(),
                ));
            Ok(())
        })
    }
}

#[tokio::test]
async fn installed_http_and_log_callbacks_keep_wire_shape_and_identity() {
    let contexts = CallbackContextRegistry::new();
    let lease = contexts.open(CallbackAuthority::new("plugin-a", None));
    let mut router = HostCallbackRouter::new(contexts);
    let http = Arc::new(HttpAuthority {
        seen: Mutex::new(Vec::new()),
    });
    let log = Arc::new(LogAuthority::default());
    install_standard_host_callbacks(
        &mut router,
        HostHttpBridge::new(http.clone()),
        Arc::new(HttpStreamBridge::default()),
        StreamBridge::new(),
        log.clone(),
    )
    .unwrap();

    let request = to_raw_value(&serde_json::json!({
        "method": "POST",
        "url": "https://provider.example/v1",
        "body": "cmVxdWVzdA=="
    }))
    .unwrap();
    let response = router
        .dispatch("plugin-a", lease.id(), METHOD_HOST_HTTP_DO, &request, 0)
        .await
        .unwrap();
    let response: Value = serde_json::from_str(response.result.unwrap().get()).unwrap();
    assert_eq!(response["status_code"], 204);
    assert_eq!(http.seen.lock().unwrap()[0].body, b"request");

    let request = to_raw_value(&serde_json::json!({
        "level": "warning",
        "message": "bounded",
        "fields": {"attempt": 1}
    }))
    .unwrap();
    router
        .dispatch("plugin-a", lease.id(), METHOD_HOST_LOG, &request, 0)
        .await
        .unwrap();
    assert_eq!(
        log.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_slice(),
        &[(
            "plugin-a".to_owned(),
            "warn".to_owned(),
            "bounded".to_owned()
        )]
    );
}

#[tokio::test]
async fn http_stream_handle_rejects_another_plugin_owner() {
    let authority = Arc::new(HttpAuthority {
        seen: Mutex::new(Vec::new()),
    });
    let response = HostHttpBridge::new(authority)
        .execute_stream(HttpRequest {
            method: "GET".to_owned(),
            url: "https://provider.example/stream".to_owned(),
            ..HttpRequest::default()
        })
        .await
        .unwrap();
    let streams = HttpStreamBridge::default();
    let handle = streams.open("plugin-a", response);
    assert_eq!(
        streams.read("plugin-b", &handle.stream_id).await,
        Err(HttpStreamBridgeError::WrongOwner)
    );
    assert!(streams.close("plugin-a", &handle.stream_id).unwrap());
}
