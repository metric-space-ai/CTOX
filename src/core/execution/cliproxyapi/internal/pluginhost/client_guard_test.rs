// ref: internal/pluginhost/client_guard_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: exercises guarded process-client lifecycle semantics
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::value::to_raw_value;
use tokio::sync::{mpsc, oneshot};

use crate::sdk::pluginabi::Envelope;

use super::abi::{PluginCall, PluginClient, PluginClientError, PluginFuture, PluginStream};
use super::client_guard::GuardedPluginClient;

struct TestClient {
    release: Mutex<Option<oneshot::Receiver<()>>>,
    shutdowns: AtomicUsize,
}

impl PluginClient for TestClient {
    fn call<'a>(&'a self, _call: PluginCall) -> PluginFuture<'a, Envelope> {
        Box::pin(async move {
            let receiver = self
                .release
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            if let Some(receiver) = receiver {
                let _ = receiver.await;
            }
            Ok(Envelope::success(Some(to_raw_value(&"ok").unwrap())))
        })
    }

    fn call_stream<'a>(&'a self, _call: PluginCall) -> PluginFuture<'a, PluginStream> {
        Box::pin(async move {
            let (sender, receiver) = mpsc::channel(1);
            sender
                .send(Ok(to_raw_value(&"chunk").unwrap()))
                .await
                .unwrap();
            drop(sender);
            Ok(PluginStream { chunks: receiver })
        })
    }

    fn shutdown<'a>(&'a self) -> PluginFuture<'a, ()> {
        Box::pin(async move {
            self.shutdowns.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

fn call() -> PluginCall {
    PluginCall {
        method: "plugin.test".to_owned(),
        payload: to_raw_value(&()).unwrap(),
        deadline_unix_ms: None,
    }
}

#[tokio::test]
async fn shutdown_detaches_immediately_and_drains_active_call_once() {
    let (release, waiting) = oneshot::channel();
    let inner = Arc::new(TestClient {
        release: Mutex::new(Some(waiting)),
        shutdowns: AtomicUsize::new(0),
    });
    let guarded = Arc::new(GuardedPluginClient::new(inner.clone()));
    let caller = {
        let guarded = guarded.clone();
        tokio::spawn(async move { guarded.call(call()).await })
    };
    tokio::task::yield_now().await;
    let shutdown = {
        let guarded = guarded.clone();
        tokio::spawn(async move { guarded.shutdown().await })
    };
    tokio::task::yield_now().await;
    assert!(guarded.is_closed());
    assert!(matches!(
        guarded.call(call()).await,
        Err(PluginClientError::Closed)
    ));
    assert!(!shutdown.is_finished());
    release.send(()).unwrap();
    caller.await.unwrap().unwrap();
    shutdown.await.unwrap().unwrap();
    guarded.shutdown().await.unwrap();
    assert_eq!(inner.shutdowns.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn stream_lifetime_is_an_active_call_until_forwarding_finishes() {
    let inner = Arc::new(TestClient {
        release: Mutex::new(None),
        shutdowns: AtomicUsize::new(0),
    });
    let guarded = GuardedPluginClient::new(inner.clone());
    let mut stream = guarded.call_stream(call()).await.unwrap();
    assert_eq!(
        stream.chunks.recv().await.unwrap().unwrap().get(),
        "\"chunk\""
    );
    assert!(stream.chunks.recv().await.is_none());
    guarded.shutdown().await.unwrap();
    assert_eq!(inner.shutdowns.load(Ordering::SeqCst), 1);
}
