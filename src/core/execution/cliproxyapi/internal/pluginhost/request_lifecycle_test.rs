// ref: internal/pluginhost/request_lifecycle_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: ordered interceptor termination and detached lifecycle completion
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::Notify;

use crate::sdk::pluginapi::{
    PluginFuture, RequestCompletion, RequestCompletionOutcome, RequestInterceptRequest,
    RequestInterceptResponse, RequestInterceptor, RequestLifecyclePlugin,
};

use super::adapters_interceptors::{
    RequestCompletionDispatcher, RequestInterceptorChain, RequestInterceptorRecord,
    RequestLifecycleRecord,
};

struct Interceptor {
    calls: Arc<AtomicUsize>,
    terminate: bool,
}

impl RequestInterceptor for Interceptor {
    fn intercept_request_before_auth<'a>(
        &'a self,
        _request: RequestInterceptRequest,
    ) -> PluginFuture<'a, RequestInterceptResponse> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(RequestInterceptResponse {
                terminate: self.terminate,
                status_code: if self.terminate { 403 } else { 0 },
                response_body: if self.terminate {
                    br#"{"error":"blocked"}"#.to_vec()
                } else {
                    Vec::new()
                },
                ..RequestInterceptResponse::default()
            })
        })
    }

    fn intercept_request_after_auth<'a>(
        &'a self,
        request: RequestInterceptRequest,
    ) -> PluginFuture<'a, RequestInterceptResponse> {
        self.intercept_request_before_auth(request)
    }
}

#[tokio::test]
async fn terminating_high_priority_interceptor_stops_chain() {
    let high = Arc::new(AtomicUsize::new(0));
    let low = Arc::new(AtomicUsize::new(0));
    let chain = RequestInterceptorChain::new(vec![
        RequestInterceptorRecord {
            plugin_id: "low".to_owned(),
            priority: 10,
            interceptor: Arc::new(Interceptor {
                calls: low.clone(),
                terminate: false,
            }),
        },
        RequestInterceptorRecord {
            plugin_id: "high".to_owned(),
            priority: 20,
            interceptor: Arc::new(Interceptor {
                calls: high.clone(),
                terminate: true,
            }),
        },
    ]);
    let response = chain
        .before_auth(RequestInterceptRequest {
            request_id: "request-1".to_owned(),
            ..RequestInterceptRequest::default()
        })
        .await;
    assert!(response.terminate);
    assert_eq!(response.status_code, 403);
    assert_eq!(response.response_body, br#"{"error":"blocked"}"#);
    assert_eq!(high.load(Ordering::Relaxed), 1);
    assert_eq!(low.load(Ordering::Relaxed), 0);
}

struct Lifecycle {
    seen: Arc<Mutex<Vec<RequestCompletion>>>,
    started: Arc<Notify>,
    release: Arc<Notify>,
}

impl RequestLifecyclePlugin for Lifecycle {
    fn handle_request_complete<'a>(
        &'a self,
        completion: RequestCompletion,
    ) -> PluginFuture<'a, ()> {
        Box::pin(async move {
            self.seen
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(completion);
            self.started.notify_one();
            self.release.notified().await;
            Ok(())
        })
    }
}

#[tokio::test]
async fn completion_is_detached_and_owns_cloned_metadata() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let dispatcher = RequestCompletionDispatcher::new(vec![RequestLifecycleRecord {
        plugin_id: "lifecycle".to_owned(),
        priority: 1,
        lifecycle: Arc::new(Lifecycle {
            seen: seen.clone(),
            started: started.clone(),
            release: release.clone(),
        }),
    }]);
    let mut completion = RequestCompletion {
        request_id: "request-1".to_owned(),
        outcome: RequestCompletionOutcome(RequestCompletionOutcome::CANCELED.to_owned()),
        metadata: [(
            "nested".to_owned(),
            serde_json::json!({"value": "original"}),
        )]
        .into(),
        ..RequestCompletion::default()
    };
    dispatcher.complete(completion.clone());
    completion
        .metadata
        .get_mut("nested")
        .expect("nested metadata")["value"] = serde_json::json!("mutated");
    tokio::time::timeout(Duration::from_secs(1), started.notified())
        .await
        .unwrap();
    assert_eq!(
        seen.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[0]
            .metadata["nested"]["value"],
        "original"
    );
    release.notify_one();
}
