// Origin: CTOX
// License: AGPL-3.0-only

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};

use super::auth::{
    AuthError, AuthMutationOptions, AuthRefresher, CooldownStateRecord, CooldownStateStore,
    CooldownStoreError, GenericExecutionError, ProviderExecutorRegistration, RefreshExecutorError,
};
use super::service_test_support::{auth, runtime_fixture};
use crate::sdk::pluginapi::{
    ExecutorHttpRequest, ExecutorHttpResponse, ExecutorRequest, ExecutorResponse,
    ExecutorStreamChunk, ExecutorStreamResponse, PluginExecutionError, PluginFuture,
    ProviderExecutor,
};

#[derive(Default)]
struct MemoryCooldownStore(Mutex<Vec<CooldownStateRecord>>);

impl CooldownStateStore for MemoryCooldownStore {
    fn load(&self) -> Result<Vec<CooldownStateRecord>, CooldownStoreError> {
        Ok(self.0.lock().unwrap().clone())
    }

    fn save(&self, records: &[CooldownStateRecord]) -> Result<(), CooldownStoreError> {
        *self.0.lock().unwrap() = records.to_vec();
        Ok(())
    }
}

#[derive(Default)]
struct RouteExecutor {
    unary_calls: AtomicUsize,
    count_calls: AtomicUsize,
    stream_calls: AtomicUsize,
    refreshes: AtomicUsize,
}

impl RouteExecutor {
    fn stale(request: &ExecutorRequest) -> bool {
        request
            .auth_metadata
            .get("access_token")
            .and_then(serde_json::Value::as_str)
            != Some("fresh")
    }

    fn unauthorized() -> PluginExecutionError {
        Arc::new(AuthError {
            code: "unauthorized".into(),
            message: "expired test credential".into(),
            retryable: false,
            http_status: 401,
        })
    }
}

impl AuthRefresher for RouteExecutor {
    fn refresh(
        &self,
        auth: &mut super::auth::Auth,
    ) -> Result<Option<super::auth::Auth>, RefreshExecutorError> {
        self.refreshes.fetch_add(1, Ordering::AcqRel);
        auth.metadata
            .insert("access_token".into(), serde_json::json!("fresh"));
        Ok(None)
    }
}

impl ProviderExecutor for RouteExecutor {
    fn identifier(&self) -> &str {
        "claude"
    }

    fn execute<'a>(&'a self, request: ExecutorRequest) -> PluginFuture<'a, ExecutorResponse> {
        Box::pin(async move {
            self.unary_calls.fetch_add(1, Ordering::AcqRel);
            if Self::stale(&request) {
                return Err(Self::unauthorized());
            }
            Ok(ExecutorResponse {
                payload: b"unary".to_vec(),
                ..ExecutorResponse::default()
            })
        })
    }

    fn count_tokens<'a>(&'a self, request: ExecutorRequest) -> PluginFuture<'a, ExecutorResponse> {
        Box::pin(async move {
            self.count_calls.fetch_add(1, Ordering::AcqRel);
            assert!(!Self::stale(&request));
            Ok(ExecutorResponse {
                payload: b"count".to_vec(),
                ..ExecutorResponse::default()
            })
        })
    }

    fn execute_stream<'a>(
        &'a self,
        request: ExecutorRequest,
    ) -> PluginFuture<'a, ExecutorStreamResponse> {
        Box::pin(async move {
            self.stream_calls.fetch_add(1, Ordering::AcqRel);
            assert!(!Self::stale(&request));
            let (sender, receiver) = tokio::sync::mpsc::channel(1);
            sender
                .send(ExecutorStreamChunk {
                    payload: b"stream".to_vec(),
                    error: None,
                })
                .await
                .unwrap();
            drop(sender);
            Ok(ExecutorStreamResponse {
                headers: Default::default(),
                chunks: receiver,
            })
        })
    }

    fn http_request<'a>(
        &'a self,
        request: ExecutorHttpRequest,
    ) -> PluginFuture<'a, ExecutorHttpResponse> {
        Box::pin(async move {
            Ok(ExecutorHttpResponse {
                status_code: 200,
                body: request.body,
                ..ExecutorHttpResponse::default()
            })
        })
    }
}

fn request() -> ExecutorRequest {
    ExecutorRequest {
        model: "claude-model".into(),
        ..ExecutorRequest::default()
    }
}

#[tokio::test]
async fn service_route_facade_owns_unary_count_stream_and_401_replay() {
    let cooldown: Arc<dyn CooldownStateStore> = Arc::new(MemoryCooldownStore::default());
    let fixture = runtime_fixture(Some(cooldown));
    let executor = Arc::new(RouteExecutor::default());
    let manager = fixture.runtime.auth_manager();
    manager.register_executor(Arc::new(
        ProviderExecutorRegistration::new("claude", executor.clone())
            .unwrap()
            .with_execution(executor.clone())
            .unwrap(),
    ));
    let mut credential = auth("account-a", "claude");
    credential
        .metadata
        .insert("access_token".into(), serde_json::json!("stale"));
    credential
        .metadata
        .insert("refresh_token".into(), serde_json::json!("refresh"));
    manager
        .register(
            credential,
            AuthMutationOptions::default(),
            "2026-08-04T12:00:00Z"
                .parse::<DateTime<Utc>>()
                .expect("fixed test time"),
        )
        .unwrap();

    let providers = ["claude".to_owned()];
    let unary = fixture
        .runtime
        .execute_provider_route(&providers, request())
        .await
        .unwrap();
    assert_eq!(unary.payload, b"unary");
    let count = fixture
        .runtime
        .count_tokens_provider_route(&providers, request())
        .await
        .unwrap();
    assert_eq!(count.payload, b"count");
    let mut stream = fixture
        .runtime
        .execute_stream_provider_route(&providers, request())
        .await
        .unwrap();
    assert_eq!(stream.chunks.recv().await.unwrap().payload, b"stream");
    assert!(stream.chunks.recv().await.is_none());

    assert_eq!(executor.unary_calls.load(Ordering::Acquire), 2);
    assert_eq!(executor.count_calls.load(Ordering::Acquire), 1);
    assert_eq!(executor.stream_calls.load(Ordering::Acquire), 1);
    assert_eq!(executor.refreshes.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn service_route_facade_fails_closed_without_persisted_conductor() {
    let fixture = runtime_fixture(None);
    assert!(matches!(
        fixture
            .runtime
            .execute_provider_route(&["claude".into()], request())
            .await,
        Err(GenericExecutionError::ConductorUnavailable)
    ));
}
