// ref: sdk/cliproxy/auth/conductor_executor_replace_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use super::{
    Auth, AuthRefresher, AuthRefresherResolver, ExecutionSessionCloser, ProviderDispatchError,
    ProviderExecutorRegistration, ProviderExecutorRegistrationError, ProviderExecutorRegistry,
    RefreshExecutorError, CLOSE_ALL_EXECUTION_SESSIONS_ID,
};
use crate::sdk::pluginapi::{
    ExecutorHttpRequest, ExecutorHttpResponse, ExecutorRequest, ExecutorResponse,
    ExecutorStreamChunk, ExecutorStreamResponse, PluginFuture, ProviderExecutor,
};

#[derive(Default)]
struct ReplaceAwareExecutor {
    closed_session_ids: Mutex<Vec<String>>,
}

impl AuthRefresher for ReplaceAwareExecutor {
    fn refresh(&self, _auth: &mut Auth) -> Result<Option<Auth>, RefreshExecutorError> {
        Ok(None)
    }
}

impl ExecutionSessionCloser for ReplaceAwareExecutor {
    fn close_execution_session(&self, session_id: &str) {
        self.closed_session_ids
            .lock()
            .expect("closed session ids")
            .push(session_id.to_owned());
    }
}

impl ReplaceAwareExecutor {
    fn closed(&self) -> Vec<String> {
        self.closed_session_ids
            .lock()
            .expect("closed session ids")
            .clone()
    }
}

struct AsyncExecutor(&'static str);

impl ProviderExecutor for AsyncExecutor {
    fn identifier(&self) -> &str {
        self.0
    }

    fn execute<'a>(&'a self, request: ExecutorRequest) -> PluginFuture<'a, ExecutorResponse> {
        Box::pin(async move {
            Ok(ExecutorResponse {
                payload: request.payload,
                ..ExecutorResponse::default()
            })
        })
    }

    fn execute_stream<'a>(
        &'a self,
        request: ExecutorRequest,
    ) -> PluginFuture<'a, ExecutorStreamResponse> {
        Box::pin(async move {
            let (sender, receiver) = tokio::sync::mpsc::channel(1);
            sender
                .send(ExecutorStreamChunk {
                    payload: request.payload,
                    error: None,
                })
                .await
                .expect("stream receiver");
            Ok(ExecutorStreamResponse {
                headers: Default::default(),
                chunks: receiver,
            })
        })
    }

    fn count_tokens<'a>(&'a self, request: ExecutorRequest) -> PluginFuture<'a, ExecutorResponse> {
        Box::pin(async move {
            Ok(ExecutorResponse {
                payload: (request.payload.len() as u64).to_string().into_bytes(),
                ..ExecutorResponse::default()
            })
        })
    }

    fn http_request<'a>(
        &'a self,
        request: ExecutorHttpRequest,
    ) -> PluginFuture<'a, ExecutorHttpResponse> {
        Box::pin(async move {
            Ok(ExecutorHttpResponse {
                status_code: 202,
                body: request.body,
                ..ExecutorHttpResponse::default()
            })
        })
    }
}

fn registration(
    provider: &str,
    executor: Arc<ReplaceAwareExecutor>,
) -> Arc<ProviderExecutorRegistration> {
    let refresher: Arc<dyn AuthRefresher> = executor.clone();
    let closer: Arc<dyn ExecutionSessionCloser> = executor;
    Arc::new(
        ProviderExecutorRegistration::new(provider, refresher)
            .expect("valid provider")
            .with_session_closer(closer),
    )
}

#[test]
fn register_executor_closes_replaced_execution_sessions() {
    let registry = ProviderExecutorRegistry::default();
    let replaced = Arc::new(ReplaceAwareExecutor::default());
    let current = Arc::new(ReplaceAwareExecutor::default());

    assert!(!registry.register(registration("codex", replaced.clone())));
    assert!(registry.register(registration("codex", current.clone())));

    assert_eq!(
        replaced.closed(),
        vec![CLOSE_ALL_EXECUTION_SESSIONS_ID.to_owned()]
    );
    assert!(current.closed().is_empty());
}

#[test]
fn executor_lookup_and_refresh_resolution_are_case_insensitive() {
    let registry = ProviderExecutorRegistry::default();
    let executor = Arc::new(ReplaceAwareExecutor::default());
    registry.register(registration("  CoDeX  ", executor));

    assert_eq!(registry.get("CODEX").unwrap().provider(), "codex");
    assert!(AuthRefresherResolver::resolve(&registry, " codeX ").is_some());
    assert!(registry.get("unknown").is_none());
}

#[test]
fn reregistering_same_registration_is_idempotent() {
    let registry = ProviderExecutorRegistry::default();
    let executor = Arc::new(ReplaceAwareExecutor::default());
    let registered = registration("codex", executor.clone());

    registry.register(registered.clone());
    assert!(!registry.register(registered));
    assert!(executor.closed().is_empty());
}

#[test]
fn separately_wrapped_same_capability_arcs_are_idempotent() {
    let registry = ProviderExecutorRegistry::default();
    let executor = Arc::new(ReplaceAwareExecutor::default());

    registry.register(registration("codex", executor.clone()));
    assert!(!registry.register(registration("CODEX", executor.clone())));
    assert!(executor.closed().is_empty());
}

#[test]
fn unregister_does_not_close_externally_owned_executor() {
    let registry = ProviderExecutorRegistry::default();
    let executor = Arc::new(ReplaceAwareExecutor::default());
    registry.register(registration("codex", executor.clone()));

    let removed = registry.unregister(" CODEX ");
    assert!(removed.is_some());
    assert!(registry.is_empty());
    assert!(executor.closed().is_empty());
}

#[test]
fn close_all_sessions_uses_typed_optional_capability() {
    let registry = ProviderExecutorRegistry::default();
    let executor = Arc::new(ReplaceAwareExecutor::default());
    registry.register(registration("codex", executor.clone()));

    assert!(registry.close_all_sessions("CODEX"));
    assert!(!registry.close_all_sessions("missing"));
    assert_eq!(
        executor.closed(),
        vec![CLOSE_ALL_EXECUTION_SESSIONS_ID.to_owned()]
    );
}

#[test]
fn registration_rejects_empty_provider_keys() {
    let refresher: Arc<dyn AuthRefresher> = Arc::new(ReplaceAwareExecutor::default());
    assert!(ProviderExecutorRegistration::new(" \t ", refresher).is_none());
}

#[tokio::test]
async fn registry_dispatches_every_async_execution_capability() {
    let registry = ProviderExecutorRegistry::default();
    let refresher: Arc<dyn AuthRefresher> = Arc::new(ReplaceAwareExecutor::default());
    let execution: Arc<dyn ProviderExecutor> = Arc::new(AsyncExecutor("codex"));
    registry.register(Arc::new(
        ProviderExecutorRegistration::new("CODEX", refresher)
            .expect("provider")
            .with_execution(execution)
            .expect("matching execution"),
    ));

    let request = || ExecutorRequest {
        payload: b"payload".to_vec(),
        ..ExecutorRequest::default()
    };
    assert_eq!(
        registry
            .execute("codex", request())
            .await
            .expect("execute")
            .payload,
        b"payload"
    );
    let mut stream = registry
        .execute_stream("CODEX", request())
        .await
        .expect("stream");
    assert_eq!(
        stream.chunks.recv().await.expect("chunk").payload,
        b"payload"
    );
    assert_eq!(
        registry
            .count_tokens("codex", request())
            .await
            .expect("count")
            .payload,
        b"7"
    );
    let response = registry
        .http_request(
            "codex",
            ExecutorHttpRequest {
                body: b"body".to_vec(),
                ..ExecutorHttpRequest::default()
            },
        )
        .await
        .expect("http");
    assert_eq!(response.status_code, 202);
    assert_eq!(response.body, b"body");
}

#[tokio::test]
async fn registry_rejects_mismatched_or_missing_execution_capabilities() {
    let refresher: Arc<dyn AuthRefresher> = Arc::new(ReplaceAwareExecutor::default());
    let mismatched: Arc<dyn ProviderExecutor> = Arc::new(AsyncExecutor("claude"));
    assert!(matches!(
        ProviderExecutorRegistration::new("codex", refresher.clone())
            .expect("provider")
            .with_execution(mismatched),
        Err(ProviderExecutorRegistrationError::ProviderMismatch)
    ));

    let registry = ProviderExecutorRegistry::default();
    registry.register(Arc::new(
        ProviderExecutorRegistration::new("codex", refresher).expect("provider"),
    ));
    assert!(matches!(
        registry.execute("codex", ExecutorRequest::default()).await,
        Err(ProviderDispatchError::ExecutionUnavailable)
    ));
    assert!(matches!(
        registry
            .execute("missing", ExecutorRequest::default())
            .await,
        Err(ProviderDispatchError::ProviderNotRegistered)
    ));
}

#[test]
fn changed_execution_capability_closes_sessions_even_with_same_refresh_owner() {
    let registry = ProviderExecutorRegistry::default();
    let owner = Arc::new(ReplaceAwareExecutor::default());
    let build = |execution: Arc<dyn ProviderExecutor>| {
        let refresher: Arc<dyn AuthRefresher> = owner.clone();
        let closer: Arc<dyn ExecutionSessionCloser> = owner.clone();
        Arc::new(
            ProviderExecutorRegistration::new("codex", refresher)
                .expect("provider")
                .with_execution(execution)
                .expect("execution")
                .with_session_closer(closer),
        )
    };

    registry.register(build(Arc::new(AsyncExecutor("codex"))));
    assert!(registry.register(build(Arc::new(AsyncExecutor("codex")))));
    assert_eq!(
        owner.closed(),
        vec![CLOSE_ALL_EXECUTION_SESSIONS_ID.to_owned()]
    );
}
