// ref: sdk/cliproxy/auth/home_unauthorized_refresh_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: ephemeral selection refresh is tested through the instance-owned registry
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::sdk::pluginapi::{
    ExecutorHttpRequest, ExecutorHttpResponse, ExecutorRequest, ExecutorResponse,
    ExecutorStreamChunk, ExecutorStreamResponse, PluginExecutionError, PluginFuture,
    ProviderExecutor,
};

use super::{
    Auth, AuthError, AuthRefresher, HomeRefreshError, ProviderExecutorRegistration,
    RefreshExecutorError, RefreshLeadRuntime,
};

struct FreshRefresher {
    calls: AtomicUsize,
    fail: bool,
}

impl AuthRefresher for FreshRefresher {
    fn refresh(&self, auth: &mut Auth) -> Result<Option<Auth>, RefreshExecutorError> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        if self.fail {
            return Err(RefreshExecutorError::Failed(super::AuthError {
                code: "refresh_temporarily_unavailable".into(),
                message: "temporary refresh failure".into(),
                retryable: true,
                http_status: 503,
            }));
        }
        auth.metadata
            .insert("access_token".into(), serde_json::json!("fresh-token"));
        Ok(None)
    }
}

struct ReplacementRefresher;

impl AuthRefresher for ReplacementRefresher {
    fn refresh(&self, _: &mut Auth) -> Result<Option<Auth>, RefreshExecutorError> {
        let mut replacement = Auth::default();
        replacement
            .metadata
            .insert("access_token".into(), serde_json::json!("fresh-token"));
        Ok(Some(replacement))
    }
}

struct RuntimeMarker;

impl RefreshLeadRuntime for RuntimeMarker {}

fn oauth(auth: &mut Auth, token: &str) {
    auth.metadata
        .insert("access_token".into(), serde_json::json!(token));
    auth.metadata
        .insert("refresh_token".into(), serde_json::json!("refresh-token"));
}

#[derive(Clone, Copy)]
enum UnauthorizedMode {
    Execute,
    Count,
    StreamBootstrap,
    StreamStarted,
}

struct UnauthorizedExecutor {
    mode: UnauthorizedMode,
    calls: AtomicUsize,
    refresh_calls: AtomicUsize,
}

impl UnauthorizedExecutor {
    fn new(mode: UnauthorizedMode) -> Arc<Self> {
        Arc::new(Self {
            mode,
            calls: AtomicUsize::new(0),
            refresh_calls: AtomicUsize::new(0),
        })
    }

    fn unauthorized() -> PluginExecutionError {
        Arc::new(AuthError {
            code: "unauthorized".into(),
            message: "expired access token".into(),
            http_status: 401,
            ..AuthError::default()
        })
    }

    fn token(request: &ExecutorRequest) -> &str {
        request
            .auth_metadata
            .get("access_token")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
    }
}

impl AuthRefresher for UnauthorizedExecutor {
    fn refresh(&self, auth: &mut Auth) -> Result<Option<Auth>, RefreshExecutorError> {
        self.refresh_calls.fetch_add(1, Ordering::AcqRel);
        auth.metadata
            .insert("access_token".into(), serde_json::json!("fresh-token"));
        Ok(None)
    }
}

impl ProviderExecutor for UnauthorizedExecutor {
    fn identifier(&self) -> &str {
        "codex"
    }

    fn execute<'a>(&'a self, request: ExecutorRequest) -> PluginFuture<'a, ExecutorResponse> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::AcqRel);
            if matches!(self.mode, UnauthorizedMode::Execute)
                && Self::token(&request) != "fresh-token"
            {
                return Err(Self::unauthorized());
            }
            Ok(ExecutorResponse {
                payload: b"ok".to_vec(),
                ..ExecutorResponse::default()
            })
        })
    }

    fn count_tokens<'a>(&'a self, request: ExecutorRequest) -> PluginFuture<'a, ExecutorResponse> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::AcqRel);
            if matches!(self.mode, UnauthorizedMode::Count)
                && Self::token(&request) != "fresh-token"
            {
                return Err(Self::unauthorized());
            }
            Ok(ExecutorResponse {
                payload: b"1".to_vec(),
                ..ExecutorResponse::default()
            })
        })
    }

    fn execute_stream<'a>(
        &'a self,
        request: ExecutorRequest,
    ) -> PluginFuture<'a, ExecutorStreamResponse> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::AcqRel);
            let stale = Self::token(&request) != "fresh-token";
            let (sender, receiver) = tokio::sync::mpsc::channel(2);
            match (self.mode, stale) {
                (UnauthorizedMode::StreamBootstrap, true) => sender
                    .send(ExecutorStreamChunk {
                        payload: Vec::new(),
                        error: Some(Self::unauthorized()),
                    })
                    .await
                    .unwrap(),
                (UnauthorizedMode::StreamStarted, true) => {
                    sender
                        .send(ExecutorStreamChunk {
                            payload: b"started".to_vec(),
                            error: None,
                        })
                        .await
                        .unwrap();
                    sender
                        .send(ExecutorStreamChunk {
                            payload: Vec::new(),
                            error: Some(Self::unauthorized()),
                        })
                        .await
                        .unwrap();
                }
                _ => sender
                    .send(ExecutorStreamChunk {
                        payload: b"ok".to_vec(),
                        error: None,
                    })
                    .await
                    .unwrap(),
            }
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

fn runtime_for_unauthorized(
    mode: UnauthorizedMode,
) -> (
    Arc<super::HomeAuthRuntime>,
    Arc<UnauthorizedExecutor>,
    Arc<super::home_execution_paths_test::TestHomeTransport>,
) {
    let transport = super::home_execution_paths_test::TestHomeTransport::with_auth_ids(&[]);
    transport.push_auth_with_metadata(
        "auth",
        serde_json::Map::from_iter([
            ("access_token".into(), serde_json::json!("stale-token")),
            ("refresh_token".into(), serde_json::json!("refresh-token")),
        ]),
    );
    let executor = UnauthorizedExecutor::new(mode);
    let manager = Arc::new(super::api_key_model_capabilities_test::manager());
    let refresher: Arc<dyn AuthRefresher> = executor.clone();
    let execution: Arc<dyn ProviderExecutor> = executor.clone();
    manager.register_executor(Arc::new(
        ProviderExecutorRegistration::new("codex", refresher)
            .unwrap()
            .with_execution(execution)
            .unwrap(),
    ));
    let client = Arc::new(crate::internal::home::Client::new(
        crate::internal::home::HomeConfig {
            enabled: true,
            ..Default::default()
        },
        transport.clone(),
    ));
    client.set_heartbeat(true);
    let runtime = Arc::new(super::HomeAuthRuntime::new(manager));
    runtime.publish_dispatch(
        client,
        Arc::new(crate::sdk::cliproxy::executionregistry::Registry::new()),
        1,
    );
    (runtime, executor, transport)
}

#[test]
fn unauthorized_refresh_updates_same_selection_and_is_attempted_once() {
    let transport = super::home_execution_paths_test::TestHomeTransport::with_auth_ids(&["auth"]);
    let executor = super::home_execution_paths_test::TestExecutor::failing(0);
    let (runtime, _) = super::home_execution_paths_test::runtime(transport, executor.clone());
    let refresher = Arc::new(FreshRefresher {
        calls: AtomicUsize::new(0),
        fail: false,
    });
    runtime.manager().register_executor(Arc::new(
        ProviderExecutorRegistration::new("codex", refresher.clone())
            .unwrap()
            .with_execution(executor)
            .unwrap(),
    ));
    let selection = runtime
        .pick_selection(super::conductor_home::HomeSelectionRequest {
            model: "gpt".into(),
            request_id: "request".into(),
            ..Default::default()
        })
        .unwrap();
    let mut failed = selection.clone_auth();
    oauth(&mut failed, "old-token");
    selection.replace_auth(failed.clone());

    let refreshed = runtime
        .refresh_home_selection_after_unauthorized(&selection, &failed)
        .unwrap()
        .unwrap();
    assert_eq!(refreshed.metadata["access_token"], "fresh-token");
    assert_eq!(
        selection.clone_auth().metadata["access_token"],
        "fresh-token"
    );
    let reused = runtime
        .refresh_home_selection_after_unauthorized(&selection, &failed)
        .unwrap()
        .unwrap();
    assert_eq!(reused.metadata["access_token"], "fresh-token");
    assert_eq!(refresher.calls.load(Ordering::Acquire), 1);
}

#[test]
fn transient_refresh_failure_is_returned_without_replacing_auth() {
    let transport = super::home_execution_paths_test::TestHomeTransport::with_auth_ids(&["auth"]);
    let executor = super::home_execution_paths_test::TestExecutor::failing(0);
    let (runtime, _) = super::home_execution_paths_test::runtime(transport, executor.clone());
    let refresher = Arc::new(FreshRefresher {
        calls: AtomicUsize::new(0),
        fail: true,
    });
    runtime.manager().register_executor(Arc::new(
        ProviderExecutorRegistration::new("codex", refresher)
            .unwrap()
            .with_execution(executor)
            .unwrap(),
    ));
    let selection = runtime
        .pick_selection(super::conductor_home::HomeSelectionRequest {
            model: "gpt".into(),
            request_id: "request".into(),
            ..Default::default()
        })
        .unwrap();
    let mut failed = selection.clone_auth();
    oauth(&mut failed, "old-token");
    selection.replace_auth(failed.clone());
    assert!(matches!(
        runtime.refresh_home_selection_after_unauthorized(&selection, &failed),
        Err(HomeRefreshError::Failed(_))
    ));
    assert_eq!(selection.clone_auth().metadata["access_token"], "old-token");
}

#[test]
fn replacement_refresh_preserves_nonserialized_runtime_authority() {
    let transport = super::home_execution_paths_test::TestHomeTransport::with_auth_ids(&["auth"]);
    let executor = super::home_execution_paths_test::TestExecutor::failing(0);
    let (runtime, _) = super::home_execution_paths_test::runtime(transport, executor.clone());
    runtime.manager().register_executor(Arc::new(
        ProviderExecutorRegistration::new("codex", Arc::new(ReplacementRefresher))
            .unwrap()
            .with_execution(executor)
            .unwrap(),
    ));
    let selection = runtime
        .pick_selection(super::conductor_home::HomeSelectionRequest {
            model: "gpt".into(),
            request_id: "request".into(),
            ..Default::default()
        })
        .unwrap();
    let mut failed = selection.clone_auth();
    oauth(&mut failed, "old-token");
    let marker: Arc<dyn RefreshLeadRuntime> = Arc::new(RuntimeMarker);
    failed.runtime = Some(marker.clone());
    selection.replace_auth(failed.clone());

    let refreshed = runtime
        .refresh_home_selection_after_unauthorized(&selection, &failed)
        .unwrap()
        .unwrap();
    assert!(refreshed
        .runtime
        .as_ref()
        .is_some_and(|runtime| Arc::ptr_eq(runtime, &marker)));
}

#[tokio::test]
async fn execute_and_count_refresh_same_home_selection_before_redispatch() {
    for (mode, count_tokens) in [
        (UnauthorizedMode::Execute, false),
        (UnauthorizedMode::Count, true),
    ] {
        let (runtime, executor, transport) = runtime_for_unauthorized(mode);
        let response = runtime
            .execute_home(
                super::home_execution_paths_test::request("gpt"),
                "",
                count_tokens,
            )
            .await
            .unwrap();
        assert!(!response.payload.is_empty());
        assert_eq!(executor.calls.load(Ordering::Acquire), 2);
        assert_eq!(executor.refresh_calls.load(Ordering::Acquire), 1);
        assert_eq!(
            transport.requests().len(),
            1,
            "must not redispatch Home auth"
        );
    }
}

#[tokio::test]
async fn stream_refreshes_only_before_first_payload() {
    let (runtime, executor, transport) =
        runtime_for_unauthorized(UnauthorizedMode::StreamBootstrap);
    let mut response = runtime
        .execute_home_stream(super::home_execution_paths_test::request("gpt"), "")
        .await
        .unwrap();
    assert_eq!(response.chunks.recv().await.unwrap().payload, b"ok");
    assert!(response.chunks.recv().await.is_none());
    assert_eq!(executor.calls.load(Ordering::Acquire), 2);
    assert_eq!(executor.refresh_calls.load(Ordering::Acquire), 1);
    assert_eq!(transport.requests().len(), 1);

    let (runtime, executor, _) = runtime_for_unauthorized(UnauthorizedMode::StreamStarted);
    let mut response = runtime
        .execute_home_stream(super::home_execution_paths_test::request("gpt"), "")
        .await
        .unwrap();
    assert_eq!(response.chunks.recv().await.unwrap().payload, b"started");
    assert_eq!(
        super::plugin_error_status(
            response
                .chunks
                .recv()
                .await
                .unwrap()
                .error
                .as_ref()
                .unwrap()
        ),
        401
    );
    assert_eq!(executor.calls.load(Ordering::Acquire), 1);
    assert_eq!(executor.refresh_calls.load(Ordering::Acquire), 0);
}
