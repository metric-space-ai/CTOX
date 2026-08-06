// ref: internal/runtime/executor/codex_websockets_executor_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;

use super::codex_executor_auth::CodexSubscriptionAuth;
use super::codex_executor_reasoning::CodexReasoningReplayCache;
use super::codex_websockets_connection::{
    CodexWebsocketConnection, CodexWebsocketFrame, CodexWebsocketTransport,
};
use super::codex_websockets_errors::CodexWebsocketError;
use super::codex_websockets_executor::{CodexWebsocketExecutionRequest, CodexWebsocketsExecutor};
use super::codex_websockets_session::CodexWebsocketSessionStore;
use crate::internal::auth::codex::{
    CodexCredentialHandles, CodexRefreshCoordinator, CodexRefreshHttpResponse, CodexRefreshRequest,
    CodexRefreshTransport, CodexRefreshTransportFailure, CodexSecretHandle, CodexSecretKind,
    CodexSecretStore, CodexStoredCredentials, RefreshClock, SecretStoreError, SecretString,
};

struct Store(CodexStoredCredentials);
impl CodexSecretStore for Store {
    fn load_credentials(
        &self,
        _: &CodexCredentialHandles,
    ) -> Result<CodexStoredCredentials, SecretStoreError> {
        Ok(self.0.clone())
    }
    fn store_credentials(
        &self,
        _: &CodexCredentialHandles,
        _: &CodexStoredCredentials,
    ) -> Result<(), SecretStoreError> {
        Ok(())
    }
}

struct Clock;
impl RefreshClock for Clock {
    fn now(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }
    fn sleep(
        &self,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), CodexRefreshTransportFailure>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }
}

struct Refresh;
impl CodexRefreshTransport for Refresh {
    fn execute<'a>(
        &'a self,
        _: &'a CodexRefreshRequest,
        _: Duration,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<CodexRefreshHttpResponse, CodexRefreshTransportFailure>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async { Err(CodexRefreshTransportFailure::Protocol) })
    }
}

struct Connection {
    frames: VecDeque<Result<CodexWebsocketFrame, CodexWebsocketError>>,
}
impl CodexWebsocketConnection for Connection {
    fn send<'a>(
        &'a mut self,
        _: CodexWebsocketFrame,
    ) -> Pin<Box<dyn Future<Output = Result<(), CodexWebsocketError>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
    fn receive<'a>(
        &'a mut self,
    ) -> Pin<Box<dyn Future<Output = Result<CodexWebsocketFrame, CodexWebsocketError>> + Send + 'a>>
    {
        Box::pin(async move {
            self.frames
                .pop_front()
                .unwrap_or_else(|| Err(CodexWebsocketError::protocol("closed", true)))
        })
    }
    fn close<'a>(
        &'a mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), CodexWebsocketError>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

struct Transport {
    connections: Mutex<VecDeque<VecDeque<Result<CodexWebsocketFrame, CodexWebsocketError>>>>,
    connects: AtomicUsize,
}
impl CodexWebsocketTransport for Transport {
    fn connect<'a>(
        &'a self,
        _: &'a str,
        _: &'a super::codex_websockets_request::CodexWebsocketHeaders,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Box<dyn CodexWebsocketConnection>, CodexWebsocketError>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            self.connects.fetch_add(1, Ordering::SeqCst);
            let frames = self
                .connections
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| CodexWebsocketError::protocol("no_connection", false))?;
            Ok(Box::new(Connection { frames }) as Box<dyn CodexWebsocketConnection>)
        })
    }
}

fn auth() -> Arc<CodexSubscriptionAuth> {
    let claims = serde_json::json!({"https://api.openai.com/auth":{"chatgpt_account_id":"acct"}});
    let jwt = format!(
        "h.{}.s",
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap())
    );
    let handle = |name, kind| CodexSecretHandle::new("subscriptions", name, kind).unwrap();
    let handles = CodexCredentialHandles::new(
        handle("id", CodexSecretKind::IdToken),
        handle("access", CodexSecretKind::AccessToken),
        handle("refresh", CodexSecretKind::RefreshToken),
    )
    .unwrap();
    Arc::new(CodexSubscriptionAuth::new(
        handles,
        Arc::new(Store(CodexStoredCredentials::new(
            SecretString::new(jwt).unwrap(),
            SecretString::new("access").unwrap(),
            SecretString::new("refresh").unwrap(),
        ))),
        Arc::new(Refresh),
        Arc::new(Clock),
        Arc::new(CodexRefreshCoordinator::default()),
    ))
}

fn request() -> CodexWebsocketExecutionRequest {
    CodexWebsocketExecutionRequest {
        auth_id: "auth-a".to_owned(),
        session_id: "session-a".to_owned(),
        responses_url: "https://chatgpt.com/backend-api/codex/responses".to_owned(),
        body: br#"{"model":"gpt","input":[]}"#.to_vec(),
        headers: Default::default(),
        execution_lifecycle: None,
    }
}

#[tokio::test]
async fn reconnects_once_before_commit_and_never_after_commit() {
    let before = VecDeque::from([Err(CodexWebsocketError::protocol("disconnect", true))]);
    let completed = VecDeque::from([
        Ok(CodexWebsocketFrame::Text(
            br#"{"type":"response.created","response":{"id":"r"}}"#.to_vec(),
        )),
        Ok(CodexWebsocketFrame::Text(
            br#"{"type":"response.completed","response":{"id":"r","output":[]}}"#.to_vec(),
        )),
    ]);
    let transport = Arc::new(Transport {
        connections: Mutex::new(VecDeque::from([before, completed])),
        connects: AtomicUsize::new(0),
    });
    let executor = CodexWebsocketsExecutor::new(
        auth(),
        transport.clone(),
        Arc::new(CodexWebsocketSessionStore::default()),
        Arc::new(CodexReasoningReplayCache::default()),
    );
    let result = executor.execute(request()).await.unwrap();
    assert_eq!(result.reconnects, 1);
    assert!(result.committed);
    assert_eq!(transport.connects.load(Ordering::SeqCst), 2);

    executor.close_execution_session("session-a").await;
    let post_commit = VecDeque::from([
        Ok(CodexWebsocketFrame::Text(
            br#"{"type":"response.created","response":{"id":"r2"}}"#.to_vec(),
        )),
        Err(CodexWebsocketError::protocol(
            "disconnect-after-commit",
            true,
        )),
    ]);
    *transport.connections.lock().unwrap() = VecDeque::from([post_commit]);
    let error = executor.execute(request()).await.unwrap_err();
    assert_eq!(error.code.as_deref(), Some("disconnect-after-commit"));
    assert_eq!(transport.connects.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn streamed_execution_releases_events_through_bounded_live_channel() {
    let frames = VecDeque::from([
        Ok(CodexWebsocketFrame::Text(
            br#"{"type":"response.created","response":{"id":"live"}}"#.to_vec(),
        )),
        Ok(CodexWebsocketFrame::Text(
            br#"{"type":"response.completed","response":{"id":"live","output":[]}}"#.to_vec(),
        )),
    ]);
    let transport = Arc::new(Transport {
        connections: Mutex::new(VecDeque::from([frames])),
        connects: AtomicUsize::new(0),
    });
    let executor = CodexWebsocketsExecutor::new(
        auth(),
        transport,
        Arc::new(CodexWebsocketSessionStore::default()),
        Arc::new(CodexReasoningReplayCache::default()),
    );
    let mut stream = executor.execute_streamed(request(), 1).await.unwrap();
    let first = stream.next_chunk().await.unwrap().unwrap();
    assert!(String::from_utf8(first)
        .unwrap()
        .contains("response.created"));
    assert!(stream.committed());
    let second = stream.next_chunk().await.unwrap().unwrap();
    assert!(String::from_utf8(second)
        .unwrap()
        .contains("response.completed"));
    assert_eq!(
        stream.next_chunk().await.unwrap().unwrap(),
        b"data: [DONE]\n\n"
    );
}
