// ref: internal/runtime/executor/codex_websockets_executor.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use serde_json::Value;
use tokio::sync::mpsc;

use super::codex_executor_auth::CodexSubscriptionAuth;
use super::codex_executor_reasoning::{
    codex_reasoning_replay_session_key, CodexReasoningReplayCache, CodexReasoningReplayScope,
};
use super::codex_executor_request::{
    apply_codex_identity_confuse_body, apply_codex_identity_confuse_headers,
    CodexIdentityConfuseState, CodexIdentityPolicy,
};
use super::codex_executor_terminal::{CodexTerminalAccumulator, CodexTerminalEvent};
use super::codex_websockets_connection::{
    build_codex_responses_websocket_url, build_codex_websocket_request_body, CodexWebsocketFrame,
    CodexWebsocketTransport,
};
use super::codex_websockets_errors::{
    normalize_codex_websocket_completion, parse_codex_websocket_error, CodexWebsocketError,
};
use super::codex_websockets_request::{
    apply_codex_websocket_headers, CodexWebsocketHeaderDefaults, CodexWebsocketHeaders,
};
use super::codex_websockets_session::{
    CodexWebsocketSession, CodexWebsocketSessionState, CodexWebsocketSessionStore,
};
use crate::sdk::cliproxy::executor::{bind_execution_resource, ExecutionLifecycle};

const DEFAULT_MAX_BUFFERED_EVENTS: usize = 4_096;
const DEFAULT_MAX_BUFFERED_EVENT_BYTES: usize = 64 * 1024 * 1024;

struct CodexWebsocketAttemptContext<'a> {
    payload: &'a [u8],
    replay_scope: Option<&'a CodexReasoningReplayScope>,
    reconnects: usize,
    sender: Option<&'a mpsc::Sender<Result<Vec<u8>, CodexWebsocketError>>>,
    committed_flag: Option<&'a Arc<AtomicBool>>,
    identity_state: &'a CodexIdentityConfuseState,
}

#[derive(Clone)]
pub struct CodexWebsocketExecutionRequest {
    pub auth_id: String,
    pub session_id: String,
    pub responses_url: String,
    pub body: Vec<u8>,
    pub headers: CodexWebsocketHeaders,
    pub execution_lifecycle: Option<Arc<dyn ExecutionLifecycle>>,
}

impl fmt::Debug for CodexWebsocketExecutionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodexWebsocketExecutionRequest")
            .field("auth_id", &"[REDACTED]")
            .field("session_id", &"[REDACTED]")
            .field("responses_url", &self.responses_url)
            .field("body_bytes", &self.body.len())
            .field("headers", &self.headers)
            .field("execution_lifecycle", &self.execution_lifecycle.is_some())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexWebsocketExecutionResult {
    pub completed: Vec<u8>,
    pub events: Vec<Vec<u8>>,
    pub reconnects: usize,
    pub committed: bool,
}

#[derive(Clone)]
pub struct CodexWebsocketsExecutor {
    auth: Arc<CodexSubscriptionAuth>,
    transport: Arc<dyn CodexWebsocketTransport>,
    sessions: Arc<CodexWebsocketSessionStore>,
    reasoning: Arc<CodexReasoningReplayCache>,
    defaults: CodexWebsocketHeaderDefaults,
    max_reconnects: usize,
    max_buffered_events: usize,
    max_buffered_event_bytes: usize,
    identity: Option<(String, CodexIdentityPolicy)>,
}

impl CodexWebsocketsExecutor {
    pub fn new(
        auth: Arc<CodexSubscriptionAuth>,
        transport: Arc<dyn CodexWebsocketTransport>,
        sessions: Arc<CodexWebsocketSessionStore>,
        reasoning: Arc<CodexReasoningReplayCache>,
    ) -> Self {
        Self {
            auth,
            transport,
            sessions,
            reasoning,
            defaults: CodexWebsocketHeaderDefaults::default(),
            max_reconnects: 1,
            max_buffered_events: DEFAULT_MAX_BUFFERED_EVENTS,
            max_buffered_event_bytes: DEFAULT_MAX_BUFFERED_EVENT_BYTES,
            identity: None,
        }
    }

    pub fn with_header_defaults(mut self, defaults: CodexWebsocketHeaderDefaults) -> Self {
        self.defaults = defaults;
        self
    }

    pub fn with_max_reconnects(mut self, max_reconnects: usize) -> Self {
        self.max_reconnects = max_reconnects.min(2);
        self
    }

    pub fn with_buffer_limits(mut self, max_events: usize, max_bytes: usize) -> Self {
        self.max_buffered_events = max_events.max(1);
        self.max_buffered_event_bytes = max_bytes.max(1);
        self
    }

    pub fn with_identity_policy(
        mut self,
        auth_id: impl Into<String>,
        policy: CodexIdentityPolicy,
    ) -> Self {
        self.identity = Some((auth_id.into(), policy));
        self
    }

    pub fn sessions(&self) -> &Arc<CodexWebsocketSessionStore> {
        &self.sessions
    }

    pub async fn execute(
        &self,
        request: CodexWebsocketExecutionRequest,
    ) -> Result<CodexWebsocketExecutionResult, CodexWebsocketError> {
        self.execute_with_live_sink(request, None, None).await
    }

    pub(crate) async fn execute_with_live_sink(
        &self,
        request: CodexWebsocketExecutionRequest,
        sender: Option<mpsc::Sender<Result<Vec<u8>, CodexWebsocketError>>>,
        committed: Option<Arc<AtomicBool>>,
    ) -> Result<CodexWebsocketExecutionResult, CodexWebsocketError> {
        let session = self
            .sessions
            .get_or_create(&request.session_id)
            .ok_or_else(|| CodexWebsocketError::protocol("missing_session_id", false))?;
        let mut state = session.execution.lock().await;
        let execution_lifecycle = request.execution_lifecycle.clone();
        state.committed = false;
        let credentials = self
            .auth
            .load()
            .await
            .map_err(|_| CodexWebsocketError::protocol("credential_load_failed", false))?;
        let account_id =
            crate::internal::auth::codex::parse_jwt_token(credentials.id_token().expose_secret())
                .map(|claims| claims.account_id().to_owned())
                .unwrap_or_default();
        let websocket_url = build_codex_responses_websocket_url(&request.responses_url)?;
        let mut headers = apply_codex_websocket_headers(
            request.headers,
            credentials.access_token(),
            &account_id,
            &request.session_id,
            &self.defaults,
        );
        let model = serde_json::from_slice::<Value>(&request.body)
            .ok()
            .and_then(|value| {
                value
                    .get("model")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| "codex".to_owned());
        let replay_scope =
            CodexReasoningReplayScope::from_request(&model, &request.session_id, &request.body);
        let replay_body = replay_scope
            .as_ref()
            .map(|scope| self.reasoning.apply(scope, &request.body))
            .unwrap_or_else(|| request.body.clone());
        let (body, mut identity_state) = match &self.identity {
            Some((auth_id, policy)) => {
                apply_codex_identity_confuse_body(policy, auth_id, &request.body, &replay_body)
            }
            None => (replay_body, CodexIdentityConfuseState::default()),
        };
        if let Some((auth_id, _)) = &self.identity {
            apply_codex_identity_confuse_headers(&mut headers, auth_id, &mut identity_state);
        }
        let payload = build_codex_websocket_request_body(&body);

        for reconnects in 0..=self.max_reconnects {
            if let Err(error) = self
                .ensure_connection(
                    &session,
                    &mut state,
                    &request.auth_id,
                    &websocket_url,
                    &headers,
                    execution_lifecycle.as_ref(),
                )
                .await
            {
                close_state_connection(&mut state, "lifecycle_bind_failed").await;
                return Err(error);
            }
            let result = self
                .execute_once(
                    &mut state,
                    CodexWebsocketAttemptContext {
                        payload: &payload,
                        replay_scope: replay_scope.as_ref(),
                        reconnects,
                        sender: sender.as_ref(),
                        committed_flag: committed.as_ref(),
                        identity_state: &identity_state,
                    },
                )
                .await;
            match result {
                Ok(result) => return Ok(result),
                Err(error)
                    if error.retryable && !state.committed && reconnects < self.max_reconnects =>
                {
                    close_state_connection(&mut state, "retry").await;
                }
                Err(error) => {
                    close_state_connection(&mut state, "execution_failed").await;
                    return Err(error);
                }
            }
        }
        Err(CodexWebsocketError::protocol("reconnect_exhausted", false))
    }

    async fn ensure_connection(
        &self,
        _session: &CodexWebsocketSession,
        state: &mut CodexWebsocketSessionState,
        auth_id: &str,
        url: &str,
        headers: &CodexWebsocketHeaders,
        lifecycle: Option<&Arc<dyn ExecutionLifecycle>>,
    ) -> Result<(), CodexWebsocketError> {
        let target_changed = state.auth_id != auth_id || state.target_url != url;
        if target_changed {
            close_state_connection(state, "target_replaced").await;
        }
        if state.connection.is_none() {
            state.connection = Some(self.transport.connect(url, headers).await?);
            state.auth_id = auth_id.to_owned();
            state.target_url = url.to_owned();
            state.generation = state.generation.saturating_add(1);
        }
        bind_connection_lifecycle(state, lifecycle)
    }

    async fn execute_once(
        &self,
        state: &mut CodexWebsocketSessionState,
        context: CodexWebsocketAttemptContext<'_>,
    ) -> Result<CodexWebsocketExecutionResult, CodexWebsocketError> {
        let connection = state
            .connection
            .as_mut()
            .ok_or_else(|| CodexWebsocketError::protocol("connection_missing", true))?;
        connection
            .send(CodexWebsocketFrame::Text(context.payload.to_vec()))
            .await?;
        let mut terminal = CodexTerminalAccumulator::default();
        let mut events = Vec::new();
        let mut buffered_bytes = 0usize;
        loop {
            let frame = connection.receive().await?;
            let payload = match frame {
                CodexWebsocketFrame::Text(payload) | CodexWebsocketFrame::Binary(payload) => {
                    payload
                }
                CodexWebsocketFrame::Ping(payload) => {
                    connection.send(CodexWebsocketFrame::Pong(payload)).await?;
                    continue;
                }
                CodexWebsocketFrame::Pong(_) => continue,
                CodexWebsocketFrame::Close { code } => {
                    return Err(
                        super::codex_websockets_connection::map_codex_websocket_close(code),
                    );
                }
            };
            if let Some(error) = parse_codex_websocket_error(&payload) {
                if let Some(scope) = context.replay_scope {
                    self.reasoning
                        .clear_on_invalid_signature(scope, error.status, &payload);
                }
                return Err(error);
            }
            let payload = context
                .identity_state
                .expose_response(&normalize_codex_websocket_completion(&payload));
            let event_type = serde_json::from_slice::<Value>(&payload)
                .ok()
                .and_then(|value| {
                    value
                        .get("type")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned)
                });
            if event_type
                .as_deref()
                .is_some_and(|kind| kind.starts_with("response."))
            {
                state.committed = true;
                if let Some(flag) = context.committed_flag {
                    flag.store(true, Ordering::Release);
                }
            }
            buffered_bytes = buffered_bytes.saturating_add(payload.len());
            if events.len() >= self.max_buffered_events
                || buffered_bytes > self.max_buffered_event_bytes
            {
                return Err(CodexWebsocketError::protocol(
                    "buffered_response_limit_exceeded",
                    false,
                ));
            }
            if let Some(sender) = context.sender {
                if sender
                    .send(Ok(
                        super::codex_websockets_errors::encode_codex_websocket_as_sse(&payload),
                    ))
                    .await
                    .is_err()
                {
                    return Err(CodexWebsocketError::protocol("downstream_closed", false));
                }
            } else {
                events.push(payload.clone());
            }
            match terminal.ingest(&payload, SystemTime::now()) {
                CodexTerminalEvent::Continue => {}
                CodexTerminalEvent::Completed(completed) => {
                    if let Some(scope) = context.replay_scope.cloned() {
                        self.reasoning.commit_completed(scope, &completed);
                    }
                    return Ok(CodexWebsocketExecutionResult {
                        completed,
                        events,
                        reconnects: context.reconnects,
                        committed: state.committed,
                    });
                }
                CodexTerminalEvent::Failed(error) => {
                    return Err(CodexWebsocketError {
                        status: error.status,
                        code: error.code,
                        retryable: !state.committed
                            && matches!(error.status, 408 | 429 | 500..=599),
                        request_scoped: true,
                        headers: Default::default(),
                    });
                }
            }
        }
    }

    pub async fn close_execution_session(&self, session_id: &str) {
        self.sessions.close(session_id).await;
    }
}

impl fmt::Debug for CodexWebsocketsExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodexWebsocketsExecutor")
            .field("sessions", &self.sessions)
            .field("reasoning", &self.reasoning)
            .field("max_reconnects", &self.max_reconnects)
            .field("max_buffered_events", &self.max_buffered_events)
            .field("max_buffered_event_bytes", &self.max_buffered_event_bytes)
            .field("credentials", &"[REDACTED]")
            .finish()
    }
}

pub(crate) fn bind_connection_lifecycle(
    state: &mut CodexWebsocketSessionState,
    lifecycle: Option<&Arc<dyn ExecutionLifecycle>>,
) -> Result<(), CodexWebsocketError> {
    let Some(lifecycle) = lifecycle else {
        return Ok(());
    };
    if state.lifecycle_generation == state.generation
        && state
            .lifecycle
            .as_ref()
            .is_some_and(|bound| Arc::ptr_eq(bound, lifecycle))
    {
        return Ok(());
    }
    let closer = state
        .connection
        .as_ref()
        .and_then(|connection| connection.lifecycle_closer())
        .ok_or_else(|| CodexWebsocketError::protocol("lifecycle_close_unsupported", false))?;
    bind_execution_resource(Some(lifecycle.as_ref()), Some(closer))
        .map_err(|_| CodexWebsocketError::protocol("lifecycle_bind_failed", false))?;
    let previous = state.lifecycle.replace(Arc::clone(lifecycle));
    state.lifecycle_generation = state.generation;
    if let Some(previous) = previous.filter(|previous| !Arc::ptr_eq(previous, lifecycle)) {
        previous.end("target_replaced");
    }
    Ok(())
}

async fn close_state_connection(state: &mut CodexWebsocketSessionState, reason: &str) {
    state.close(reason).await;
}

pub fn codex_websockets_enabled(
    attributes: &std::collections::BTreeMap<String, String>,
    metadata: &std::collections::BTreeMap<String, Value>,
) -> bool {
    if let Some(value) = attributes.get("websockets") {
        return value.trim().eq_ignore_ascii_case("true");
    }
    metadata.get("websockets").is_some_and(|value| {
        value.as_bool().unwrap_or_else(|| {
            value
                .as_str()
                .is_some_and(|value| value.trim().eq_ignore_ascii_case("true"))
        })
    })
}

pub fn codex_replay_session_from_request(
    body: &[u8],
    headers: &CodexWebsocketHeaders,
) -> Option<String> {
    codex_reasoning_replay_session_key(
        body,
        super::codex_websockets_request::codex_session_header_value(headers),
    )
}
