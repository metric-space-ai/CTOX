// ref: sdk/api/handlers/handlers_context.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::sdk::cliproxy::executor::{
    ExecutionMetadata, Headers, QueryValues, SelectedAuthCallback, SelectedAuthIndexCallback,
};
use crate::sdk::cliproxy::session::caller_scope;

/// Request-local cancellation handle. It carries no task or process authority;
/// the CTOX-owned host decides how cancellation is observed by its executor.
#[derive(Clone, Debug, Default)]
pub struct HandlerCancellation(Arc<AtomicBool>);

impl HandlerCancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Transport-neutral replacement for values upstream stores in a Gin context.
#[derive(Clone, Default)]
pub struct HandlerRequestContext {
    pub headers: Headers,
    pub query: QueryValues,
    pub request_path: String,
    pub pinned_auth_id: String,
    pub execution_session_id: String,
    pub caller_api_key: String,
    pub disallow_free_auth: bool,
    pub websocket_upgrade: bool,
    pub selected_auth_callback: Option<SelectedAuthCallback>,
    pub selected_auth_index_callback: Option<SelectedAuthIndexCallback>,
    pub cancellation: HandlerCancellation,
}

impl HandlerRequestContext {
    #[must_use]
    pub fn with_pinned_auth_id(mut self, auth_id: impl Into<String>) -> Self {
        self.pinned_auth_id = auth_id.into().trim().to_owned();
        self
    }

    #[must_use]
    pub fn with_execution_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.execution_session_id = session_id.into().trim().to_owned();
        self
    }

    #[must_use]
    pub fn with_disallow_free_auth(mut self) -> Self {
        self.disallow_free_auth = true;
        self
    }

    #[must_use]
    pub fn execution_metadata(&self) -> ExecutionMetadata {
        let idempotency_key = header_value(&self.headers, "Idempotency-Key");
        let caller_scope = caller_scope(&self.caller_api_key);
        let mut metadata = ExecutionMetadata {
            request_path: non_empty(&self.request_path),
            disallow_free_auth: self.disallow_free_auth,
            pinned_auth_id: non_empty(&self.pinned_auth_id),
            selected_auth_callback: self.selected_auth_callback.clone(),
            execution_session_id: non_empty(&self.execution_session_id),
            caller_scope: (!caller_scope.is_empty()).then_some(caller_scope),
            ..ExecutionMetadata::default()
        };
        if !self.websocket_upgrade {
            metadata.selected_auth_index_callback = self.selected_auth_index_callback.clone();
        }
        if let Some(idempotency_key) = idempotency_key {
            metadata.extensions.insert(
                "idempotency_key".to_owned(),
                serde_json::Value::String(idempotency_key),
            );
        }
        metadata
    }
}

fn non_empty(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn header_value(headers: &Headers, name: &str) -> Option<String> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .and_then(|(_, values)| values.first())
        .and_then(|value| non_empty(value))
}
