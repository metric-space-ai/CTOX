// ref: sdk/api/handlers/handlers_execution.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

use crate::sdk::cliproxy::executor::ExecutionMetadata;
use crate::sdk::pluginapi::ExecutorRequest;

use super::{HandlerRequestContext, ProtocolExecutionRequest};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HandlerExecutionError {
    pub status_code: u16,
    pub message: String,
}

impl fmt::Display for HandlerExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for HandlerExecutionError {}

#[must_use]
pub fn build_executor_request(
    context: &HandlerRequestContext,
    request: &ProtocolExecutionRequest,
    provider: &str,
    translated_body: Vec<u8>,
) -> ExecutorRequest {
    let metadata = context.execution_metadata();
    ExecutorRequest {
        auth_provider: provider.trim().to_owned(),
        model: request.model.trim().to_owned(),
        format: request.exit_protocol.trim().to_owned(),
        stream: request.stream,
        alt: request.alt.clone(),
        headers: if request.headers.is_empty() {
            context.headers.clone()
        } else {
            request.headers.clone()
        },
        query: if request.query.is_empty() {
            context.query.clone()
        } else {
            request.query.clone()
        },
        original_request: request.body.clone(),
        source_format: request.entry_protocol.trim().to_owned(),
        payload: translated_body,
        metadata: metadata_to_json(&metadata),
        ..ExecutorRequest::default()
    }
}

fn metadata_to_json(metadata: &ExecutionMetadata) -> crate::sdk::pluginapi::JsonMetadata {
    let mut result = metadata.extensions.clone();
    for (key, value) in [
        ("request_path", metadata.request_path.as_ref()),
        ("pinned_auth_id", metadata.pinned_auth_id.as_ref()),
        (
            "execution_session_id",
            metadata.execution_session_id.as_ref(),
        ),
        ("caller_scope", metadata.caller_scope.as_ref()),
    ] {
        if let Some(value) = value {
            result.insert(key.to_owned(), serde_json::Value::String(value.clone()));
        }
    }
    if metadata.disallow_free_auth {
        result.insert(
            "disallow_free_auth".to_owned(),
            serde_json::Value::Bool(true),
        );
    }
    result
}
