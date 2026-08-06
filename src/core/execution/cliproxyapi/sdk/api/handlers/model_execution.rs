// ref: sdk/api/handlers/model_execution.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

use tokio::sync::mpsc;

use crate::sdk::cliproxy::executor::{Headers, QueryValues};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProtocolExecutionRequest {
    pub entry_protocol: String,
    pub exit_protocol: String,
    pub forced_provider: String,
    pub auth_selection_model: String,
    pub model: String,
    pub stream: bool,
    pub body: Vec<u8>,
    pub headers: Headers,
    pub query: QueryValues,
    pub alt: String,
    pub source_plugin_id: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelExecutionRequest {
    pub entry_protocol: String,
    pub exit_protocol: String,
    pub model: String,
    pub stream: bool,
    pub body: Vec<u8>,
    pub headers: Headers,
    pub query: QueryValues,
    pub alt: String,
    pub source_plugin_id: String,
}

impl From<ModelExecutionRequest> for ProtocolExecutionRequest {
    fn from(request: ModelExecutionRequest) -> Self {
        Self {
            entry_protocol: request.entry_protocol,
            exit_protocol: request.exit_protocol,
            model: request.model,
            stream: request.stream,
            body: request.body,
            headers: request.headers,
            query: request.query,
            alt: request.alt,
            source_plugin_id: request.source_plugin_id,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelExecutionResponse {
    pub status_code: u16,
    pub headers: Headers,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelExecutionChunk {
    pub payload: Vec<u8>,
    pub error: Option<ModelExecutionError>,
}

pub struct ModelExecutionStream {
    pub status_code: u16,
    pub headers: Headers,
    pub chunks: mpsc::Receiver<ModelExecutionChunk>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelExecutionError {
    pub status_code: u16,
    pub message: String,
    pub headers: Headers,
}

impl fmt::Display for ModelExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ModelExecutionError {}

#[must_use]
pub fn response_protocol(entry_protocol: &str, exit_protocol: &str) -> String {
    let exit = exit_protocol.trim();
    if exit.is_empty() {
        entry_protocol.trim().to_owned()
    } else {
        exit.to_owned()
    }
}

#[cfg(test)]
#[path = "model_execution_test.rs"]
mod model_execution_test;
