// ref: internal/pluginhost/executor_route.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: direct executor routes use snapshot-bound process capabilities
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::pluginapi::{
    ExecutorRequest, ExecutorResponse, ExecutorStreamResponse, ModelRouteRequest, ProviderExecutor,
};
use crate::sdk::translator::{claude, openai, openai_response, Format, Registry};

use super::adapters::plugin_from_record;
use super::auth_provider::HostConfigSummarySource;
use super::callback_contexts::CallbackContextRegistry;
use super::snapshot::{CapabilityRecord, Snapshot};
use super::stream_bridge::StreamBridge;

#[derive(Clone)]
pub struct ExecutorRoute {
    snapshot: Arc<Snapshot>,
    contexts: CallbackContextRegistry,
    streams: StreamBridge,
    host: Arc<dyn HostConfigSummarySource>,
    translators: Arc<Registry>,
}

impl std::fmt::Debug for ExecutorRoute {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutorRoute")
            .field("active_plugins", &self.snapshot.records().len())
            .finish_non_exhaustive()
    }
}

impl ExecutorRoute {
    pub fn new(
        snapshot: Arc<Snapshot>,
        contexts: CallbackContextRegistry,
        streams: StreamBridge,
        host: Arc<dyn HostConfigSummarySource>,
        translators: Arc<Registry>,
    ) -> Self {
        Self {
            snapshot,
            contexts,
            streams,
            host,
            translators,
        }
    }

    pub fn ready(&self, plugin_id: &str, request: &ModelRouteRequest) -> bool {
        let Some(record) = self.record(plugin_id) else {
            return false;
        };
        self.formats(record, &normalize_format(&request.source_format))
            .is_some()
    }

    pub fn request_to_format(&self, plugin_id: &str, request: &ExecutorRequest) -> Option<Format> {
        let record = self.record(plugin_id)?;
        let requested = if request.source_format.trim().is_empty() {
            normalize_format(&request.format)
        } else {
            normalize_format(&request.source_format)
        };
        self.formats(record, &requested).map(|formats| formats.0)
    }

    pub async fn execute(
        &self,
        plugin_id: &str,
        request: ExecutorRequest,
    ) -> Result<ExecutorResponse, ExecutorRouteError> {
        let executor = self.executor(plugin_id, &request)?;
        executor
            .execute(request)
            .await
            .map_err(|_| ExecutorRouteError::Execution)
    }

    pub async fn execute_stream(
        &self,
        plugin_id: &str,
        request: ExecutorRequest,
    ) -> Result<ExecutorStreamResponse, ExecutorRouteError> {
        let executor = self.executor(plugin_id, &request)?;
        executor
            .execute_stream(request)
            .await
            .map_err(|_| ExecutorRouteError::Execution)
    }

    pub async fn count_tokens(
        &self,
        plugin_id: &str,
        request: ExecutorRequest,
    ) -> Result<ExecutorResponse, ExecutorRouteError> {
        let executor = self.executor(plugin_id, &request)?;
        executor
            .count_tokens(request)
            .await
            .map_err(|_| ExecutorRouteError::Execution)
    }

    fn executor(
        &self,
        plugin_id: &str,
        request: &ExecutorRequest,
    ) -> Result<Arc<dyn ProviderExecutor>, ExecutorRouteError> {
        let record = self.record(plugin_id).ok_or(ExecutorRouteError::NotFound)?;
        let requested = if request.source_format.trim().is_empty() {
            normalize_format(&request.format)
        } else {
            normalize_format(&request.source_format)
        };
        self.formats(record, &requested)
            .ok_or(ExecutorRouteError::UnsupportedFormat)?;
        plugin_from_record(
            record,
            self.contexts.clone(),
            self.streams.clone(),
            self.host.clone(),
        )
        .map_err(|_| ExecutorRouteError::InvalidPlugin)?
        .capabilities
        .executor
        .ok_or(ExecutorRouteError::InvalidPlugin)
    }

    fn record(&self, plugin_id: &str) -> Option<&Arc<CapabilityRecord>> {
        self.snapshot.record(plugin_id.trim()).filter(|record| {
            record.capabilities.executor
                && record.identifiers.contains_key("executor")
                && matches!(
                    record.capabilities.executor_model_scope.0.as_str(),
                    "" | "both" | "static"
                )
        })
    }

    fn formats(&self, record: &CapabilityRecord, requested: &Format) -> Option<(Format, Format)> {
        let inputs = normalized_formats(&record.capabilities.executor_input_formats);
        let outputs = normalized_formats(&record.capabilities.executor_output_formats);
        let input = inputs.iter().find_map(|candidate| {
            (candidate == requested
                || requested.as_str().is_empty()
                || self
                    .translators
                    .has_request_transformer(requested, candidate))
            .then(|| candidate.clone())
        })?;
        let output = outputs.iter().find_map(|candidate| {
            (candidate == requested
                || requested.as_str().is_empty()
                || self
                    .translators
                    .has_response_transformer(requested, candidate)
                || self
                    .snapshot
                    .records()
                    .iter()
                    .any(|record| record.capabilities.response_translator))
            .then(|| candidate.clone())
        })?;
        Some((input, output))
    }
}

fn normalized_formats(raw: &[String]) -> Vec<Format> {
    let mut formats = raw
        .iter()
        .map(|value| normalize_format(value))
        .filter(|format| !format.as_str().is_empty())
        .collect::<Vec<_>>();
    formats.sort();
    formats.dedup();
    formats
}

fn normalize_format(raw: &str) -> Format {
    match raw.trim().to_ascii_lowercase().as_str() {
        "chat-completions"
        | "chat_completions"
        | "openai-chat-completions"
        | "openai_chat_completions" => openai(),
        "responses" | "openai-responses" | "openai_responses" => openai_response(),
        "anthropic" => claude(),
        value => Format::from(value),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutorRouteError {
    NotFound,
    InvalidPlugin,
    UnsupportedFormat,
    Execution,
}

impl std::fmt::Display for ExecutorRouteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NotFound => "plugin executor was not found",
            Self::InvalidPlugin => "plugin executor registration is invalid",
            Self::UnsupportedFormat => "plugin executor does not support the requested format",
            Self::Execution => "plugin executor call failed",
        })
    }
}

impl std::error::Error for ExecutorRouteError {}
