// ref: internal/runtime/executor/antigravity_executor.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use zeroize::Zeroizing;

use crate::internal::auth::antigravity::SecretString;
use crate::internal::cache::SignatureKvStore;
use crate::internal::runtime::executor::antigravity_reasoning_replay::{
    replay_now_ms, AntigravityReasoningReplayAccumulator,
};
use crate::internal::translator::antigravity::claude::{
    convert_antigravity_response_to_claude_stream_with_runtime, AntigravityClaudeStreamState,
};
use crate::internal::translator::antigravity::openai::responses::{
    convert_antigravity_response_to_openai_responses_stream, AntigravityToResponsesState,
};
use crate::internal::translator::common::SseDecoder;

pub const DEFAULT_ANTIGRAVITY_BASE_URL: &str = "https://daily-cloudcode-pa.googleapis.com";
pub const ANTIGRAVITY_GENERATE_PATH: &str = "/v1internal:generateContent";
pub const ANTIGRAVITY_STREAM_PATH: &str = "/v1internal:streamGenerateContent";
pub const ANTIGRAVITY_MODELS_PATH: &str = "/v1internal:fetchAvailableModels";
pub const ANTIGRAVITY_COUNT_TOKENS_PATH: &str = "/v1internal:countTokens";
pub const ANTIGRAVITY_USER_AGENT: &str = "antigravity/hub/2.2.1 darwin/arm64";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AntigravityUpstreamTarget {
    base_url: String,
}

impl AntigravityUpstreamTarget {
    pub fn new(base_url: impl Into<String>) -> Result<Self, AntigravityTargetError> {
        let parsed = url::Url::parse(base_url.into().trim())
            .map_err(|_| AntigravityTargetError::InvalidUrl)?;
        if !matches!(parsed.scheme(), "http" | "https")
            || parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some()
            || parsed.fragment().is_some()
        {
            return Err(AntigravityTargetError::InvalidUrl);
        }
        Ok(Self {
            base_url: parsed.as_str().trim_end_matches('/').to_owned(),
        })
    }

    pub fn default_subscription() -> Self {
        Self {
            base_url: DEFAULT_ANTIGRAVITY_BASE_URL.to_owned(),
        }
    }

    pub fn generate_url(&self) -> String {
        format!("{}{ANTIGRAVITY_GENERATE_PATH}", self.base_url)
    }

    pub fn stream_url(&self) -> String {
        format!("{}{ANTIGRAVITY_STREAM_PATH}", self.base_url)
    }

    pub fn models_url(&self) -> String {
        format!("{}{ANTIGRAVITY_MODELS_PATH}", self.base_url)
    }

    pub fn count_tokens_url(&self) -> String {
        format!("{}{ANTIGRAVITY_COUNT_TOKENS_PATH}", self.base_url)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntigravityTargetError {
    InvalidUrl,
}

impl fmt::Display for AntigravityTargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Antigravity upstream URL is invalid")
    }
}

impl std::error::Error for AntigravityTargetError {}

pub struct AntigravityGenerateRequest {
    url: String,
    authorization: SecretString,
    body: Zeroizing<Vec<u8>>,
}

impl AntigravityGenerateRequest {
    pub fn new(
        target: &AntigravityUpstreamTarget,
        authorization: SecretString,
        body: Vec<u8>,
    ) -> Self {
        Self {
            url: target.generate_url(),
            authorization,
            body: Zeroizing::new(body),
        }
    }

    pub fn new_stream(
        target: &AntigravityUpstreamTarget,
        authorization: SecretString,
        body: Vec<u8>,
    ) -> Self {
        Self {
            url: target.stream_url(),
            authorization,
            body: Zeroizing::new(body),
        }
    }

    pub fn new_model_discovery(
        target: &AntigravityUpstreamTarget,
        authorization: SecretString,
    ) -> Self {
        Self {
            url: target.models_url(),
            authorization,
            body: Zeroizing::new(b"{}".to_vec()),
        }
    }

    pub fn new_count_tokens(
        target: &AntigravityUpstreamTarget,
        authorization: SecretString,
        body: Vec<u8>,
    ) -> Self {
        Self {
            url: target.count_tokens_url(),
            authorization,
            body: Zeroizing::new(body),
        }
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn access_token(&self) -> &SecretString {
        &self.authorization
    }

    pub fn body(&self) -> &[u8] {
        &self.body
    }
}

impl fmt::Debug for AntigravityGenerateRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AntigravityGenerateRequest")
            .field("url", &self.url)
            .field("authorization", &"Bearer [REDACTED]")
            .field("body", &"[REDACTED]")
            .finish()
    }
}

pub struct AntigravityGenerateResponse {
    status: u16,
    retry_after: Option<String>,
    body: Zeroizing<Vec<u8>>,
}

impl AntigravityGenerateResponse {
    pub fn new(status: u16, retry_after: Option<String>, body: Vec<u8>) -> Self {
        Self {
            status,
            retry_after,
            body: Zeroizing::new(body),
        }
    }

    pub fn status(&self) -> u16 {
        self.status
    }

    pub fn retry_after(&self) -> Option<&str> {
        self.retry_after.as_deref()
    }

    pub fn body(&self) -> &[u8] {
        &self.body
    }
}

impl fmt::Debug for AntigravityGenerateResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AntigravityGenerateResponse")
            .field("status", &self.status)
            .field("retry_after", &self.retry_after)
            .field("body", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntigravityGenerateTransportFailure {
    Timeout,
    Connect,
    Protocol,
}

pub trait AntigravityGenerateTransport: Send + Sync {
    fn execute<'a>(
        &'a self,
        request: &'a AntigravityGenerateRequest,
        timeout: Duration,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        AntigravityGenerateResponse,
                        AntigravityGenerateTransportFailure,
                    >,
                > + Send
                + 'a,
        >,
    >;
}

pub trait AntigravityGenerateStreamingTransport: Send + Sync {
    fn execute_stream<'a>(
        &'a self,
        request: &'a AntigravityGenerateRequest,
        timeout: Duration,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        AntigravityGenerateStreamResponse,
                        AntigravityGenerateTransportFailure,
                    >,
                > + Send
                + 'a,
        >,
    >;
}

pub struct AntigravityGenerateStreamResponse {
    status: u16,
    retry_after: Option<String>,
    chunks: mpsc::Receiver<Result<Vec<u8>, AntigravityGenerateTransportFailure>>,
}

impl AntigravityGenerateStreamResponse {
    pub fn new(
        status: u16,
        retry_after: Option<String>,
        chunks: mpsc::Receiver<Result<Vec<u8>, AntigravityGenerateTransportFailure>>,
    ) -> Self {
        Self {
            status,
            retry_after,
            chunks,
        }
    }

    pub fn status(&self) -> u16 {
        self.status
    }
    pub fn retry_after(&self) -> Option<&str> {
        self.retry_after.as_deref()
    }
    async fn next_chunk(&mut self) -> Option<Result<Vec<u8>, AntigravityGenerateTransportFailure>> {
        self.chunks.recv().await
    }
}

impl fmt::Debug for AntigravityGenerateStreamResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AntigravityGenerateStreamResponse")
            .field("status", &self.status)
            .field("retry_after", &self.retry_after)
            .field("chunks", &"[REDACTED]")
            .finish()
    }
}

pub struct AntigravityResponsesStream {
    upstream: AntigravityGenerateStreamResponse,
    decoder: SseDecoder,
    adapter: AntigravityDownstreamStreamAdapter,
    original_request: Vec<u8>,
    translated_request: Vec<u8>,
    pending: VecDeque<Vec<u8>>,
    replay: Option<AntigravityReasoningReplayAccumulator>,
    finished: bool,
}

enum AntigravityDownstreamStreamAdapter {
    Responses(AntigravityToResponsesState),
    Claude {
        state: AntigravityClaudeStreamState,
        web_search_tool_use_id: String,
        signature_store: Option<Arc<dyn SignatureKvStore>>,
    },
}

impl AntigravityResponsesStream {
    pub fn new(
        upstream: AntigravityGenerateStreamResponse,
        original_request: Vec<u8>,
        translated_request: Vec<u8>,
    ) -> Self {
        Self {
            upstream,
            decoder: SseDecoder::new(),
            adapter: AntigravityDownstreamStreamAdapter::Responses(
                AntigravityToResponsesState::default(),
            ),
            original_request,
            translated_request,
            pending: VecDeque::new(),
            replay: None,
            finished: false,
        }
    }

    pub fn new_claude(
        upstream: AntigravityGenerateStreamResponse,
        original_request: Vec<u8>,
        translated_request: Vec<u8>,
        web_search_tool_use_id: String,
        signature_store: Option<Arc<dyn SignatureKvStore>>,
    ) -> Self {
        Self {
            upstream,
            decoder: SseDecoder::new(),
            adapter: AntigravityDownstreamStreamAdapter::Claude {
                state: AntigravityClaudeStreamState::default(),
                web_search_tool_use_id,
                signature_store,
            },
            original_request,
            translated_request,
            pending: VecDeque::new(),
            replay: None,
            finished: false,
        }
    }

    pub fn with_replay_accumulator(
        mut self,
        accumulator: AntigravityReasoningReplayAccumulator,
    ) -> Self {
        self.replay = Some(accumulator);
        self
    }

    pub fn status(&self) -> u16 {
        self.upstream.status()
    }
    pub fn retry_after(&self) -> Option<&str> {
        self.upstream.retry_after()
    }

    pub async fn bootstrap(&mut self) -> Result<(), AntigravityGenerateTransportFailure> {
        if !(200..300).contains(&self.status()) {
            return Ok(());
        }
        while self.pending.is_empty() {
            self.fill().await?;
            if self.finished && self.pending.is_empty() {
                return Err(AntigravityGenerateTransportFailure::Protocol);
            }
        }
        Ok(())
    }

    pub async fn next_event(
        &mut self,
    ) -> Option<Result<Vec<u8>, AntigravityGenerateTransportFailure>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(Ok(event));
            }
            if self.finished {
                return None;
            }
            if let Err(error) = self.fill().await {
                return Some(Err(error));
            }
        }
    }

    async fn fill(&mut self) -> Result<(), AntigravityGenerateTransportFailure> {
        match self.upstream.next_chunk().await {
            Some(Ok(chunk)) => {
                for event in self.decoder.push(&chunk) {
                    if let Some(replay) = self.replay.as_mut() {
                        replay.observe_response_payload(&event.data);
                    }
                    self.translate_event(&event.data);
                }
                Ok(())
            }
            Some(Err(error)) => {
                self.finished = true;
                Err(error)
            }
            None => {
                for event in self.decoder.finish() {
                    if let Some(replay) = self.replay.as_mut() {
                        replay.observe_response_payload(&event.data);
                    }
                    self.translate_event(&event.data);
                }
                self.translate_event(b"[DONE]");
                if let Some(replay) = self.replay.take() {
                    let _ = replay.commit(replay_now_ms());
                }
                self.finished = true;
                Ok(())
            }
        }
    }

    fn translate_event(&mut self, data: &[u8]) {
        let events = match &mut self.adapter {
            AntigravityDownstreamStreamAdapter::Responses(state) => {
                convert_antigravity_response_to_openai_responses_stream(
                    &self.original_request,
                    &self.translated_request,
                    data,
                    state,
                )
            }
            AntigravityDownstreamStreamAdapter::Claude {
                state,
                web_search_tool_use_id,
                signature_store,
            } => convert_antigravity_response_to_claude_stream_with_runtime(
                &self.original_request,
                &self.translated_request,
                data,
                state,
                web_search_tool_use_id,
                signature_store.as_deref(),
            ),
        };
        self.pending.extend(events);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::auth::antigravity::SecretString;

    #[test]
    fn target_and_request_reject_credential_urls_and_redact_payload() {
        assert_eq!(
            AntigravityUpstreamTarget::new("https://user:pass@example.com").unwrap_err(),
            AntigravityTargetError::InvalidUrl
        );
        let request = AntigravityGenerateRequest::new(
            &AntigravityUpstreamTarget::default_subscription(),
            SecretString::new("access-secret").unwrap(),
            br#"{"secret":"body"}"#.to_vec(),
        );
        let debug = format!("{request:?}");
        assert!(!debug.contains("access-secret"));
        assert!(!debug.contains("secret\":\"body"));
    }
}
