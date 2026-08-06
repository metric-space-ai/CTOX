// ref: internal/runtime/executor/codex_executor.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::SystemTime;

use serde_json::Value;
use tokio::sync::mpsc;
use zeroize::Zeroizing;

use crate::internal::auth::codex::SecretString;
use crate::internal::translator::common::SseDecoder;

use super::codex_executor_reasoning::{CodexReasoningReplayCache, CodexReasoningReplayScope};
use super::codex_executor_request::CodexIdentityConfuseState;
use super::codex_executor_stream::{CodexSseTerminalStream, CodexStreamTerminal};

pub const CODEX_USER_AGENT: &str =
    "codex-tui/0.146.0 (Mac OS 26.5.0; arm64) iTerm.app/3.6.10 (codex-tui; 0.146.0)";
pub const CODEX_ORIGINATOR: &str = "codex-tui";
pub const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexUpstreamTarget {
    base_url: String,
}

impl CodexUpstreamTarget {
    pub fn new(base_url: impl Into<String>) -> Result<Self, CodexTargetError> {
        let base_url = base_url.into();
        let parsed = url::Url::parse(base_url.trim()).map_err(|_| CodexTargetError::InvalidUrl)?;
        if !matches!(parsed.scheme(), "http" | "https")
            || parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some()
            || parsed.fragment().is_some()
        {
            return Err(CodexTargetError::InvalidUrl);
        }
        Ok(Self {
            base_url: parsed.as_str().trim_end_matches('/').to_owned(),
        })
    }

    pub fn default_subscription() -> Self {
        Self {
            base_url: DEFAULT_CODEX_BASE_URL.to_owned(),
        }
    }

    pub fn responses_url(&self) -> String {
        format!("{}/responses", self.base_url)
    }

    pub fn compact_url(&self) -> String {
        format!("{}/responses/compact", self.base_url)
    }

    pub fn direct_image_url(&self, endpoint_path: &str) -> Result<String, CodexTargetError> {
        if !matches!(endpoint_path, "/images/generations" | "/images/edits") {
            return Err(CodexTargetError::InvalidUrl);
        }
        Ok(format!("{}{}", self.base_url, endpoint_path))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexTargetError {
    InvalidUrl,
}

impl fmt::Display for CodexTargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Codex upstream URL is invalid")
    }
}

impl std::error::Error for CodexTargetError {}

pub struct CodexResponsesRequest {
    url: String,
    authorization: SecretString,
    account_id: String,
    session_id: Option<String>,
    body: Zeroizing<Vec<u8>>,
    stream: bool,
    headers: BTreeMap<String, String>,
    content_type: String,
    passthrough_stream: bool,
}

struct CodexRequestWireOptions {
    stream: bool,
    content_type: String,
    passthrough_stream: bool,
}

impl CodexResponsesRequest {
    pub fn new(
        target: &CodexUpstreamTarget,
        access_token: SecretString,
        account_id: impl Into<String>,
        session_id: Option<String>,
        body: Vec<u8>,
    ) -> Self {
        Self::for_url(
            target.responses_url(),
            access_token,
            account_id,
            session_id,
            body,
            CodexRequestWireOptions {
                stream: true,
                content_type: "application/json".to_owned(),
                passthrough_stream: false,
            },
        )
    }

    pub fn compact(
        target: &CodexUpstreamTarget,
        access_token: SecretString,
        account_id: impl Into<String>,
        session_id: Option<String>,
        body: Vec<u8>,
    ) -> Self {
        Self::for_url(
            target.compact_url(),
            access_token,
            account_id,
            session_id,
            body,
            CodexRequestWireOptions {
                stream: false,
                content_type: "application/json".to_owned(),
                passthrough_stream: false,
            },
        )
    }

    fn for_url(
        url: String,
        access_token: SecretString,
        account_id: impl Into<String>,
        session_id: Option<String>,
        body: Vec<u8>,
        wire: CodexRequestWireOptions,
    ) -> Self {
        Self {
            url,
            authorization: access_token,
            account_id: account_id.into(),
            session_id,
            body: Zeroizing::new(body),
            stream: wire.stream,
            headers: BTreeMap::new(),
            content_type: wire.content_type,
            passthrough_stream: wire.passthrough_stream,
        }
    }

    pub fn direct_image(
        target: &CodexUpstreamTarget,
        endpoint_path: &str,
        access_token: SecretString,
        account_id: impl Into<String>,
        body: Vec<u8>,
        content_type: impl Into<String>,
        stream: bool,
    ) -> Result<Self, CodexTargetError> {
        Ok(Self::for_url(
            target.direct_image_url(endpoint_path)?,
            access_token,
            account_id,
            None,
            body,
            CodexRequestWireOptions {
                stream,
                content_type: content_type.into(),
                passthrough_stream: stream,
            },
        ))
    }

    pub fn with_headers(mut self, headers: BTreeMap<String, String>) -> Self {
        self.headers = headers;
        self
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn access_token(&self) -> &SecretString {
        &self.authorization
    }

    pub fn account_id(&self) -> &str {
        &self.account_id
    }

    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    pub fn body(&self) -> &[u8] {
        &self.body
    }

    pub fn stream(&self) -> bool {
        self.stream
    }

    pub fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    pub fn content_type(&self) -> &str {
        &self.content_type
    }

    pub fn passthrough_stream(&self) -> bool {
        self.passthrough_stream
    }
}

impl fmt::Debug for CodexResponsesRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodexResponsesRequest")
            .field("url", &self.url)
            .field("authorization", &"Bearer [REDACTED]")
            .field("account_id", &"[REDACTED]")
            .field(
                "session_id",
                &self.session_id.as_ref().map(|_| "[REDACTED]"),
            )
            .field("body", &"[REDACTED]")
            .field("stream", &self.stream)
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .finish()
    }
}

pub struct CodexResponsesResponse {
    status: u16,
    retry_after: Option<String>,
    body: Zeroizing<Vec<u8>>,
}

impl CodexResponsesResponse {
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

impl fmt::Debug for CodexResponsesResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodexResponsesResponse")
            .field("status", &self.status)
            .field("retry_after", &self.retry_after)
            .field("body", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexResponsesTransportFailure {
    Timeout,
    Connect,
    Protocol,
    Cancelled,
}

pub trait CodexResponsesTransport: Send + Sync {
    fn execute<'a>(
        &'a self,
        request: &'a CodexResponsesRequest,
        timeout: std::time::Duration,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<CodexResponsesResponse, CodexResponsesTransportFailure>>
                + Send
                + 'a,
        >,
    >;
}

pub trait CodexResponsesStreamingTransport: Send + Sync {
    fn execute_stream<'a>(
        &'a self,
        request: &'a CodexResponsesRequest,
        timeout: std::time::Duration,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<CodexResponsesStreamResponse, CodexResponsesTransportFailure>,
                > + Send
                + 'a,
        >,
    >;
}

const MAX_STREAM_BOOTSTRAP_BYTES: usize = 1024 * 1024;

/// HTTP status plus a cancel-on-drop Codex Responses body stream.
pub struct CodexResponsesStreamResponse {
    status: u16,
    retry_after: Option<String>,
    bootstrap: Option<Vec<u8>>,
    chunks: mpsc::Receiver<Result<Vec<u8>, CodexResponsesTransportFailure>>,
    terminal: CodexSseTerminalStream,
    replay: Option<(Arc<CodexReasoningReplayCache>, CodexReasoningReplayScope)>,
    identity: Option<CodexIdentityConfuseState>,
    eof_reported: bool,
    terminal_required: bool,
}

impl CodexResponsesStreamResponse {
    pub fn new(
        status: u16,
        retry_after: Option<String>,
        chunks: mpsc::Receiver<Result<Vec<u8>, CodexResponsesTransportFailure>>,
    ) -> Self {
        Self {
            status,
            retry_after,
            bootstrap: None,
            chunks,
            terminal: CodexSseTerminalStream::default(),
            replay: None,
            identity: None,
            eof_reported: false,
            terminal_required: true,
        }
    }

    pub fn status(&self) -> u16 {
        self.status
    }

    pub fn retry_after(&self) -> Option<&str> {
        self.retry_after.as_deref()
    }

    pub async fn next_chunk(&mut self) -> Option<Result<Vec<u8>, CodexResponsesTransportFailure>> {
        let next = if let Some(bytes) = self.bootstrap.take() {
            Some(Ok(bytes))
        } else {
            self.chunks.recv().await
        };
        match next {
            Some(Ok(bytes)) => {
                let bytes = self
                    .identity
                    .as_ref()
                    .map(|state| state.expose_response(&bytes))
                    .unwrap_or(bytes);
                if let CodexStreamTerminal::Completed(completed) =
                    self.terminal.push(&bytes, SystemTime::now())
                {
                    if let Some((cache, scope)) = &self.replay {
                        cache.commit_completed(scope.clone(), &completed);
                    }
                }
                Some(Ok(bytes))
            }
            Some(Err(error)) => Some(Err(error)),
            None if self.terminal_required && !self.terminal.committed() && !self.eof_reported => {
                self.eof_reported = true;
                Some(Err(CodexResponsesTransportFailure::Protocol))
            }
            None => None,
        }
    }

    pub fn attach_reasoning_replay(
        &mut self,
        cache: Arc<CodexReasoningReplayCache>,
        scope: CodexReasoningReplayScope,
    ) {
        self.replay = Some((cache, scope));
    }

    pub fn attach_identity(&mut self, identity: CodexIdentityConfuseState) {
        self.identity = Some(identity);
    }

    pub fn set_passthrough(&mut self) {
        self.terminal_required = false;
    }

    pub fn terminal_committed(&self) -> bool {
        self.terminal.committed()
    }

    pub(crate) async fn bootstrap_first_response_event(
        &mut self,
    ) -> Result<(), CodexResponsesTransportFailure> {
        if self.bootstrap.is_some() || !(200..300).contains(&self.status) {
            return Ok(());
        }
        let mut pending = Vec::new();
        loop {
            match self.chunks.recv().await {
                Some(Ok(chunk)) => {
                    pending.extend_from_slice(&chunk);
                    if pending.len() > MAX_STREAM_BOOTSTRAP_BYTES {
                        return Err(CodexResponsesTransportFailure::Protocol);
                    }
                    match inspect_response_bootstrap(&pending, false) {
                        ResponseBootstrapInspection::Pending => {}
                        ResponseBootstrapInspection::Started => {
                            self.bootstrap = Some(pending);
                            return Ok(());
                        }
                        ResponseBootstrapInspection::Failed => {
                            return Err(CodexResponsesTransportFailure::Protocol)
                        }
                    }
                }
                Some(Err(error)) => return Err(error),
                None => {
                    return match inspect_response_bootstrap(&pending, true) {
                        ResponseBootstrapInspection::Started => {
                            self.bootstrap = Some(pending);
                            Ok(())
                        }
                        ResponseBootstrapInspection::Pending
                        | ResponseBootstrapInspection::Failed => {
                            Err(CodexResponsesTransportFailure::Protocol)
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn synthetic(status: u16) -> Self {
        let (_sender, receiver) = mpsc::channel(1);
        Self::new(status, None, receiver)
    }
}

impl fmt::Debug for CodexResponsesStreamResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodexResponsesStreamResponse")
            .field("status", &self.status)
            .field("retry_after", &self.retry_after)
            .field("bootstrap", &self.bootstrap.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseBootstrapInspection {
    Pending,
    Started,
    Failed,
}

fn inspect_response_bootstrap(bytes: &[u8], eof: bool) -> ResponseBootstrapInspection {
    let mut decoder = SseDecoder::new();
    let mut events = decoder.push(bytes);
    if eof {
        events.extend(decoder.finish());
    }
    for event in events {
        let Ok(value) = serde_json::from_slice::<Value>(&event.data) else {
            continue;
        };
        match value.get("type").and_then(Value::as_str) {
            Some("error" | "response.failed") => return ResponseBootstrapInspection::Failed,
            Some(kind) if kind.starts_with("response.") => {
                return ResponseBootstrapInspection::Started
            }
            _ => {}
        }
    }
    ResponseBootstrapInspection::Pending
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_rejects_credentials_query_and_non_http_schemes() {
        for invalid in [
            "file:///tmp/codex",
            "https://user:secret@example.com/codex",
            "https://example.com/codex?token=x",
        ] {
            assert_eq!(
                CodexUpstreamTarget::new(invalid),
                Err(CodexTargetError::InvalidUrl)
            );
        }
        assert_eq!(
            CodexUpstreamTarget::default_subscription().responses_url(),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn request_debug_redacts_token_and_body() {
        let request = CodexResponsesRequest::new(
            &CodexUpstreamTarget::default_subscription(),
            SecretString::new("access-do-not-leak").unwrap(),
            "acct-1",
            Some("session-1".to_owned()),
            br#"{"secret":"body-do-not-leak"}"#.to_vec(),
        );
        let rendered = format!("{request:?}");
        assert!(!rendered.contains("access-do-not-leak"));
        assert!(!rendered.contains("body-do-not-leak"));
    }
}
