// ref: internal/pluginhost/http_bridge.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: network authority is injected as a pre-scoped HostHttpClient
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use url::Url;

use crate::sdk::pluginapi::{
    HostHttpClient, HttpRequest, HttpResponse, HttpStreamResponse, PluginExecutionError,
};

pub const MAX_PLUGIN_HTTP_BODY_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone)]
pub struct HostHttpBridge {
    authority: Arc<dyn HostHttpClient>,
}

impl std::fmt::Debug for HostHttpBridge {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostHttpBridge")
            .finish_non_exhaustive()
    }
}

impl HostHttpBridge {
    pub fn new(authority: Arc<dyn HostHttpClient>) -> Self {
        Self { authority }
    }

    pub async fn execute(
        &self,
        request: HttpRequest,
    ) -> Result<HttpResponse, PluginExecutionError> {
        validate_request(&request).map_err(|error| Arc::new(error) as PluginExecutionError)?;
        self.authority.execute(request).await
    }

    pub async fn execute_stream(
        &self,
        request: HttpRequest,
    ) -> Result<HttpStreamResponse, PluginExecutionError> {
        validate_request(&request).map_err(|error| Arc::new(error) as PluginExecutionError)?;
        self.authority.execute_stream(request).await
    }
}

fn validate_request(request: &HttpRequest) -> Result<(), HttpBridgeError> {
    if request.method.is_empty()
        || request.method.len() > 32
        || !request
            .method
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte == b'-')
        || request.body.len() > MAX_PLUGIN_HTTP_BODY_BYTES
    {
        return Err(HttpBridgeError::InvalidRequest);
    }
    let url = Url::parse(&request.url).map_err(|_| HttpBridgeError::InvalidUrl)?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.host_str().is_none()
    {
        return Err(HttpBridgeError::InvalidUrl);
    }
    if request.headers.iter().any(|(name, values)| {
        name.is_empty()
            || name.bytes().any(|byte| byte <= b' ' || byte == b':')
            || values.iter().any(|value| value.contains(['\r', '\n']))
    }) {
        return Err(HttpBridgeError::InvalidHeaders);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpBridgeError {
    InvalidRequest,
    InvalidUrl,
    InvalidHeaders,
}

impl std::fmt::Display for HttpBridgeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRequest => "plugin HTTP request is invalid",
            Self::InvalidUrl => "plugin HTTP URL is invalid",
            Self::InvalidHeaders => "plugin HTTP headers are invalid",
        })
    }
}

impl std::error::Error for HttpBridgeError {}
