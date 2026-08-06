// ref: internal/api/handlers/management/api_tools.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use url::Url;

const MAX_API_TOOL_BODY_BYTES: usize = 4 * 1024 * 1024;
const MAX_API_TOOL_HEADERS: usize = 128;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagementApiToolRequest {
    #[serde(default, alias = "authIndex", alias = "AuthIndex")]
    pub auth_index: String,
    pub method: String,
    pub url: String,
    #[serde(default, alias = "header")]
    pub headers: BTreeMap<String, String>,
    #[serde(default, alias = "data")]
    pub body: String,
}

impl fmt::Debug for ManagementApiToolRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementApiToolRequest")
            .field("auth_index", &self.auth_index)
            .field("method", &self.method)
            .field("url", &self.url)
            .field("headers", &"[REDACTED]")
            .field("body_bytes", &self.body.len())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ManagementApiToolResponse {
    pub status_code: u16,
    pub headers: BTreeMap<String, Vec<String>>,
    pub body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagementApiToolError {
    InvalidBody,
    MissingMethod,
    UnsupportedMethod,
    MissingUrl,
    InvalidUrl,
    InsecureUrl,
    RequestTooLarge,
    TooManyHeaders,
    AuthorityRejected,
    CredentialUnavailable,
    ExecutionUnavailable,
}

pub trait ManagementApiCallExecutor: Send + Sync {
    /// Executes a validated request through the host's isolated network and
    /// credential authority. `$TOKEN$` replacement and proxy selection occur
    /// behind this boundary; the management layer never receives secret data.
    fn execute(
        &self,
        request: &ManagementApiToolRequest,
    ) -> Result<ManagementApiToolResponse, ManagementApiToolError>;
}

pub struct ManagementApiTools {
    executor: Arc<dyn ManagementApiCallExecutor>,
}

impl ManagementApiTools {
    #[must_use]
    pub fn new(executor: Arc<dyn ManagementApiCallExecutor>) -> Self {
        Self { executor }
    }

    pub fn execute_json(
        &self,
        body: &[u8],
    ) -> Result<ManagementApiToolResponse, ManagementApiToolError> {
        if body.len() > MAX_API_TOOL_BODY_BYTES {
            return Err(ManagementApiToolError::RequestTooLarge);
        }
        let request =
            serde_json::from_slice(body).map_err(|_| ManagementApiToolError::InvalidBody)?;
        self.execute(request)
    }

    pub fn execute(
        &self,
        mut request: ManagementApiToolRequest,
    ) -> Result<ManagementApiToolResponse, ManagementApiToolError> {
        request.auth_index = request.auth_index.trim().to_owned();
        request.method = request.method.trim().to_ascii_uppercase();
        request.url = request.url.trim().to_owned();
        if request.method.is_empty() {
            return Err(ManagementApiToolError::MissingMethod);
        }
        if !matches!(
            request.method.as_str(),
            "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "HEAD" | "OPTIONS"
        ) {
            return Err(ManagementApiToolError::UnsupportedMethod);
        }
        if request.url.is_empty() {
            return Err(ManagementApiToolError::MissingUrl);
        }
        let parsed = Url::parse(&request.url).map_err(|_| ManagementApiToolError::InvalidUrl)?;
        if parsed.host_str().is_none() {
            return Err(ManagementApiToolError::InvalidUrl);
        }
        if parsed.scheme() != "https" {
            return Err(ManagementApiToolError::InsecureUrl);
        }
        if request.headers.len() > MAX_API_TOOL_HEADERS {
            return Err(ManagementApiToolError::TooManyHeaders);
        }
        if request.body.len() > MAX_API_TOOL_BODY_BYTES {
            return Err(ManagementApiToolError::RequestTooLarge);
        }
        self.executor.execute(&request)
    }
}

impl fmt::Debug for ManagementApiTools {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementApiTools")
            .finish_non_exhaustive()
    }
}
