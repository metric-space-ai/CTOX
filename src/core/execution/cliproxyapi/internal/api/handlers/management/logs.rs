// ref: internal/api/handlers/management/logs.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

const DEFAULT_LOG_LIMIT: usize = 200;
const MAX_LOG_LIMIT: usize = 10_000;

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagementLogQuery {
    #[serde(default)]
    pub after: Option<i64>,
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ManagementLogPage {
    pub lines: Vec<String>,
    pub line_count: usize,
    pub latest_timestamp: i64,
    pub next_cursor: String,
    pub cursor_reset: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ManagementRequestErrorLog {
    pub name: String,
    pub size: u64,
    pub modified: i64,
}

#[derive(Clone, PartialEq, Eq)]
pub struct ManagementLogAttachment {
    pub name: String,
    pub content_type: String,
    pub bytes: Vec<u8>,
}

impl fmt::Debug for ManagementLogAttachment {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementLogAttachment")
            .field("name", &self.name)
            .field("content_type", &self.content_type)
            .field("bytes", &format_args!("[{} BYTES]", self.bytes.len()))
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagementLogError {
    InvalidLimit,
    InvalidCursor,
    InvalidRequestId,
    InvalidFileName,
    NotFound,
    Disabled,
    StoreUnavailable,
}

pub trait ManagementLogStore: Send + Sync {
    fn read(&self, query: &ManagementLogQuery) -> Result<ManagementLogPage, ManagementLogError>;
    fn clear(&self) -> Result<usize, ManagementLogError>;
    fn list_request_errors(&self) -> Result<Vec<ManagementRequestErrorLog>, ManagementLogError>;
    fn request_log_by_id(
        &self,
        request_id: &str,
    ) -> Result<ManagementLogAttachment, ManagementLogError>;
    fn request_error_log(&self, name: &str) -> Result<ManagementLogAttachment, ManagementLogError>;
}

pub struct ManagementLogs {
    store: Arc<dyn ManagementLogStore>,
}

impl ManagementLogs {
    #[must_use]
    pub fn new(store: Arc<dyn ManagementLogStore>) -> Self {
        Self { store }
    }

    pub fn get_logs(
        &self,
        mut query: ManagementLogQuery,
    ) -> Result<ManagementLogPage, ManagementLogError> {
        query.limit = Some(query.limit.unwrap_or(DEFAULT_LOG_LIMIT));
        if query.limit == Some(0) || query.limit.is_some_and(|limit| limit > MAX_LOG_LIMIT) {
            return Err(ManagementLogError::InvalidLimit);
        }
        if query
            .cursor
            .as_ref()
            .is_some_and(|cursor| cursor.trim().is_empty())
        {
            return Err(ManagementLogError::InvalidCursor);
        }
        self.store.read(&query)
    }

    pub fn delete_logs(&self) -> Result<usize, ManagementLogError> {
        self.store.clear()
    }

    pub fn request_error_logs(&self) -> Result<Vec<ManagementRequestErrorLog>, ManagementLogError> {
        let mut logs = self.store.list_request_errors()?;
        logs.sort_by(|left, right| {
            right
                .modified
                .cmp(&left.modified)
                .then_with(|| left.name.cmp(&right.name))
        });
        Ok(logs)
    }

    pub fn request_log_by_id(
        &self,
        request_id: &str,
    ) -> Result<ManagementLogAttachment, ManagementLogError> {
        let request_id = request_id.trim();
        if !is_safe_identifier(request_id) {
            return Err(ManagementLogError::InvalidRequestId);
        }
        self.store.request_log_by_id(request_id)
    }

    pub fn request_error_log(
        &self,
        name: &str,
    ) -> Result<ManagementLogAttachment, ManagementLogError> {
        let name = name.trim();
        if !is_safe_error_log_name(name) {
            return Err(ManagementLogError::InvalidFileName);
        }
        self.store.request_error_log(name)
    }
}

impl fmt::Debug for ManagementLogs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementLogs")
            .finish_non_exhaustive()
    }
}

fn is_safe_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn is_safe_error_log_name(name: &str) -> bool {
    name.starts_with("error-")
        && name.ends_with(".log")
        && name.len() <= 512
        && !name.contains(['/', '\\'])
        && name != "error-.log"
}
