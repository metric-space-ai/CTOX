// ref: sdk/api/handlers/handlers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::internal::config::SdkConfig;
use crate::internal::thinking::extract_reasoning_effort;
use crate::sdk::cliproxy::executor::ExecutionMetadata;

const DEFAULT_STREAMING_KEEP_ALIVE_SECONDS: i32 = 0;
const DEFAULT_STREAMING_BOOTSTRAP_RETRIES: i32 = 0;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub code: String,
}

/// Builds the OpenAI-compatible fallback error envelope. Valid upstream JSON
/// is preserved as JSON instead of being nested in a second error object.
#[must_use]
pub fn build_error_response_body(status: u16, error_text: &str) -> Vec<u8> {
    let status = if status == 0 { 500 } else { status };
    let error_text = if error_text.trim().is_empty() {
        status_text(status)
    } else {
        error_text
    };
    let trimmed = error_text.trim();
    if serde_json::from_str::<Value>(trimmed).is_ok() {
        return trimmed.as_bytes().to_vec();
    }

    let (error_type, code) = match status {
        401 => ("authentication_error", "invalid_api_key"),
        403 => ("permission_error", "insufficient_quota"),
        429 => ("rate_limit_error", "rate_limit_exceeded"),
        404 => ("invalid_request_error", "model_not_found"),
        500..=u16::MAX => ("server_error", "internal_server_error"),
        _ => ("invalid_request_error", ""),
    };
    let response = ErrorResponse {
        error: ErrorDetail {
            message: error_text.to_owned(),
            error_type: error_type.to_owned(),
            code: code.to_owned(),
        },
    };
    serde_json::to_vec(&response).unwrap_or_else(|_| {
        br#"{"error":{"message":"Internal Server Error","type":"server_error","code":"internal_server_error"}}"#.to_vec()
    })
}

#[must_use]
pub fn streaming_keep_alive_interval(config: Option<&SdkConfig>) -> Duration {
    positive_seconds(
        config.map_or(DEFAULT_STREAMING_KEEP_ALIVE_SECONDS, |config| {
            config.streaming.keepalive_seconds
        }),
    )
}

#[must_use]
pub fn non_streaming_keep_alive_interval(config: Option<&SdkConfig>) -> Duration {
    positive_seconds(config.map_or(0, |config| config.nonstream_keepalive_interval))
}

#[must_use]
pub fn streaming_bootstrap_retries(config: Option<&SdkConfig>) -> usize {
    config
        .map_or(DEFAULT_STREAMING_BOOTSTRAP_RETRIES, |config| {
            config.streaming.bootstrap_retries
        })
        .max(0) as usize
}

#[must_use]
pub fn passthrough_headers_enabled(config: Option<&SdkConfig>) -> bool {
    config.is_some_and(|config| config.passthrough_headers)
}

pub fn set_reasoning_effort_metadata(
    metadata: &mut ExecutionMetadata,
    handler_type: &str,
    model: &str,
    raw_json: &[u8],
) {
    let effort = extract_reasoning_effort(raw_json, handler_type, model);
    metadata.reasoning_effort = (!effort.is_empty()).then_some(effort);
}

pub fn set_service_tier_metadata(metadata: &mut ExecutionMetadata, raw_json: &[u8]) {
    let tier = serde_json::from_slice::<Value>(raw_json)
        .ok()
        .and_then(|document| document.get("service_tier")?.as_str().map(str::to_owned))
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "auto".to_owned());
    metadata.service_tier = Some(tier);
}

pub fn set_generate_metadata(metadata: &mut ExecutionMetadata, raw_json: &[u8]) {
    metadata.generate = Some(
        serde_json::from_slice::<Value>(raw_json)
            .ok()
            .and_then(|document| document.get("generate")?.as_bool())
            .unwrap_or(true),
    );
}

fn positive_seconds(seconds: i32) -> Duration {
    u64::try_from(seconds)
        .ok()
        .filter(|seconds| *seconds > 0)
        .map_or(Duration::ZERO, Duration::from_secs)
}

fn status_text(status: u16) -> &'static str {
    match status {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        408 => "Request Timeout",
        409 => "Conflict",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Internal Server Error",
    }
}

#[cfg(test)]
#[path = "handlers_metadata_test.rs"]
mod handlers_metadata_test;
