// ref: sdk/api/handlers/openai_responses_stream_error.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct OpenAiResponsesStreamErrorChunk {
    #[serde(rename = "type")]
    chunk_type: &'static str,
    code: String,
    message: String,
    sequence_number: i64,
}

fn openai_responses_stream_error_code(status: u16) -> &'static str {
    match status {
        401 => "invalid_api_key",
        403 => "insufficient_quota",
        429 => "rate_limit_exceeded",
        404 => "model_not_found",
        408 => "request_timeout",
        500.. => "internal_server_error",
        400.. => "invalid_request_error",
        _ => "unknown_error",
    }
}

// ref: sdk/api/handlers/openai_responses_stream_error.go:43-116
pub fn build_openai_responses_stream_error_chunk(
    status: u16,
    error_text: &str,
    sequence_number: i64,
) -> Vec<u8> {
    let status = if status == 0 { 500 } else { status };
    let mut sequence_number = sequence_number.max(0);
    let mut message = error_text.trim().to_owned();
    if message.is_empty() {
        message = status_text(status).to_owned();
    }
    let mut code = openai_responses_stream_error_code(status).to_owned();

    if let Ok(Value::Object(payload)) = serde_json::from_str::<Value>(error_text.trim()) {
        if payload.get("type").and_then(Value::as_str) == Some("error") {
            if let Some(value) = non_empty_string(payload.get("message")) {
                message = value.to_owned();
            }
            if let Some(value) = display_json_code(payload.get("code")) {
                code = value;
            }
            if sequence_number == 0 {
                if let Some(value) = json_integer(payload.get("sequence_number")) {
                    sequence_number = value;
                }
            }
        }
        if let Some(Value::Object(error)) = payload.get("error") {
            if let Some(value) = non_empty_string(error.get("message")) {
                message = value.to_owned();
            }
            if let Some(value) = display_json_code(error.get("code")) {
                code = value;
            }
        }
    }
    if code.trim().is_empty() {
        code = "unknown_error".to_owned();
    }

    let chunk = OpenAiResponsesStreamErrorChunk {
        chunk_type: "error",
        code,
        message,
        sequence_number,
    };
    serde_json::to_vec(&chunk).unwrap_or_else(|_| {
        br#"{"type":"error","code":"internal_server_error","message":"internal error","sequence_number":0}"#.to_vec()
    })
}

pub fn build_openai_responses_stream_error_event(
    status: u16,
    error_text: &str,
    sequence_number: i64,
) -> Vec<u8> {
    let payload = build_openai_responses_stream_error_chunk(status, error_text, sequence_number);
    let mut event = Vec::with_capacity(payload.len() + 8);
    event.extend_from_slice(b"data: ");
    event.extend_from_slice(&payload);
    event.extend_from_slice(b"\n\n");
    event
}

fn non_empty_string(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn display_json_code(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::Null => None,
        Value::String(value) => {
            let value = value.trim();
            (!value.is_empty()).then(|| value.to_owned())
        }
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        value => Some(value.to_string()),
    }
}

fn json_integer(value: Option<&Value>) -> Option<i64> {
    let value = value?;
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
        .or_else(|| value.as_f64().map(|value| value as i64))
}

fn status_text(status: u16) -> &'static str {
    match status {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        408 => "Request Timeout",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ if status >= 500 => "Internal Server Error",
        _ => "Unknown Error",
    }
}

#[cfg(test)]
#[path = "openai_responses_stream_error_test.rs"]
mod openai_responses_stream_error_test;
