// ref: internal/runtime/executor/codex_executor_terminal.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::time::{Duration, SystemTime};

use serde_json::Value;

pub const CODEX_INCOMPLETE_STREAM_MESSAGE: &str =
    "stream error: stream disconnected before completion: stream closed before response.completed";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexTerminalError {
    pub status: u16,
    pub code: Option<String>,
    pub error_type: Option<String>,
    pub retry_after: Option<Duration>,
    pub context_length: bool,
}

impl fmt::Display for CodexTerminalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Codex terminal stream error ({})", self.status)
    }
}

impl std::error::Error for CodexTerminalError {}

#[derive(Debug, Clone, PartialEq)]
pub enum CodexTerminalEvent {
    Continue,
    Completed(Vec<u8>),
    Failed(CodexTerminalError),
}

/// Request-owned terminal accumulator. Output items are committed only when a
/// `response.completed`/`response.incomplete` event arrives.
#[derive(Debug, Default)]
pub struct CodexTerminalAccumulator {
    indexed: BTreeMap<i64, Value>,
    fallback: Vec<Value>,
    committed: bool,
}

impl CodexTerminalAccumulator {
    pub fn ingest(&mut self, event_data: &[u8], now: SystemTime) -> CodexTerminalEvent {
        if self.committed {
            return CodexTerminalEvent::Failed(CodexTerminalError {
                status: 502,
                code: Some("event_after_terminal".to_owned()),
                error_type: Some("protocol_error".to_owned()),
                retry_after: None,
                context_length: false,
            });
        }
        let Ok(value) = serde_json::from_slice::<Value>(event_data) else {
            return CodexTerminalEvent::Continue;
        };
        match value.get("type").and_then(Value::as_str) {
            Some("response.output_item.done") => {
                if let Some(item) = value.get("item").cloned() {
                    if let Some(index) = value.get("output_index").and_then(Value::as_i64) {
                        self.indexed.insert(index, item);
                    } else {
                        self.fallback.push(item);
                    }
                }
                CodexTerminalEvent::Continue
            }
            Some("response.completed" | "response.incomplete") => {
                let Some(mut response) = value.get("response").cloned() else {
                    return CodexTerminalEvent::Failed(protocol_error("missing_response"));
                };
                let empty = response
                    .get("output")
                    .and_then(Value::as_array)
                    .is_none_or(Vec::is_empty);
                if empty && (!self.indexed.is_empty() || !self.fallback.is_empty()) {
                    let Some(object) = response.as_object_mut() else {
                        return CodexTerminalEvent::Failed(protocol_error("invalid_response"));
                    };
                    let mut output = self.indexed.values().cloned().collect::<Vec<_>>();
                    output.extend(self.fallback.iter().cloned());
                    object.insert("output".to_owned(), Value::Array(output));
                } else if let Some(output) =
                    response.get_mut("output").and_then(Value::as_array_mut)
                {
                    hydrate_missing_output_item_ids(output, &self.indexed);
                }
                self.committed = true;
                match serde_json::to_vec(&response) {
                    Ok(payload) => CodexTerminalEvent::Completed(payload),
                    Err(_) => CodexTerminalEvent::Failed(protocol_error("encode_response")),
                }
            }
            Some("error" | "response.failed") => {
                self.committed = true;
                CodexTerminalEvent::Failed(parse_codex_terminal_error(&value, now))
            }
            _ => CodexTerminalEvent::Continue,
        }
    }

    pub fn finish(self) -> Result<(), CodexIncompleteStreamError> {
        if self.committed {
            Ok(())
        } else {
            Err(CodexIncompleteStreamError)
        }
    }

    pub fn committed(&self) -> bool {
        self.committed
    }
}

fn hydrate_missing_output_item_ids(output: &mut [Value], indexed: &BTreeMap<i64, Value>) {
    for (position, item) in output.iter_mut().enumerate() {
        let missing = item
            .get("id")
            .is_none_or(|id| id.is_null() || id.as_str().is_some_and(|id| id.trim().is_empty()));
        if !missing {
            continue;
        }
        let Ok(index) = i64::try_from(position) else {
            continue;
        };
        let Some(id) = indexed
            .get(&index)
            .and_then(|completed| completed.get("id"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
        else {
            continue;
        };
        if let Some(object) = item.as_object_mut() {
            object.insert("id".to_owned(), Value::String(id.to_owned()));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodexIncompleteStreamError;

impl fmt::Display for CodexIncompleteStreamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(CODEX_INCOMPLETE_STREAM_MESSAGE)
    }
}

impl std::error::Error for CodexIncompleteStreamError {}

pub fn parse_codex_terminal_error(value: &Value, now: SystemTime) -> CodexTerminalError {
    let root = if value.get("type").and_then(Value::as_str) == Some("response.failed") {
        value
            .get("response")
            .and_then(|response| response.get("error"))
            .unwrap_or(value)
    } else {
        value.get("error").unwrap_or(value)
    };
    let code = string_field(root, "code");
    let error_type = string_field(root, "type");
    let status = root
        .get("status")
        .and_then(Value::as_u64)
        .and_then(|value| u16::try_from(value).ok())
        .unwrap_or_else(|| codex_terminal_status(error_type.as_deref(), code.as_deref()));
    let message = root
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    let context_length = code
        .as_deref()
        .is_some_and(|value| matches!(value, "context_length_exceeded" | "max_tokens_exceeded"))
        || message.contains("context length");
    CodexTerminalError {
        status,
        code,
        error_type,
        retry_after: parse_codex_retry_after(status, root, now),
        context_length,
    }
}

pub fn codex_terminal_status(error_type: Option<&str>, code: Option<&str>) -> u16 {
    match (error_type.unwrap_or_default(), code.unwrap_or_default()) {
        ("invalid_request_error" | "bad_request_error", _)
        | (_, "context_length_exceeded" | "invalid_request") => 400,
        ("authentication_error", _) | (_, "invalid_api_key" | "unauthorized") => 401,
        ("permission_error", _) | (_, "forbidden" | "permission_denied") => 403,
        ("not_found_error", _) | (_, "not_found" | "model_not_found") => 404,
        ("rate_limit_error", _) | (_, "rate_limit_exceeded" | "usage_limit_reached") => 429,
        _ => 502,
    }
}

pub fn parse_codex_retry_after(status: u16, error: &Value, now: SystemTime) -> Option<Duration> {
    if status != 429 {
        return None;
    }
    for key in ["retry_after", "retry_after_seconds"] {
        if let Some(seconds) = error.get(key).and_then(value_as_f64) {
            if seconds.is_finite() && seconds >= 0.0 {
                return Some(Duration::from_secs_f64(seconds));
            }
        }
    }
    let reset = error
        .get("resets_at")
        .or_else(|| error.get("reset_at"))
        .and_then(Value::as_i64)?;
    let deadline = SystemTime::UNIX_EPOCH.checked_add(Duration::from_secs(reset.max(0) as u64))?;
    deadline.duration_since(now).ok()
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn value_as_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse().ok())
}

fn protocol_error(code: &str) -> CodexTerminalError {
    CodexTerminalError {
        status: 502,
        code: Some(code.to_owned()),
        error_type: Some("protocol_error".to_owned()),
        retry_after: None,
        context_length: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_commit_reconstructs_output_once() {
        let mut terminal = CodexTerminalAccumulator::default();
        terminal.ingest(
            br#"{"type":"response.output_item.done","output_index":1,"item":{"id":"b"}}"#,
            SystemTime::UNIX_EPOCH,
        );
        terminal.ingest(
            br#"{"type":"response.output_item.done","output_index":0,"item":{"id":"a"}}"#,
            SystemTime::UNIX_EPOCH,
        );
        let CodexTerminalEvent::Completed(payload) = terminal.ingest(
            br#"{"type":"response.completed","response":{"output":[]}}"#,
            SystemTime::UNIX_EPOCH,
        ) else {
            panic!("completion")
        };
        let value: Value = serde_json::from_slice(&payload).unwrap();
        assert_eq!(value["output"][0]["id"], "a");
        assert!(terminal.committed());
    }

    #[test]
    fn errors_are_classified_without_retaining_provider_message() {
        let value = serde_json::json!({"type":"error","error":{"type":"rate_limit_error","code":"usage_limit_reached","message":"secret detail","retry_after":"2.5"}});
        let error = parse_codex_terminal_error(&value, SystemTime::UNIX_EPOCH);
        assert_eq!(error.status, 429);
        assert_eq!(error.retry_after, Some(Duration::from_millis(2500)));
        assert!(!format!("{error:?} {error}").contains("secret detail"));
    }
}
