// Origin: CTOX
// License: AGPL-3.0-only

use std::collections::BTreeMap;

use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExampleRegistration {
    pub id: &'static str,
    pub capabilities: &'static [&'static str],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExampleReply {
    pub result: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExampleError {
    pub code: &'static str,
    pub message: String,
}

pub type ExampleResult = Result<ExampleReply, ExampleError>;

pub fn registration(
    id: &'static str,
    capabilities: &'static [&'static str],
) -> ExampleRegistration {
    ExampleRegistration { id, capabilities }
}

pub fn reply(result: Value) -> ExampleResult {
    Ok(ExampleReply { result })
}

pub fn unknown(method: &str) -> ExampleResult {
    Err(ExampleError {
        code: "unknown_method",
        message: format!("unknown method: {method}"),
    })
}

pub fn tagged_body(key: &str, value: &str) -> Value {
    json!({ "Body": serde_json::to_vec(&json!({key: value})).expect("JSON value serializes") })
}

/// Narrow host boundary used by callback examples. Implementations are
/// injected by the caller; examples never inspect ambient configuration,
/// launch a process, or obtain credentials themselves.
pub trait ExampleHost {
    fn call(&self, method: &str, payload: Value) -> Result<Value, ExampleError>;
}

#[derive(Default)]
pub struct RecordingHost {
    replies: BTreeMap<String, Value>,
}

impl RecordingHost {
    pub fn with_reply(mut self, method: &str, reply: Value) -> Self {
        self.replies.insert(method.to_owned(), reply);
        self
    }
}

impl ExampleHost for RecordingHost {
    fn call(&self, method: &str, _payload: Value) -> Result<Value, ExampleError> {
        self.replies
            .get(method)
            .cloned()
            .ok_or_else(|| ExampleError {
                code: "host_method_unavailable",
                message: method.to_owned(),
            })
    }
}
