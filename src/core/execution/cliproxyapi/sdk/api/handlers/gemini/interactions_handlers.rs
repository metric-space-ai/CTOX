// ref: sdk/api/handlers/gemini/interactions_handlers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

use serde::Deserialize;
use serde_json::value::RawValue;

use super::GeminiHandlerError;

pub const INTERACTIONS_AGENT_AUTH_SELECTION_MODEL: &str = "gemini-2.5-flash";

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct InteractionsRequestTarget {
    pub model: String,
    pub agent: String,
    pub stream: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct InteractionsExecutionRequest {
    pub entry_protocol: String,
    pub exit_protocol: String,
    pub forced_provider: String,
    pub auth_selection_model: String,
    pub model: String,
    pub stream: bool,
    pub body: Vec<u8>,
    pub alt: String,
}

#[derive(Deserialize)]
struct RawInteractionsTarget<'a> {
    #[serde(default, borrow)]
    model: Option<&'a RawValue>,
    #[serde(default, borrow)]
    agent: Option<&'a RawValue>,
    #[serde(default, borrow)]
    stream: Option<&'a RawValue>,
}

pub fn parse_interactions_request_target(
    raw_json: &[u8],
) -> Result<InteractionsRequestTarget, GeminiHandlerError> {
    let raw: RawInteractionsTarget<'_> =
        serde_json::from_slice(raw_json).map_err(|_| invalid_request("invalid JSON body"))?;
    let model = optional_string(raw.model, "model")?;
    let agent = optional_string(raw.agent, "agent")?;
    if model.is_empty() == agent.is_empty() {
        return Err(invalid_request(
            "request requires exactly one of model or agent",
        ));
    }
    let stream = raw.stream.map_or(Ok(false), |stream| {
        serde_json::from_str::<bool>(stream.get())
            .map_err(|_| invalid_request("stream must be a boolean"))
    })?;
    Ok(InteractionsRequestTarget {
        model,
        agent,
        stream,
    })
}

#[must_use]
pub fn prepare_interactions_execution_target(
    raw_json: &[u8],
    target: &InteractionsRequestTarget,
) -> (String, Vec<u8>) {
    if !target.agent.is_empty() {
        return (target.agent.clone(), raw_json.to_vec());
    }
    let model = normalize_gemini_model_resource_name(&target.model);
    if model == target.model {
        return (model, raw_json.to_vec());
    }
    let body = replace_top_level_model(raw_json, &model).unwrap_or_else(|| raw_json.to_vec());
    (model, body)
}

#[must_use]
pub fn build_interactions_execution_request(
    target: &InteractionsRequestTarget,
    model: impl Into<String>,
    body: Vec<u8>,
    alt: impl Into<String>,
) -> InteractionsExecutionRequest {
    let agent = !target.agent.is_empty();
    InteractionsExecutionRequest {
        entry_protocol: "interactions".to_owned(),
        exit_protocol: "interactions".to_owned(),
        forced_provider: if agent { "gemini-interactions" } else { "" }.to_owned(),
        auth_selection_model: if agent {
            INTERACTIONS_AGENT_AUTH_SELECTION_MODEL
        } else {
            ""
        }
        .to_owned(),
        model: model.into(),
        stream: target.stream,
        body,
        alt: alt.into(),
    }
}

/// Frames one Interactions chunk without changing an already framed upstream
/// SSE event. A complete blank-line delimiter is appended exactly once.
#[must_use]
pub fn frame_interactions_sse_chunk(chunk: &[u8]) -> Vec<u8> {
    if chunk.is_empty() {
        return Vec::new();
    }
    let trimmed = trim_ascii_start(chunk);
    let mut framed = if trimmed.starts_with(b"event:") || trimmed.starts_with(b"data:") {
        chunk.to_vec()
    } else {
        let mut framed = b"data: ".to_vec();
        framed.extend_from_slice(chunk);
        framed
    };
    if !framed.ends_with(b"\n\n") {
        framed.extend_from_slice(b"\n\n");
    }
    framed
}

fn normalize_gemini_model_resource_name(model: &str) -> String {
    let model = model.trim();
    model
        .strip_prefix("models/")
        .filter(|model| !model.is_empty())
        .unwrap_or(model)
        .to_owned()
}

fn optional_string(raw: Option<&RawValue>, field: &str) -> Result<String, GeminiHandlerError> {
    let Some(raw) = raw else {
        return Ok(String::new());
    };
    serde_json::from_str::<String>(raw.get())
        .map(|value| value.trim().to_owned())
        .map_err(|_| invalid_request(&format!("{field} must be a string")))
}

fn replace_top_level_model(raw_json: &[u8], model: &str) -> Option<Vec<u8>> {
    let old = serde_json::to_string(&format!("models/{model}")).ok()?;
    let new = serde_json::to_string(model).ok()?;
    let text = std::str::from_utf8(raw_json).ok()?;
    let (start, end) = locate_top_level_model_string(text, &old)?;
    let mut output = Vec::with_capacity(raw_json.len() + new.len().saturating_sub(old.len()));
    output.extend_from_slice(&raw_json[..start]);
    output.extend_from_slice(new.as_bytes());
    output.extend_from_slice(&raw_json[end..]);
    Some(output)
}

fn locate_top_level_model_string(text: &str, expected: &str) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut index = 0;
    let mut depth = 0_u32;
    while index < bytes.len() {
        match bytes[index] {
            b'{' | b'[' => {
                depth = depth.saturating_add(1);
                index += 1;
            }
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                index += 1;
            }
            b'"' => {
                let end = string_end(bytes, index)?;
                if depth == 1 && &text[index..=end] == "\"model\"" {
                    let mut colon = end + 1;
                    while bytes.get(colon).is_some_and(u8::is_ascii_whitespace) {
                        colon += 1;
                    }
                    if bytes.get(colon) != Some(&b':') {
                        index = end + 1;
                        continue;
                    }
                    let mut value = colon + 1;
                    while bytes.get(value).is_some_and(u8::is_ascii_whitespace) {
                        value += 1;
                    }
                    if text.get(value..value + expected.len()) == Some(expected) {
                        return Some((value, value + expected.len()));
                    }
                }
                index = end + 1;
            }
            _ => index += 1,
        }
    }
    None
}

fn string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut escaped = false;
    for (offset, byte) in bytes.get(start + 1..)?.iter().enumerate() {
        if escaped {
            escaped = false;
        } else if *byte == b'\\' {
            escaped = true;
        } else if *byte == b'"' {
            return Some(start + 1 + offset);
        }
    }
    None
}

fn trim_ascii_start(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    bytes
}

fn invalid_request(message: &str) -> GeminiHandlerError {
    GeminiHandlerError {
        status: 400,
        message: message.to_owned(),
        error_type: "invalid_request_error".to_owned(),
    }
}

impl fmt::Display for InteractionsExecutionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.entry_protocol, self.model)
    }
}

#[cfg(test)]
#[path = "interactions_handlers_test.rs"]
mod interactions_handlers_test;
