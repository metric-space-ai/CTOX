// ref: internal/runtime/executor/helps/claude_input_tokens.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;

use serde_json::value::RawValue;
use tiktoken_rs::o200k_base_singleton;

use crate::sdk::translator::{claude, Format};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaudeInputTokenError {
    InvalidJson,
}

impl fmt::Display for ClaudeInputTokenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Claude input-token request is invalid JSON")
    }
}

impl std::error::Error for ClaudeInputTokenError {}

/// Estimates Claude request input tokens with the pinned upstream O200k
/// tokenizer and the same semantic segment selection.
pub fn count_claude_input_tokens(payload: &[u8]) -> Result<i64, ClaudeInputTokenError> {
    let segments = collect_claude_input_token_segments(payload)?;
    if segments.is_empty() {
        return Ok(0);
    }
    Ok(o200k_base_singleton().count_ordinary(&segments.join("\n")) as i64)
}

/// Request-scoped one-shot patch state for Claude `message_start` usage.
///
/// The tokenizer is process-shared by `tiktoken-rs`; request bytes and the
/// handled bit remain local, replacing upstream's codec-bearing mutable state.
pub struct ClaudeInputTokenState {
    upstream_format: Format,
    response_format: Format,
    original_request: Vec<u8>,
    handled: bool,
    failure_sink: Option<Arc<dyn ClaudeInputTokenFailureSink>>,
}

impl fmt::Debug for ClaudeInputTokenState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ClaudeInputTokenState")
            .field("upstream_format", &self.upstream_format)
            .field("response_format", &self.response_format)
            .field("original_request_len", &self.original_request.len())
            .field("handled", &self.handled)
            .field("has_failure_sink", &self.failure_sink.is_some())
            .finish()
    }
}

pub trait ClaudeInputTokenFailureSink: Send + Sync {
    fn estimation_failed(
        &self,
        upstream_format: &Format,
        response_format: &Format,
        error: ClaudeInputTokenError,
    );
}

impl ClaudeInputTokenState {
    #[must_use]
    pub fn new(
        source_format: &Format,
        upstream_format: &Format,
        response_format: &Format,
        original_request: &[u8],
    ) -> Self {
        let claude = claude();
        let enabled =
            source_format == &claude && upstream_format != &claude && response_format == &claude;
        Self {
            upstream_format: upstream_format.clone(),
            response_format: response_format.clone(),
            original_request: original_request.to_vec(),
            handled: !enabled,
            failure_sink: None,
        }
    }

    #[must_use]
    pub fn with_failure_sink(mut self, sink: Arc<dyn ClaudeInputTokenFailureSink>) -> Self {
        self.failure_sink = Some(sink);
        self
    }

    #[must_use]
    pub fn handled(&self) -> bool {
        self.handled
    }

    /// Patches the first complete Claude message-start event and leaves all
    /// surrounding bytes, including CRLF and SSE spacing, unchanged.
    pub fn apply(&mut self, mut chunks: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        if self.handled {
            return chunks;
        }
        for chunk in &mut chunks {
            let (updated, found) = self.apply_chunk(chunk);
            if found {
                self.handled = true;
                *chunk = updated;
                break;
            }
        }
        chunks
    }

    fn apply_chunk(&self, chunk: &[u8]) -> (Vec<u8>, bool) {
        let mut line_start = 0;
        while line_start < chunk.len() {
            let relative_end = chunk[line_start..].iter().position(|byte| *byte == b'\n');
            let line_end = relative_end.map_or(chunk.len(), |index| line_start + index);
            let content_end = if line_end > line_start && chunk[line_end - 1] == b'\r' {
                line_end - 1
            } else {
                line_end
            };
            let line = &chunk[line_start..content_end];
            let leading = line
                .iter()
                .position(|byte| !matches!(byte, b' ' | b'\t'))
                .unwrap_or(line.len());
            if line[leading..].starts_with(b"data:") {
                let mut payload_offset = leading + 5;
                while payload_offset < line.len() && matches!(line[payload_offset], b' ' | b'\t') {
                    payload_offset += 1;
                }
                let mut payload_end = line.len();
                while payload_end > payload_offset && matches!(line[payload_end - 1], b' ' | b'\t')
                {
                    payload_end -= 1;
                }
                let payload = &line[payload_offset..payload_end];
                if is_message_start(payload) {
                    let existing = message_start_input_tokens(payload);
                    if existing.is_some_and(|tokens| tokens != 0) {
                        return (chunk.to_vec(), true);
                    }
                    let count = match count_claude_input_tokens(&self.original_request) {
                        Ok(count) => count,
                        Err(error) => {
                            if let Some(sink) = &self.failure_sink {
                                sink.estimation_failed(
                                    &self.upstream_format,
                                    &self.response_format,
                                    error,
                                );
                            }
                            return (chunk.to_vec(), true);
                        }
                    };
                    if count == 0 {
                        return (chunk.to_vec(), true);
                    }
                    let Some(updated_payload) = set_input_tokens_surgically(payload, count) else {
                        return (chunk.to_vec(), true);
                    };
                    let payload_start = line_start + payload_offset;
                    let payload_stop = line_start + payload_end;
                    let mut updated = Vec::with_capacity(
                        chunk.len() + updated_payload.len().saturating_sub(payload.len()),
                    );
                    updated.extend_from_slice(&chunk[..payload_start]);
                    updated.extend_from_slice(&updated_payload);
                    updated.extend_from_slice(&chunk[payload_stop..]);
                    return (updated, true);
                }
            }
            if line_end == chunk.len() {
                break;
            }
            line_start = line_end + 1;
        }
        (chunk.to_vec(), false)
    }

    #[must_use]
    pub fn route(&self) -> (&Format, &Format) {
        (&self.upstream_format, &self.response_format)
    }
}

fn is_message_start(payload: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(payload)
        .ok()
        .and_then(|value| {
            value
                .get("type")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .as_deref()
        == Some("message_start")
}

fn message_start_input_tokens(payload: &[u8]) -> Option<i64> {
    serde_json::from_slice::<serde_json::Value>(payload)
        .ok()?
        .pointer("/message/usage/input_tokens")?
        .as_i64()
}

fn set_input_tokens_surgically(payload: &[u8], count: i64) -> Option<Vec<u8>> {
    let usage_key = find_json_key(payload, b"\"usage\"")?;
    let colon = payload[usage_key..].iter().position(|byte| *byte == b':')? + usage_key;
    let open = payload[colon + 1..].iter().position(|byte| *byte == b'{')? + colon + 1;
    let close = matching_object_close(payload, open)?;
    let input_key =
        find_json_key(&payload[open + 1..close], b"\"input_tokens\"").map(|index| open + 1 + index);
    if let Some(key_start) = input_key {
        let colon = payload[key_start..].iter().position(|byte| *byte == b':')? + key_start;
        let mut value_start = colon + 1;
        while value_start < payload.len() && payload[value_start].is_ascii_whitespace() {
            value_start += 1;
        }
        let mut value_end = value_start;
        if payload.get(value_end) == Some(&b'-') {
            value_end += 1;
        }
        while value_end < payload.len() && payload[value_end].is_ascii_digit() {
            value_end += 1;
        }
        if value_end == value_start {
            return None;
        }
        let replacement = count.to_string();
        let mut out = Vec::with_capacity(payload.len() + replacement.len());
        out.extend_from_slice(&payload[..value_start]);
        out.extend_from_slice(replacement.as_bytes());
        out.extend_from_slice(&payload[value_end..]);
        return Some(out);
    }

    let empty = payload[open + 1..close].iter().all(u8::is_ascii_whitespace);
    let insertion = if empty {
        format!("\"input_tokens\":{count}")
    } else {
        format!("\"input_tokens\":{count},")
    };
    let mut out = Vec::with_capacity(payload.len() + insertion.len());
    out.extend_from_slice(&payload[..open + 1]);
    out.extend_from_slice(insertion.as_bytes());
    out.extend_from_slice(&payload[open + 1..]);
    Some(out)
}

fn find_json_key(payload: &[u8], key: &[u8]) -> Option<usize> {
    payload.windows(key.len()).position(|window| window == key)
}

fn matching_object_close(payload: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0_u32;
    let mut quoted = false;
    let mut escaped = false;
    for (index, byte) in payload.iter().copied().enumerate().skip(open) {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        match byte {
            b'"' => quoted = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

pub(crate) fn collect_claude_input_token_segments(
    payload: &[u8],
) -> Result<Vec<String>, ClaudeInputTokenError> {
    if payload.iter().all(u8::is_ascii_whitespace) {
        return Ok(Vec::new());
    }
    let _: Box<RawValue> =
        serde_json::from_slice(payload).map_err(|_| ClaudeInputTokenError::InvalidJson)?;
    let document = std::str::from_utf8(payload).map_err(|_| ClaudeInputTokenError::InvalidJson)?;
    let root = gjson::parse(document);
    let mut segments = Vec::with_capacity(32);
    collect_system(root.get("system"), &mut segments);
    collect_messages(root.get("messages"), &mut segments);
    collect_tools(root.get("tools"), &mut segments);
    collect_tool_choice(root.get("tool_choice"), &mut segments);
    Ok(segments)
}

fn collect_system(system: gjson::Value<'_>, segments: &mut Vec<String>) {
    if system.kind() == gjson::Kind::String {
        append_string(segments, system.str());
        return;
    }
    if system.kind() != gjson::Kind::Array {
        return;
    }
    system.each(|_, part| {
        if part.kind() == gjson::Kind::String {
            append_string(segments, part.str());
        } else if part.get("type").str() == "text" {
            append_string(segments, part.get("text").str());
        }
        true
    });
}

fn collect_messages(messages: gjson::Value<'_>, segments: &mut Vec<String>) {
    if messages.kind() != gjson::Kind::Array {
        return;
    }
    messages.each(|_, message| {
        append_string(segments, message.get("role").str());
        collect_content(message.get("content"), segments);
        true
    });
}

fn collect_content(content: gjson::Value<'_>, segments: &mut Vec<String>) {
    if !content.exists() {
        return;
    }
    if content.kind() == gjson::Kind::String {
        append_string(segments, content.str());
        return;
    }
    if content.kind() == gjson::Kind::Array {
        content.each(|_, part| {
            collect_content(part, segments);
            true
        });
        return;
    }
    if content.kind() != gjson::Kind::Object {
        return;
    }

    match content.get("type").str() {
        "text" => append_string(segments, content.get("text").str()),
        "thinking" => append_string(segments, content.get("thinking").str()),
        "document" => collect_document(content, segments),
        "tool_use" | "server_tool_use" | "mcp_tool_use" => {
            append_string(segments, content.get("id").str());
            append_string(segments, content.get("name").str());
            append_json(segments, content.get("input"));
        }
        "tool_result"
        | "mcp_tool_result"
        | "web_search_tool_result"
        | "web_fetch_tool_result"
        | "code_execution_tool_result"
        | "bash_code_execution_tool_result"
        | "text_editor_code_execution_tool_result" => {
            append_string(segments, content.get("tool_use_id").str());
            append_string(segments, content.get("tool_call_id").str());
            collect_content(content.get("content"), segments);
        }
        "web_search_result" | "search_result" => {
            let source = content.get("source");
            if source.kind() == gjson::Kind::String {
                append_string(segments, source.str());
            }
            append_string(segments, content.get("title").str());
            append_string(segments, content.get("url").str());
            append_string(segments, content.get("page_age").str());
            collect_content(content.get("content"), segments);
        }
        "web_fetch_result" => {
            append_string(segments, content.get("url").str());
            append_string(segments, content.get("retrieved_at").str());
            collect_content(content.get("content"), segments);
        }
        "code_execution_result"
        | "bash_code_execution_result"
        | "text_editor_code_execution_result" => {
            append_string(segments, content.get("stdout").str());
            append_string(segments, content.get("stderr").str());
            append_string(segments, content.get("return_code").str());
            collect_content(content.get("content"), segments);
            collect_content(content.get("output"), segments);
        }
        "tool_reference" => append_string(segments, content.get("tool_name").str()),
        "image" | "input_audio" | "audio" | "video" | "redacted_thinking" => {}
        "" => append_json(segments, content),
        _ => append_string(segments, content.get("text").str()),
    }
}

fn collect_document(document: gjson::Value<'_>, segments: &mut Vec<String>) {
    let source = document.get("source");
    if source.get("type").str() != "text" {
        return;
    }
    append_string(segments, document.get("title").str());
    append_string(segments, document.get("context").str());
    append_string(segments, source.get("data").str());
    append_string(segments, source.get("content").str());
}

fn collect_tools(tools: gjson::Value<'_>, segments: &mut Vec<String>) {
    if tools.kind() != gjson::Kind::Array {
        return;
    }
    tools.each(|_, tool| {
        append_string(segments, tool.get("type").str());
        append_string(segments, tool.get("name").str());
        append_string(segments, tool.get("description").str());
        append_json(segments, tool.get("input_schema"));
        true
    });
}

fn collect_tool_choice(tool_choice: gjson::Value<'_>, segments: &mut Vec<String>) {
    if !tool_choice.exists() {
        return;
    }
    if tool_choice.kind() == gjson::Kind::String {
        append_string(segments, tool_choice.str());
        return;
    }
    append_string(segments, tool_choice.get("type").str());
    append_string(segments, tool_choice.get("name").str());
}

fn append_string(segments: &mut Vec<String>, value: &str) {
    let value = value.trim();
    if !value.is_empty() {
        segments.push(value.to_owned());
    }
}

fn append_json(segments: &mut Vec<String>, value: gjson::Value<'_>) {
    if !value.exists() {
        return;
    }
    if value.kind() == gjson::Kind::String {
        append_string(segments, value.str());
        return;
    }
    let raw = value.json().trim();
    if raw.is_empty() {
        return;
    }
    append_string(segments, compact_json(raw).as_str());
}

fn compact_json(raw: &str) -> String {
    let mut compact = String::with_capacity(raw.len());
    let mut quoted = false;
    let mut escaped = false;
    for character in raw.chars() {
        if quoted {
            compact.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                quoted = false;
            }
        } else if character == '"' {
            quoted = true;
            compact.push(character);
        } else if !character.is_ascii_whitespace() {
            compact.push(character);
        }
    }
    compact
}
