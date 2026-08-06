// ref: internal/translator/gemini/claude/gemini_claude_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};

use crate::internal::translator::antigravity::claude::{
    claude_token_count, convert_antigravity_response_to_claude_non_stream,
    convert_antigravity_response_to_claude_stream, AntigravityClaudeStreamState,
};

use super::gemini_claude_request::translated_request_envelope;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GeminiToClaudeStreamState {
    inner: AntigravityClaudeStreamState,
}

pub fn convert_gemini_response_to_claude_stream(
    original_request: &[u8],
    translated_request: &[u8],
    raw: &[u8],
    state: &mut GeminiToClaudeStreamState,
) -> Vec<Vec<u8>> {
    let translated = translated_request_envelope(translated_request);
    let wrapped = wrap_response(raw);
    convert_antigravity_response_to_claude_stream(
        original_request,
        &translated,
        &wrapped,
        &mut state.inner,
        "",
    )
}

pub fn convert_gemini_response_to_claude_non_stream(
    original_request: &[u8],
    translated_request: &[u8],
    raw: &[u8],
) -> Vec<u8> {
    let translated = translated_request_envelope(translated_request);
    let wrapped = wrap_response(raw);
    convert_antigravity_response_to_claude_non_stream(original_request, &translated, &wrapped, "")
}

pub fn gemini_claude_token_count(count: i64) -> Vec<u8> {
    claude_token_count(count)
}

fn wrap_response(raw: &[u8]) -> Vec<u8> {
    if raw == b"[DONE]" {
        return raw.to_vec();
    }
    let payload = raw.strip_prefix(b"data:").map(trim_ascii).unwrap_or(raw);
    let response = serde_json::from_slice::<Value>(payload).unwrap_or(Value::Null);
    serde_json::to_vec(&json!({"response":response})).unwrap_or_default()
}

fn trim_ascii(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}
