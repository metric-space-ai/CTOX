// ref: internal/signature/claude.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{is_valid_claude_thinking_signature_with_options, ClaudeSignatureValidationOptions};

pub fn strip_invalid_claude_thinking_blocks(
    payload: &[u8],
    options: ClaudeSignatureValidationOptions,
) -> Vec<u8> {
    strip_invalid(payload, options, false)
}

pub fn strip_invalid_claude_thinking_blocks_and_empty_messages(
    payload: &[u8],
    options: ClaudeSignatureValidationOptions,
) -> Vec<u8> {
    strip_invalid(payload, options, true)
}

fn strip_invalid(
    payload: &[u8],
    options: ClaudeSignatureValidationOptions,
    drop_empty_messages: bool,
) -> Vec<u8> {
    let Ok(mut root) = serde_json::from_slice::<Value>(payload) else {
        return payload.to_vec();
    };
    let Some(messages) = root.get_mut("messages").and_then(Value::as_array_mut) else {
        return payload.to_vec();
    };
    let mut modified = false;
    messages.retain_mut(|message| {
        let Some(content) = message.get_mut("content").and_then(Value::as_array_mut) else {
            return true;
        };
        let previous_len = content.len();
        content.retain(|part| {
            part.get("type").and_then(Value::as_str) != Some("thinking")
                || !should_strip(part, options)
        });
        if content.len() == previous_len {
            return true;
        }
        modified = true;
        !drop_empty_messages || !content.is_empty()
    });
    if !modified {
        payload.to_vec()
    } else {
        serde_json::to_vec(&root).unwrap_or_else(|_| payload.to_vec())
    }
}

fn should_strip(part: &Value, options: ClaudeSignatureValidationOptions) -> bool {
    if options.allow_empty_signature_with_empty_text && is_empty_placeholder(part) {
        return false;
    }
    let signature = part
        .get("signature")
        .and_then(Value::as_str)
        .unwrap_or_default();
    !is_valid_claude_thinking_signature_with_options(signature, options)
}

fn is_empty_placeholder(part: &Value) -> bool {
    let signature_empty = part
        .get("signature")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .is_empty();
    signature_empty && claude_thinking_text(part).trim().is_empty()
}

fn claude_thinking_text(part: &Value) -> &str {
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        return text;
    }
    let Some(thinking) = part.get("thinking") else {
        return "";
    };
    if let Some(text) = thinking.as_str() {
        return text;
    }
    thinking
        .get("text")
        .or_else(|| thinking.get("thinking"))
        .and_then(Value::as_str)
        .unwrap_or_default()
}
