// ref: internal/thinking/text.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

/// Extracts thinking text from the simple, wrapped, or Gemini-style shapes.
pub fn get_thinking_text(part: &Value) -> &str {
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        return text;
    }

    let Some(thinking) = part.get("thinking") else {
        return "";
    };
    if let Some(text) = thinking.as_str() {
        return text;
    }
    if let Some(text) = thinking.get("text").and_then(Value::as_str) {
        return text;
    }
    thinking
        .get("thinking")
        .and_then(Value::as_str)
        .unwrap_or("")
}
