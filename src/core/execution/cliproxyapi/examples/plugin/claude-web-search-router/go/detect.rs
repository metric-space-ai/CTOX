// ref: examples/plugin/claude-web-search-router/go/detect.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only
use serde_json::Value;
const TYPES: [&str; 2] = ["web_search_20250305", "web_search_20260209"];
pub fn is_claude_source(source: &str) -> bool {
    matches!(
        source.trim().to_ascii_lowercase().as_str(),
        "claude" | "anthropic"
    )
}
fn tools(body: &Value) -> &[Value] {
    body.get("tools")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}
pub fn has_web_search(body: &Value) -> bool {
    tools(body).iter().any(|t| {
        t.get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| TYPES.contains(&kind))
    })
}
pub fn only_web_search(body: &Value) -> bool {
    !tools(body).is_empty()
        && tools(body).iter().all(|t| {
            t.get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| TYPES.contains(&kind))
        })
}
fn message_text(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return text.to_owned();
    }
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter(|b| b.get("type") == Some(&Value::String("text".into())))
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}
pub fn query(body: &Value) -> String {
    let messages = body.get("messages").and_then(Value::as_array);
    let Some(messages) = messages else {
        return String::new();
    };
    const PREFIX: &str = "perform a web search for the query:";
    for message in messages {
        if message.get("role").and_then(Value::as_str) == Some("user") {
            let text = message_text(message.get("content").unwrap_or(&Value::Null));
            if text.to_ascii_lowercase().starts_with(PREFIX) {
                return text[PREFIX.len()..].trim().to_owned();
            }
        }
    }
    messages
        .iter()
        .rev()
        .find_map(|m| {
            let text = message_text(m.get("content")?);
            (!text.trim().is_empty()).then(|| text.trim().to_owned())
        })
        .unwrap_or_default()
}
pub fn max_uses(body: &Value, default: usize) -> usize {
    tools(body)
        .iter()
        .find(|t| {
            t.get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| TYPES.contains(&kind))
        })
        .and_then(|t| t.get("max_uses"))
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .filter(|v| *v > 0)
        .unwrap_or(if default == 0 { 5 } else { default })
}
pub fn is_builtin_web_search(body: &Value, require_only: bool) -> bool {
    has_web_search(body)
        && (!require_only || only_web_search(body))
        && (only_web_search(body) || !query(body).is_empty())
}
