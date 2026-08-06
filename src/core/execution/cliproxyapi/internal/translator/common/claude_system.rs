// ref: internal/translator/common/claude_system.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::internal::util::claude_attribution::is_claude_code_attribution_system_text;

pub fn claude_message_system_reminder_text(content: &Value) -> Option<String> {
    let parts = claude_system_text_parts(content);
    if parts.is_empty() {
        return None;
    }
    let text = parts.join("\n");
    if text.trim().is_empty() {
        return None;
    }
    Some(format!("<system-reminder>\n{text}\n</system-reminder>"))
}

fn claude_system_text_parts(content: &Value) -> Vec<String> {
    match content {
        Value::String(text)
            if !text.is_empty() && !is_claude_code_attribution_system_text(text) =>
        {
            vec![text.clone()]
        }
        Value::Array(parts) => parts
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .filter(|text| !text.is_empty() && !is_claude_code_attribution_system_text(text))
            .map(ToOwned::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::claude_message_system_reminder_text;
    use serde_json::json;

    #[test]
    fn wraps_text_parts_and_filters_attribution() {
        assert_eq!(
            claude_message_system_reminder_text(&json!([
                {"type":"text","text":"first"},
                {"type":"image","source":{}},
                {"type":"text","text":"x-anthropic-billing-header: cc_version=2"},
                {"type":"text","text":"second"}
            ]))
            .as_deref(),
            Some("<system-reminder>\nfirst\nsecond\n</system-reminder>")
        );
        assert!(claude_message_system_reminder_text(&json!(null)).is_none());
    }
}
