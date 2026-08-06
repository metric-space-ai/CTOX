// ref: internal/client/claude/models/models.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const CLAUDE_DD_MODEL_PREFIX: &str = "claude-fable-5-dd-";

pub type ClaudeModel = Map<String, Value>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClaudeModelsResponse {
    pub data: Vec<ClaudeModel>,
    pub has_more: bool,
    pub first_id: String,
    pub last_id: String,
}

// ref: internal/client/claude/models/models.go:12-49
pub fn build_response(
    available_models: &[ClaudeModel],
    disable_cloaking: bool,
) -> ClaudeModelsResponse {
    let mut models = available_models.to_vec();
    if !disable_cloaking {
        for model in &mut models {
            if let Some(Value::String(id)) = model.get_mut("id") {
                *id = ensure_claude_model_id_prefix(id);
            }
        }
    }
    models.sort_by(|left, right| {
        string_field(left, "display_name")
            .cmp(string_field(right, "display_name"))
            .then_with(|| string_field(left, "id").cmp(string_field(right, "id")))
    });

    let first_id = models
        .first()
        .and_then(|model| model.get("id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let last_id = models
        .last()
        .and_then(|model| model.get("id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    ClaudeModelsResponse {
        data: models,
        has_more: false,
        first_id,
        last_id,
    }
}

// ref: internal/client/claude/models/models.go:51-59
pub fn ensure_claude_model_id_prefix(id: &str) -> String {
    if id.is_empty() || id.starts_with("claude-") {
        return id.to_owned();
    }
    format!("{CLAUDE_DD_MODEL_PREFIX}{}", reverse_model_id(id))
}

// ref: internal/client/claude/models/models.go:61-80
pub fn resolve_claude_model_id_prefix(id: &str) -> String {
    if id.is_empty() {
        return String::new();
    }
    let (base, suffix) = split_model_thinking_suffix(id);
    let Some(encoded) = base.strip_prefix(CLAUDE_DD_MODEL_PREFIX) else {
        return id.to_owned();
    };
    if encoded.is_empty() {
        return id.to_owned();
    }
    let mut resolved = reverse_model_id(encoded);
    if let Some(suffix) = suffix {
        resolved.push('(');
        resolved.push_str(suffix);
        resolved.push(')');
    }
    resolved
}

fn string_field<'a>(model: &'a ClaudeModel, field: &str) -> &'a str {
    model.get(field).and_then(Value::as_str).unwrap_or_default()
}

fn split_model_thinking_suffix(model: &str) -> (&str, Option<&str>) {
    let Some(last_open) = model.rfind('(') else {
        return (model, None);
    };
    let Some(without_close) = model.strip_suffix(')') else {
        return (model, None);
    };
    (&model[..last_open], Some(&without_close[last_open + 1..]))
}

fn reverse_model_id(id: &str) -> String {
    id.chars().rev().collect()
}
