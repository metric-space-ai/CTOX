// ref: internal/runtime/executor/helps/openai_compat_tool_results.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use crate::internal::thinking::parse_suffix;

use super::super::openai_compat_executor::{OpenAiCompatibility, OpenAiCompatibilityModel};

pub const OPENAI_TOOL_RESULT_IMAGE_OMITTED_TEXT: &str = "[image omitted: unsupported by upstream]";

#[must_use]
pub fn should_normalize_openai_tool_results_for_model(
    compat: Option<&OpenAiCompatibility>,
    upstream_model: &str,
    requested_model: &str,
) -> bool {
    let Some(compat) = compat else {
        return false;
    };
    if let Some(result) = model_excludes_images(&compat.models, upstream_model) {
        return result;
    }
    model_excludes_images(&compat.models, requested_model).unwrap_or(false)
}

fn model_excludes_images(models: &[OpenAiCompatibilityModel], model: &str) -> Option<bool> {
    let model = parse_suffix(model.trim()).model_name;
    if model.is_empty() {
        return None;
    }
    if let Some(candidate) = models
        .iter()
        .find(|candidate| candidate.name.trim().eq_ignore_ascii_case(&model))
    {
        return Some(input_modalities_exclude_images(&candidate.input_modalities));
    }
    let aliases = models
        .iter()
        .filter(|candidate| candidate.alias.trim().eq_ignore_ascii_case(&model))
        .collect::<Vec<_>>();
    (!aliases.is_empty()).then(|| {
        aliases
            .iter()
            .all(|candidate| input_modalities_exclude_images(&candidate.input_modalities))
    })
}

fn input_modalities_exclude_images(modalities: &[String]) -> bool {
    !modalities.is_empty()
        && modalities
            .iter()
            .any(|value| value.trim().eq_ignore_ascii_case("text"))
        && !modalities
            .iter()
            .any(|value| value.trim().eq_ignore_ascii_case("image"))
}

#[must_use]
pub fn normalize_openai_tool_results_text_only(payload: &[u8]) -> Vec<u8> {
    let Ok(mut root) = serde_json::from_slice::<Value>(payload) else {
        return payload.to_vec();
    };
    let Some(messages) = root.get_mut("messages").and_then(Value::as_array_mut) else {
        return payload.to_vec();
    };
    let mut changed = false;
    for message in messages {
        if message.get("role").and_then(Value::as_str) != Some("tool") {
            continue;
        }
        let Some(content) = message.get("content") else {
            continue;
        };
        if content.is_string() {
            continue;
        }
        let flattened = flatten_tool_content(content);
        if let Some(object) = message.as_object_mut() {
            object.insert("content".into(), Value::String(flattened));
            changed = true;
        }
    }
    if changed {
        serde_json::to_vec(&root).unwrap_or_else(|_| payload.to_vec())
    } else {
        payload.to_vec()
    }
}

fn flatten_tool_content(content: &Value) -> String {
    match content {
        Value::String(value) => value.clone(),
        Value::Array(items) => items
            .iter()
            .map(tool_part_text)
            .collect::<Vec<_>>()
            .join("\n\n"),
        Value::Object(_) if is_image_part(content) => {
            OPENAI_TOOL_RESULT_IMAGE_OMITTED_TEXT.to_owned()
        }
        Value::Object(object) => object
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| serde_json::to_string(content).unwrap_or_default()),
        _ => serde_json::to_string(content).unwrap_or_default(),
    }
}

fn tool_part_text(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return text.to_owned();
    }
    if is_image_part(value) {
        return OPENAI_TOOL_RESULT_IMAGE_OMITTED_TEXT.to_owned();
    }
    value
        .get("text")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| serde_json::to_string(value).unwrap_or_default())
}

fn is_image_part(value: &Value) -> bool {
    matches!(
        value
            .get("type")
            .and_then(Value::as_str)
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref(),
        Some("image") | Some("image_url") | Some("input_image")
    ) || value.get("image_url").is_some()
        || value.get("input_image").is_some()
}
