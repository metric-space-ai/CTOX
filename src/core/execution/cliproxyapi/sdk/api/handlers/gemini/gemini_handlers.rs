// ref: sdk/api/handlers/gemini/gemini_handlers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

use serde_json::{Map, Value};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GeminiAction {
    GenerateContent { model: String },
    StreamGenerateContent { model: String },
    CountTokens { model: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeminiHandlerError {
    pub status: u16,
    pub message: String,
    pub error_type: String,
}

impl fmt::Display for GeminiHandlerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for GeminiHandlerError {}

impl GeminiAction {
    pub fn parse(action: &str, request_path: &str) -> Result<Self, GeminiHandlerError> {
        let action = action.trim().trim_start_matches('/');
        let Some((model, method)) = action.split_once(':') else {
            return Err(not_found(request_path));
        };
        let model = model.trim();
        if model.is_empty() || method.is_empty() || method.contains(':') {
            return Err(not_found(request_path));
        }
        match method {
            "generateContent" => Ok(Self::GenerateContent {
                model: model.to_owned(),
            }),
            "streamGenerateContent" => Ok(Self::StreamGenerateContent {
                model: model.to_owned(),
            }),
            "countTokens" => Ok(Self::CountTokens {
                model: model.to_owned(),
            }),
            _ => Err(not_found(request_path)),
        }
    }
}

/// Normalizes an injected registry snapshot into the Gemini model-list wire
/// shape without mutating the registry-owned catalog.
#[must_use]
pub fn normalize_gemini_models(models: &[Map<String, Value>]) -> Vec<Map<String, Value>> {
    models
        .iter()
        .cloned()
        .map(|mut model| {
            let name = model
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            if !name.is_empty() {
                let resource_name = if name.starts_with("models/") {
                    name.clone()
                } else {
                    format!("models/{name}")
                };
                model.insert("name".to_owned(), Value::String(resource_name));
                if blank_string(&model, "displayName") {
                    model.insert("displayName".to_owned(), Value::String(name.clone()));
                }
                if blank_string(&model, "description") {
                    model.insert("description".to_owned(), Value::String(name));
                }
            }
            model
                .entry("supportedGenerationMethods".to_owned())
                .or_insert_with(|| serde_json::json!(["generateContent"]));
            model
        })
        .collect()
}

fn blank_string(model: &Map<String, Value>, key: &str) -> bool {
    model
        .get(key)
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
}

fn not_found(path: &str) -> GeminiHandlerError {
    GeminiHandlerError {
        status: 404,
        message: format!("{} not found.", path.trim()),
        error_type: "invalid_request_error".to_owned(),
    }
}

#[cfg(test)]
#[path = "gemini_models_display_name_test.rs"]
mod gemini_models_display_name_test;
