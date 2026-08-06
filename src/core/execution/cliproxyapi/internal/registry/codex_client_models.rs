// ref: internal/registry/codex_client_models.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::{
    collections::HashSet,
    fmt,
    sync::{RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use serde::Deserialize;
use serde_json::{Map, Number, Value};

const DEFAULT_TEMPLATE: &str = "gpt-5.5";

#[derive(Debug, Deserialize)]
struct CatalogPayload {
    models: Vec<Map<String, Value>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodexClientModelsError(String);

impl CodexClientModelsError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for CodexClientModelsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CodexClientModelsError {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CodexClientModelsSnapshot {
    pub data: Vec<u8>,
    pub revision: u64,
}

#[derive(Debug, Default)]
struct CatalogState {
    data: Vec<u8>,
    revision: u64,
}

/// Host-owned catalog store. Upstream uses a package global; CTOX injects this
/// owner so independent harnesses cannot mutate each other's model catalogs.
#[derive(Debug, Default)]
pub struct CodexClientModelsStore {
    state: RwLock<CatalogState>,
}

impl CodexClientModelsStore {
    pub fn new(data: &[u8], source: &str) -> Result<Self, CodexClientModelsError> {
        let store = Self::default();
        store.load(data, source)?;
        Ok(store)
    }

    pub fn snapshot(&self) -> CodexClientModelsSnapshot {
        let state = read_state(&self.state);
        CodexClientModelsSnapshot {
            data: state.data.clone(),
            revision: state.revision,
        }
    }

    pub fn load(&self, data: &[u8], source: &str) -> Result<bool, CodexClientModelsError> {
        validate_codex_client_models_json(data)
            .map_err(|error| CodexClientModelsError::new(format!("{source}: {error}")))?;
        let mut state = write_state(&self.state);
        if state.data == data {
            return Ok(false);
        }
        state.data = data.to_vec();
        state.revision = state.revision.saturating_add(1);
        Ok(true)
    }
}

pub fn validate_codex_client_models_json(data: &[u8]) -> Result<(), CodexClientModelsError> {
    let payload: CatalogPayload = serde_json::from_slice(data).map_err(|error| {
        CodexClientModelsError::new(format!("decode Codex client model catalog: {error}"))
    })?;
    if payload.models.is_empty() {
        return Err(CodexClientModelsError::new(
            "Codex client model catalog has no models",
        ));
    }

    let mut seen = HashSet::with_capacity(payload.models.len());
    for (index, model) in payload.models.iter().enumerate() {
        let slug = required_string(model, "slug").map_err(|error| {
            CodexClientModelsError::new(format!(
                "Codex client model catalog models[{index}]: {error}"
            ))
        })?;
        if !seen.insert(slug.to_owned()) {
            return Err(CodexClientModelsError::new(format!(
                "Codex client model catalog contains duplicate slug {slug:?}"
            )));
        }
        validate_model(model).map_err(|error| {
            CodexClientModelsError::new(format!(
                "Codex client model catalog model {slug:?}: {error}"
            ))
        })?;
    }
    if !seen.contains(DEFAULT_TEMPLATE) {
        return Err(CodexClientModelsError::new(format!(
            "Codex client model catalog is missing default template {DEFAULT_TEMPLATE:?}"
        )));
    }
    Ok(())
}

fn validate_model(model: &Map<String, Value>) -> Result<(), CodexClientModelsError> {
    for field in [
        "display_name",
        "description",
        "base_instructions",
        "minimal_client_version",
        "visibility",
        "default_reasoning_level",
    ] {
        required_string(model, field)?;
    }

    let context_window = required_integer(model, "context_window", true)?;
    let max_context_window = required_integer(model, "max_context_window", true)?;
    if context_window > max_context_window {
        return Err(CodexClientModelsError::new(format!(
            "context_window {context_window} exceeds max_context_window {max_context_window}"
        )));
    }
    required_integer(model, "priority", false)?;

    let levels = model
        .get("supported_reasoning_levels")
        .and_then(Value::as_array)
        .filter(|levels| !levels.is_empty())
        .ok_or_else(|| {
            CodexClientModelsError::new(
                "field \"supported_reasoning_levels\" must be a non-empty array",
            )
        })?;
    let mut seen = HashSet::with_capacity(levels.len());
    for (index, raw_level) in levels.iter().enumerate() {
        let level = raw_level.as_object().ok_or_else(|| {
            CodexClientModelsError::new(format!(
                "field \"supported_reasoning_levels\" entry {index} must be an object"
            ))
        })?;
        let effort = required_string(level, "effort").map_err(|error| {
            CodexClientModelsError::new(format!(
                "field \"supported_reasoning_levels\" entry {index}: {error}"
            ))
        })?;
        if !seen.insert(effort) {
            return Err(CodexClientModelsError::new(format!(
                "field \"supported_reasoning_levels\" contains duplicate effort {effort:?}"
            )));
        }
    }
    let default_level = required_string(model, "default_reasoning_level")?;
    if !seen.contains(default_level) {
        return Err(CodexClientModelsError::new(format!(
            "default_reasoning_level {default_level:?} is not listed in supported_reasoning_levels"
        )));
    }
    Ok(())
}

fn required_string<'a>(
    model: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a str, CodexClientModelsError> {
    model
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            CodexClientModelsError::new(format!("field {field:?} must be a non-empty string"))
        })
}

fn required_integer(
    model: &Map<String, Value>,
    field: &str,
    positive: bool,
) -> Result<i64, CodexClientModelsError> {
    let value = model
        .get(field)
        .and_then(Value::as_number)
        .and_then(number_to_i64)
        .ok_or_else(|| {
            CodexClientModelsError::new(format!("field {field:?} must be an integer"))
        })?;
    if positive && value <= 0 {
        return Err(CodexClientModelsError::new(format!(
            "field {field:?} must be positive"
        )));
    }
    if !positive && value < 0 {
        return Err(CodexClientModelsError::new(format!(
            "field {field:?} must not be negative"
        )));
    }
    Ok(value)
}

fn number_to_i64(number: &Number) -> Option<i64> {
    number.as_i64().or_else(|| {
        number
            .as_u64()
            .and_then(|value| i64::try_from(value).ok())
            .or_else(|| {
                number.as_f64().and_then(|value| {
                    (value.is_finite()
                        && value.fract() == 0.0
                        && value >= i64::MIN as f64
                        && value <= i64::MAX as f64)
                        .then_some(value as i64)
                })
            })
    })
}

fn read_state(lock: &RwLock<CatalogState>) -> RwLockReadGuard<'_, CatalogState> {
    lock.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn write_state(lock: &RwLock<CatalogState>) -> RwLockWriteGuard<'_, CatalogState> {
    lock.write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
