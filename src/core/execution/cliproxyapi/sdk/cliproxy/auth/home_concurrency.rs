// ref: sdk/cliproxy/auth/home_concurrency.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: validates Home-owned admission tuples before registry installation
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::sdk::cliproxy::executionregistry::{
    PendingDispatch, Registry, RegistryError, Scope, ScopeSpec,
};

const MAX_FIELD_BYTES: usize = 256;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HomeConcurrencyTuple {
    pub accounted: bool,
    pub credential_id: String,
    pub model: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HomeConcurrencyError {
    MalformedTuple,
    InvalidResponse,
    IdentityMismatch,
    Registry(RegistryError),
}

impl fmt::Display for HomeConcurrencyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::MalformedTuple => "malformed Home concurrency tuple",
            Self::InvalidResponse => "Home concurrency response is invalid",
            Self::IdentityMismatch => "Home concurrency identity does not match dispatched auth",
            Self::Registry(_) => "Home execution registry is unavailable",
        })
    }
}

impl std::error::Error for HomeConcurrencyError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HomeConcurrencyBusyError {
    message: String,
    retry_after: Option<Duration>,
}

impl HomeConcurrencyBusyError {
    pub fn new(message: &str, retry_after: Duration) -> Self {
        let message = message.trim();
        Self {
            message: if message.is_empty() {
                "credential concurrency limit exceeded".to_owned()
            } else {
                message.to_owned()
            },
            retry_after: (!retry_after.is_zero()).then_some(retry_after),
        }
    }

    pub fn status_code(&self) -> u16 {
        429
    }

    pub fn retry_after(&self) -> Option<Duration> {
        self.retry_after
    }

    pub fn safe_response_headers(&self) -> Vec<(String, String)> {
        self.retry_after
            .map(|duration| {
                let seconds = duration.as_secs() + u64::from(duration.subsec_nanos() != 0);
                vec![("Retry-After".to_owned(), seconds.max(1).to_string())]
            })
            .unwrap_or_default()
    }
}

impl fmt::Display for HomeConcurrencyBusyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for HomeConcurrencyBusyError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HomeDispatchStatusError {
    pub code: String,
    pub message: String,
    pub status_code: u16,
}

impl fmt::Display for HomeDispatchStatusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for HomeDispatchStatusError {}

/// Decodes Home's typed error envelope without allowing missing-candidate
/// failures to collapse into a generic 500.
pub fn decode_home_dispatch_error(raw: &[u8]) -> Option<HomeDispatchStatusError> {
    let root: Value = serde_json::from_slice(raw).ok()?;
    let error = root.get("error")?.as_object()?;
    let code = error
        .get("type")
        .or_else(|| error.get("code"))
        .and_then(Value::as_str)?
        .trim()
        .to_ascii_lowercase();
    if code.is_empty() {
        return None;
    }
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|message| !message.is_empty())
        .unwrap_or("Home rejected auth dispatch")
        .to_owned();
    let status_code = match code.as_str() {
        "credential_concurrency_exceeded" | "credential_model_concurrency_exceeded" => 429,
        "auth_not_found"
        | "auth_unavailable"
        | "refresh_temporarily_unavailable"
        | "home_unavailable"
        | "concurrency_protocol_required"
        | "concurrency_tracker_unavailable"
        | "concurrency_node_unavailable" => 503,
        _ => 0,
    };
    Some(HomeDispatchStatusError {
        code,
        message,
        status_code,
    })
}

pub fn canonical_home_concurrency_model_key(model: &str) -> String {
    let trimmed = model.trim_ascii().to_ascii_lowercase();
    let Some(open) = trimmed.rfind('(') else {
        return trimmed;
    };
    if !trimmed.ends_with(')') || !recognized_suffix(&trimmed[open + 1..trimmed.len() - 1]) {
        return trimmed;
    }
    let base = trimmed[..open].trim_ascii();
    if base.is_empty() {
        trimmed
    } else {
        base.to_owned()
    }
}

pub fn validate_home_concurrency_tuple(
    tuple: &HomeConcurrencyTuple,
) -> Result<(), HomeConcurrencyError> {
    let model = canonical_home_concurrency_model_key(&tuple.model);
    if !tuple.accounted
        || !valid_field(&tuple.credential_id)
        || !valid_field(&model)
        || tuple.model != model
    {
        return Err(HomeConcurrencyError::MalformedTuple);
    }
    Ok(())
}

pub fn decode_home_concurrency(
    raw: &[u8],
) -> Result<Option<HomeConcurrencyTuple>, HomeConcurrencyError> {
    let object: Value =
        serde_json::from_slice(raw).map_err(|_| HomeConcurrencyError::InvalidResponse)?;
    let object = object
        .as_object()
        .ok_or(HomeConcurrencyError::InvalidResponse)?;
    let Some(tuple) = object.get("concurrency") else {
        return Ok(None);
    };
    let tuple: HomeConcurrencyTuple =
        serde_json::from_value(tuple.clone()).map_err(|_| HomeConcurrencyError::MalformedTuple)?;
    validate_home_concurrency_tuple(&tuple)?;
    Ok(Some(tuple))
}

pub fn install_home_concurrency_scope(
    registry: &Registry,
    pending: &Arc<PendingDispatch>,
    tuple: Option<&HomeConcurrencyTuple>,
    mut spec: ScopeSpec,
) -> Result<Scope, HomeConcurrencyError> {
    if let Some(tuple) = tuple {
        validate_home_concurrency_tuple(tuple)?;
        spec.credential_id.clone_from(&tuple.credential_id);
        spec.model.clone_from(&tuple.model);
        spec.accounted = true;
    } else {
        spec.accounted = false;
    }
    registry
        .install(pending, spec)
        .map_err(HomeConcurrencyError::Registry)
}

pub fn verify_home_concurrency_identity(
    tuple: Option<&HomeConcurrencyTuple>,
    auth_id: &str,
    auth_index: &str,
) -> Result<(), HomeConcurrencyError> {
    let Some(tuple) = tuple else {
        return Ok(());
    };
    if tuple.credential_id != auth_id || tuple.credential_id != auth_index {
        return Err(HomeConcurrencyError::IdentityMismatch);
    }
    Ok(())
}

fn valid_field(value: &str) -> bool {
    !value.is_empty() && value.trim() == value && value.len() <= MAX_FIELD_BYTES
}

fn recognized_suffix(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "none" | "auto" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "-1"
    ) || (!value.is_empty()
        && value.len() <= 10
        && value.bytes().all(|byte| byte.is_ascii_digit())
        && value.parse::<i32>().is_ok())
}
