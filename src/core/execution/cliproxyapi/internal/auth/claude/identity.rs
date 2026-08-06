// ref: internal/auth/claude/identity.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Candidate delta evidence: internal/auth/claude/identity.go
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;

use serde_json::Value;

pub const CLAUDE_DEVICE_IDS_METADATA_KEY: &str = "claude_device_ids";
pub const CLAUDE_DEVICE_POOL_SIZE: usize = 1;
const CLAUDE_DEVICE_ID_BYTE_SIZE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaudeIdentityError {
    Randomness,
    DevicePoolSize { actual: usize },
    EmptySessionId,
}

impl fmt::Display for ClaudeIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Randomness => {
                formatter.write_str("generate Claude device ID: random source failed")
            }
            Self::DevicePoolSize { actual } => write!(
                formatter,
                "select Claude device ID: device pool has {actual} entries, want {CLAUDE_DEVICE_POOL_SIZE}"
            ),
            Self::EmptySessionId => {
                formatter.write_str("select Claude device ID: session ID is empty")
            }
        }
    }
}

impl std::error::Error for ClaudeIdentityError {}

/// Generates the fixed-size device identity pool persisted with one credential.
pub fn generate_device_id_pool() -> Result<Vec<String>, ClaudeIdentityError> {
    let mut bytes = [0_u8; CLAUDE_DEVICE_ID_BYTE_SIZE];
    getrandom::fill(&mut bytes).map_err(|_| ClaudeIdentityError::Randomness)?;
    Ok(vec![encode_lower_hex(&bytes)])
}

/// Returns the first valid device ID in canonical lowercase form.
///
/// JSON arrays are the Rust equivalent of both Go `[]string` and the `[]any`
/// produced by generic JSON decoding. Non-string entries are ignored, matching
/// `NormalizeDeviceIDPool` rather than the stricter canonicality predicate.
pub fn normalize_device_id_pool(raw: Option<&Value>) -> Vec<String> {
    let Some(Value::Array(values)) = raw else {
        return Vec::new();
    };
    values
        .iter()
        .filter_map(Value::as_str)
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .find(|value| valid_device_id(value))
        .into_iter()
        .collect()
}

/// Reports whether the JSON value stores exactly one already-canonical ID.
pub fn has_canonical_device_id_pool(raw: Option<&Value>) -> bool {
    let Some(Value::Array(values)) = raw else {
        return false;
    };
    if values.len() != CLAUDE_DEVICE_POOL_SIZE {
        return false;
    }
    let Some(stored) = values[0].as_str() else {
        return false;
    };
    valid_device_id(stored)
        && normalize_device_id_pool(raw)
            .first()
            .is_some_and(|id| id == stored)
}

/// Repairs or creates the single-device pool in credential metadata.
///
/// Upstream requires a package mutex because a shared Go map can be accessed
/// concurrently. Rust requires an exclusive mutable borrow instead; the
/// credential owner remains responsible for locking the aggregate before it
/// obtains this borrow. Returned values are owned and cannot alias metadata.
pub fn ensure_device_id_pool(
    metadata: &mut BTreeMap<String, Value>,
) -> Result<(Vec<String>, bool), ClaudeIdentityError> {
    let raw = metadata.get(CLAUDE_DEVICE_IDS_METADATA_KEY);
    let mut device_ids = normalize_device_id_pool(raw);
    let changed = !has_canonical_device_id_pool(raw);
    if device_ids.is_empty() {
        device_ids = generate_device_id_pool()?;
    }
    if changed {
        store_device_id_pool(metadata, &device_ids);
    }
    Ok((device_ids, changed))
}

pub fn read_device_id_pool(metadata: &BTreeMap<String, Value>) -> Option<Value> {
    metadata.get(CLAUDE_DEVICE_IDS_METADATA_KEY).cloned()
}

pub fn store_device_id_pool(metadata: &mut BTreeMap<String, Value>, device_ids: &[String]) {
    metadata.insert(
        CLAUDE_DEVICE_IDS_METADATA_KEY.to_owned(),
        Value::Array(device_ids.iter().cloned().map(Value::String).collect()),
    );
}

pub fn read_metadata_string<'a>(
    metadata: &'a BTreeMap<String, Value>,
    key: &str,
) -> Option<&'a str> {
    metadata.get(key).and_then(Value::as_str)
}

/// Empty or whitespace-only values do not erase a previously resolved value.
pub fn store_metadata_string(
    metadata: &mut BTreeMap<String, Value>,
    key: impl Into<String>,
    value: impl Into<String>,
) -> bool {
    let value = value.into();
    if value.trim().is_empty() {
        return false;
    }
    metadata.insert(key.into(), Value::String(value));
    true
}

pub fn store_metadata_value(
    metadata: &mut BTreeMap<String, Value>,
    key: impl Into<String>,
    value: Value,
) {
    metadata.insert(key.into(), value);
}

/// Returns the credential's sole device ID after validating the session.
pub fn select_device_id(
    device_ids: &[String],
    session_id: &str,
) -> Result<String, ClaudeIdentityError> {
    let normalized = normalize_device_id_pool(Some(&Value::Array(
        device_ids.iter().cloned().map(Value::String).collect(),
    )));
    let Some(device_id) = normalized.first() else {
        return Err(ClaudeIdentityError::DevicePoolSize {
            actual: normalized.len(),
        });
    };
    if session_id.trim().is_empty() {
        return Err(ClaudeIdentityError::EmptySessionId);
    }
    Ok(device_id.clone())
}

pub fn valid_device_id(value: &str) -> bool {
    value.len() == CLAUDE_DEVICE_ID_BYTE_SIZE * 2
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}
