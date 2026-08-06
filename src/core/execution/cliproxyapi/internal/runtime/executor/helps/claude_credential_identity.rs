// ref: internal/runtime/executor/helps/claude_credential_identity.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeSet;
use std::fmt;

use serde_json::Value;
use uuid::Uuid;

use crate::internal::auth::claude::{
    ensure_device_id_pool, generate_device_id_pool, has_canonical_device_id_pool,
    normalize_device_id_pool, read_device_id_pool, read_metadata_string, select_device_id,
    store_device_id_pool, ClaudeIdentityError, CLAUDE_DEVICE_POOL_SIZE,
};
use crate::internal::home::hash_key_part;
use crate::sdk::api::handlers::header_filter::HeaderMap;
use crate::sdk::cliproxy::auth::Auth;
use crate::sdk::cliproxy::executor::ExecutionMetadata;

use super::claude_code_session::{extract_claude_code_session_id, CLAUDE_CODE_SESSION_HEADER};

pub trait ClaudeCredentialDevicePoolStore: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, ClaudeCredentialIdentityError>;
    fn set_nx(&self, key: &str, value: &[u8]) -> Result<bool, ClaudeCredentialIdentityError>;
    fn set_existing(&self, key: &str, value: &[u8]) -> Result<bool, ClaudeCredentialIdentityError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClaudeCredentialIdentityError {
    NilAuth,
    EmptyCredentialIdentity,
    EmptyAccountUuid,
    Device(ClaudeIdentityError),
    Store(String),
    InvalidStoredPool,
    MissingAfterSet,
    InvalidJson(String),
    DuplicateJsonKey(String),
}

impl ClaudeCredentialIdentityError {
    pub fn is_request_scoped(&self) -> bool {
        matches!(self, Self::InvalidJson(_) | Self::DuplicateJsonKey(_))
    }

    pub fn status_code(&self) -> Option<u16> {
        self.is_request_scoped().then_some(400)
    }
}

impl fmt::Display for ClaudeCredentialIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NilAuth => formatter.write_str("Claude credential identity: auth is nil"),
            Self::EmptyCredentialIdentity => {
                formatter.write_str("Claude credential identity is empty")
            }
            Self::EmptyAccountUuid => formatter.write_str("Claude account UUID is empty"),
            Self::Device(error) => error.fmt(formatter),
            Self::Store(message) => write!(formatter, "Claude credential device store: {message}"),
            Self::InvalidStoredPool => formatter.write_str("stored Claude device pool is invalid"),
            Self::MissingAfterSet => {
                formatter.write_str("Claude device pool missing after conditional set")
            }
            Self::InvalidJson(message) => write!(formatter, "invalid request JSON: {message}"),
            Self::DuplicateJsonKey(key) => write!(formatter, "duplicate JSON object key {key:?}"),
        }
    }
}

impl std::error::Error for ClaudeCredentialIdentityError {}

impl From<ClaudeIdentityError> for ClaudeCredentialIdentityError {
    fn from(value: ClaudeIdentityError) -> Self {
        Self::Device(value)
    }
}

/// Maps downstream conversation signals onto one provider-stable UUID.
pub fn claude_agent_session_uuid(
    headers: Option<&HeaderMap>,
    original_payload: &[u8],
    translated_payload: &[u8],
    metadata_sets: &[&ExecutionMetadata],
) -> String {
    claude_agent_session_uuid_for_request(
        headers,
        original_payload,
        translated_payload,
        true,
        metadata_sets,
    )
}

pub fn claude_agent_session_uuid_for_request(
    headers: Option<&HeaderMap>,
    original_payload: &[u8],
    translated_payload: &[u8],
    confirmed_claude_code: bool,
    metadata_sets: &[&ExecutionMetadata],
) -> String {
    let mut sanitized_headers = headers.cloned().unwrap_or_default();
    let (original, translated) = if confirmed_claude_code {
        (original_payload.to_vec(), translated_payload.to_vec())
    } else {
        sanitized_headers.retain(|key, _| !key.eq_ignore_ascii_case(CLAUDE_CODE_SESSION_HEADER));
        (
            without_claude_metadata_user_id(original_payload),
            without_claude_metadata_user_id(translated_payload),
        )
    };
    let header_view = (!sanitized_headers.is_empty()).then_some(&sanitized_headers);
    let mut identity = extract_claude_code_session_id(&original, header_view);
    if identity.is_empty() {
        identity = extract_protocol_session_id(&original);
    }
    if identity.is_empty() {
        identity = extract_claude_code_session_id(&translated, header_view);
    }
    if identity.is_empty() {
        identity = extract_protocol_session_id(&translated);
    }
    if identity.is_empty() {
        identity = metadata_sets
            .iter()
            .find_map(|metadata| {
                metadata
                    .execution_session_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .or_else(|| {
                        metadata
                            .derived_session_id
                            .as_deref()
                            .map(str::trim)
                            .filter(|value| !value.is_empty())
                    })
            })
            .unwrap_or_default()
            .to_owned();
    }
    if identity.is_empty() {
        return Uuid::new_v4().to_string();
    }
    let identity = identity.strip_prefix("claude:").unwrap_or(&identity);
    Uuid::parse_str(identity).map_or_else(
        |_| {
            Uuid::new_v5(
                &Uuid::NAMESPACE_OID,
                format!("cli-proxy-api\0claude\0agent-conversation\0{identity}").as_bytes(),
            )
            .to_string()
        },
        |uuid| uuid.to_string(),
    )
}

fn extract_protocol_session_id(payload: &[u8]) -> String {
    let Ok(root) = serde_json::from_slice::<Value>(payload) else {
        return String::new();
    };
    ["session_id", "conversation_id", "conversationId"]
        .iter()
        .find_map(|key| root.get(key).and_then(Value::as_str))
        .map(str::trim)
        .unwrap_or_default()
        .to_owned()
}

fn without_claude_metadata_user_id(payload: &[u8]) -> Vec<u8> {
    let Ok(mut root) = serde_json::from_slice::<Value>(payload) else {
        return payload.to_vec();
    };
    if let Some(metadata) = root.get_mut("metadata").and_then(Value::as_object_mut) {
        metadata.remove("user_id");
    }
    serde_json::to_vec(&root).unwrap_or_else(|_| payload.to_vec())
}

/// Ensures a canonical single-device credential pool. A supplied store is the
/// Home authority; otherwise the caller's exclusively borrowed auth is local
/// authority, replacing Go's package mutex with Rust ownership.
pub fn ensure_claude_credential_device_pool_required(
    store: Option<&dyn ClaudeCredentialDevicePoolStore>,
    auth: &mut Auth,
) -> Result<Vec<String>, ClaudeCredentialIdentityError> {
    let raw = read_device_id_pool(&auth.metadata);
    if has_canonical_device_id_pool(raw.as_ref()) {
        return Ok(normalize_device_id_pool(raw.as_ref()));
    }
    let candidate = normalize_device_id_pool(raw.as_ref());
    let Some(store) = store else {
        return ensure_device_id_pool(&mut auth.metadata)
            .map(|(pool, _)| pool)
            .map_err(Into::into);
    };

    let identity = {
        let index = auth.ensure_index();
        if index.trim().is_empty() {
            auth.id.trim().to_owned()
        } else {
            index.trim().to_owned()
        }
    };
    if identity.is_empty() {
        return Err(ClaudeCredentialIdentityError::EmptyCredentialIdentity);
    }
    let key = format!(
        "cpa:claude:credential-device-pool:{}",
        hash_key_part(&identity)
    );
    if let Some(raw) = store.get(&key)? {
        if let Ok(stored) = serde_json::from_slice::<Vec<String>>(&raw) {
            let normalized = normalize_pool_strings(&stored);
            if normalized.len() == CLAUDE_DEVICE_POOL_SIZE {
                if stored != normalized
                    && !store.set_existing(
                        &key,
                        &serde_json::to_vec(&normalized).map_err(|error| {
                            ClaudeCredentialIdentityError::Store(error.to_string())
                        })?,
                    )?
                {
                    return Err(ClaudeCredentialIdentityError::Store(
                        "canonical value was not written".to_owned(),
                    ));
                }
                store_device_id_pool(&mut auth.metadata, &normalized);
                return Ok(normalized);
            }
        }
    }

    let candidate = if candidate.len() == CLAUDE_DEVICE_POOL_SIZE {
        candidate
    } else {
        generate_device_id_pool()?
    };
    let encoded = serde_json::to_vec(&candidate)
        .map_err(|error| ClaudeCredentialIdentityError::Store(error.to_string()))?;
    let _ = store.set_nx(&key, &encoded)?;
    let raw = store
        .get(&key)?
        .ok_or(ClaudeCredentialIdentityError::MissingAfterSet)?;
    let stored = serde_json::from_slice::<Vec<String>>(&raw)
        .map_err(|_| ClaudeCredentialIdentityError::InvalidStoredPool)?;
    let normalized = normalize_pool_strings(&stored);
    if normalized.len() != CLAUDE_DEVICE_POOL_SIZE {
        return Err(ClaudeCredentialIdentityError::InvalidStoredPool);
    }
    store_device_id_pool(&mut auth.metadata, &normalized);
    Ok(normalized)
}

fn normalize_pool_strings(values: &[String]) -> Vec<String> {
    normalize_device_id_pool(Some(&Value::Array(
        values.iter().cloned().map(Value::String).collect(),
    )))
}

pub fn claude_credential_account_uuid(auth: &Auth) -> String {
    ["account_uuid", "accountUuid"]
        .iter()
        .find_map(|key| read_metadata_string(&auth.metadata, key))
        .map(str::trim)
        .unwrap_or_default()
        .to_owned()
}

pub fn apply_claude_credential_metadata(
    payload: &[u8],
    auth: &mut Auth,
    session_id: &str,
) -> Result<(Vec<u8>, String), ClaudeCredentialIdentityError> {
    let members = parse_unique_json_object(payload)?;
    let metadata_raw = members
        .iter()
        .find(|(key, _)| key == "metadata")
        .map(|(_, value)| value.as_slice());
    let existing = if let Some(metadata_raw) = metadata_raw {
        if metadata_raw
            .iter()
            .copied()
            .find(|byte| !byte.is_ascii_whitespace())
            == Some(b'{')
        {
            let metadata = parse_unique_json_object(metadata_raw)?;
            metadata
                .iter()
                .find(|(key, _)| key == "user_id")
                .and_then(|(_, raw)| serde_json::from_slice::<String>(raw).ok())
                .unwrap_or_default()
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let (device_ids, _) = ensure_device_id_pool(&mut auth.metadata)?;
    let device_id = select_device_id(&device_ids, session_id)?;
    let account_uuid = claude_credential_account_uuid(auth);
    if account_uuid.is_empty() {
        return Err(ClaudeCredentialIdentityError::EmptyAccountUuid);
    }
    let encoded_identity =
        rebuild_claude_metadata_user_id(&existing, &device_id, &account_uuid, session_id)?;

    let mut root = serde_json::from_slice::<Value>(payload)
        .map_err(|error| ClaudeCredentialIdentityError::InvalidJson(error.to_string()))?;
    let object = root.as_object_mut().ok_or_else(|| {
        ClaudeCredentialIdentityError::InvalidJson("request must be a JSON object".to_owned())
    })?;
    let metadata = object
        .entry("metadata")
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if !metadata.is_object() {
        *metadata = Value::Object(serde_json::Map::new());
    }
    metadata
        .as_object_mut()
        .expect("metadata was normalized to an object")
        .insert(
            "user_id".to_owned(),
            Value::String(String::from_utf8(encoded_identity).expect("JSON is UTF-8")),
        );
    let updated = serde_json::to_vec(&root)
        .map_err(|error| ClaudeCredentialIdentityError::InvalidJson(error.to_string()))?;
    Ok((updated, device_id))
}

fn rebuild_claude_metadata_user_id(
    existing: &str,
    device_id: &str,
    account_uuid: &str,
    session_id: &str,
) -> Result<Vec<u8>, ClaudeCredentialIdentityError> {
    let extras = if existing.trim_start().starts_with('{') {
        parse_unique_json_object(existing.as_bytes())?
            .into_iter()
            .filter(|(key, _)| !matches!(key.as_str(), "device_id" | "account_uuid" | "session_id"))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut output = Vec::new();
    output.extend_from_slice(b"{\"device_id\":");
    output.extend_from_slice(&serde_json::to_vec(device_id).expect("string JSON"));
    output.extend_from_slice(b",\"account_uuid\":");
    output.extend_from_slice(&serde_json::to_vec(account_uuid).expect("string JSON"));
    output.extend_from_slice(b",\"session_id\":");
    output.extend_from_slice(&serde_json::to_vec(session_id).expect("string JSON"));
    for (key, value) in extras {
        output.push(b',');
        output.extend_from_slice(&serde_json::to_vec(&key).expect("string JSON"));
        output.push(b':');
        output.extend_from_slice(&value);
    }
    output.push(b'}');
    Ok(output)
}

fn parse_unique_json_object(
    raw: &[u8],
) -> Result<Vec<(String, Vec<u8>)>, ClaudeCredentialIdentityError> {
    let raw = trim_ascii(raw);
    if serde_json::from_slice::<Value>(raw).is_err() || raw.first() != Some(&b'{') {
        return Err(ClaudeCredentialIdentityError::InvalidJson(
            "request must be a JSON object".to_owned(),
        ));
    }
    let mut position = 1;
    let mut seen = BTreeSet::new();
    let mut members = Vec::new();
    loop {
        position = skip_whitespace(raw, position);
        if raw.get(position) == Some(&b'}') {
            break;
        }
        let key_start = position;
        let key_end = skip_json_string(raw, key_start);
        let key = serde_json::from_slice::<String>(&raw[key_start..key_end])
            .map_err(|error| ClaudeCredentialIdentityError::InvalidJson(error.to_string()))?;
        if !seen.insert(key.clone()) {
            return Err(ClaudeCredentialIdentityError::DuplicateJsonKey(key));
        }
        position = skip_whitespace(raw, key_end);
        if raw.get(position) != Some(&b':') {
            return Err(ClaudeCredentialIdentityError::InvalidJson(
                "object key is missing a value".to_owned(),
            ));
        }
        position = skip_whitespace(raw, position + 1);
        let value_start = position;
        position = skip_json_value(raw, position);
        members.push((key, raw[value_start..position].to_vec()));
        position = skip_whitespace(raw, position);
        match raw.get(position) {
            Some(b',') => position += 1,
            Some(b'}') => break,
            _ => {
                return Err(ClaudeCredentialIdentityError::InvalidJson(
                    "object member has an invalid terminator".to_owned(),
                ))
            }
        }
    }
    Ok(members)
}

fn trim_ascii(raw: &[u8]) -> &[u8] {
    let start = raw
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(raw.len());
    let end = raw
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map_or(start, |index| index + 1);
    &raw[start..end]
}

fn skip_whitespace(raw: &[u8], mut position: usize) -> usize {
    while raw.get(position).is_some_and(u8::is_ascii_whitespace) {
        position += 1;
    }
    position
}

fn skip_json_string(raw: &[u8], mut position: usize) -> usize {
    if raw.get(position) != Some(&b'"') {
        return position;
    }
    position += 1;
    while position < raw.len() {
        match raw[position] {
            b'\\' => position += 2,
            b'"' => return position + 1,
            _ => position += 1,
        }
    }
    position
}

fn skip_json_value(raw: &[u8], mut position: usize) -> usize {
    match raw.get(position) {
        Some(b'"') => skip_json_string(raw, position),
        Some(b'{') | Some(b'[') => {
            let mut stack = vec![raw[position]];
            position += 1;
            while position < raw.len() && !stack.is_empty() {
                match raw[position] {
                    b'"' => {
                        position = skip_json_string(raw, position);
                        continue;
                    }
                    b'{' | b'[' => stack.push(raw[position]),
                    b'}' | b']' => {
                        stack.pop();
                    }
                    _ => {}
                }
                position += 1;
            }
            position
        }
        _ => {
            while position < raw.len()
                && !matches!(raw[position], b',' | b'}' | b']')
                && !raw[position].is_ascii_whitespace()
            {
                position += 1;
            }
            position
        }
    }
}
