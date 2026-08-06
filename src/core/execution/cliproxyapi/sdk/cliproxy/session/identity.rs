// ref: sdk/cliproxy/session/identity.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

//! Stable conversation identities derived from protocol request roots.

use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::sdk::cliproxy::executor::{ExecutionMetadata, Headers, Options, Request};
use crate::sdk::translator::{claude, codex, gemini, interactions, openai_response, Format};

const IDENTITY_VERSION: &str = "cpa-session-root-v1";
const IDENTITY_PREFIX: &str = "ctx:v1:";
const INSTRUCTION_RUNE_LIMIT: usize = 50;

#[derive(Serialize)]
struct CanonicalRoot {
    version: &'static str,
    format: String,
    caller_scope: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    instructions: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    user: Vec<CanonicalPart>,
    #[serde(skip_serializing_if = "String::is_empty")]
    resource: String,
}

#[derive(Clone, Serialize)]
struct CanonicalPart {
    kind: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    mime: String,
    value: String,
}

/// Validates an explicit client-provided session identifier.
///
/// Opaque printable values are retained, while empty, control-bearing, or
/// oversized (more than 256 UTF-8 bytes) values are rejected.
#[must_use]
pub fn normalize_explicit_id(raw: &str) -> String {
    if raw.chars().any(char::is_control) {
        return String::new();
    }
    let normalized = raw.trim();
    if normalized.is_empty() || normalized.len() > 256 {
        String::new()
    } else {
        normalized.to_owned()
    }
}

/// Extracts a Claude Code session from current JSON metadata or the legacy
/// `_session_<lowercase-hex-or-hyphen>` user-id suffix.
#[must_use]
pub fn claude_metadata_session_id(payload: &[u8]) -> String {
    let Ok(root) = serde_json::from_slice::<Value>(payload) else {
        return String::new();
    };
    let user_id = root
        .get("metadata")
        .and_then(|value| value.get("user_id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if user_id.is_empty() {
        return String::new();
    }
    if user_id.starts_with('{') {
        return serde_json::from_str::<Value>(user_id)
            .ok()
            .and_then(|value| {
                value
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .map_or_else(String::new, |value| normalize_explicit_id(&value));
    }
    legacy_claude_session_suffix(user_id)
        .map(normalize_explicit_id)
        .unwrap_or_default()
}

fn legacy_claude_session_suffix(user_id: &str) -> Option<&str> {
    let (_, suffix) = user_id.rsplit_once("_session_")?;
    (!suffix.is_empty()
        && suffix
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte) || byte == b'-'))
    .then_some(suffix)
}

/// Returns an irreversible namespace for a downstream caller credential.
#[must_use]
pub fn caller_scope(value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        return String::new();
    }
    let digest = Sha256::digest(
        [
            b"cli-proxy-api:caller-scope:v1\0".as_slice(),
            value.as_bytes(),
        ]
        .concat(),
    );
    hex_lower(&digest)
}

/// Returns a derived session identity stored in typed execution metadata.
#[must_use]
pub fn derived_id(metadata: &ExecutionMetadata) -> String {
    metadata
        .derived_session_id
        .as_deref()
        .unwrap_or_default()
        .trim()
        .to_owned()
}

/// Derives a session identity once and places it in request and option metadata.
#[must_use]
pub fn enrich(mut request: Request, mut options: Options) -> (Request, Options) {
    if options.original_request.is_empty() && !request.payload.is_empty() {
        options.original_request = request.payload.clone();
    }
    let payload = options.original_request.as_slice();

    if let Some(execution_id) = first_normalized_id([
        options.metadata.execution_session_id.as_deref(),
        request.metadata.execution_session_id.as_deref(),
    ]) {
        request.metadata.execution_session_id = Some(execution_id.clone());
        options.metadata.execution_session_id = Some(execution_id);
        request.metadata.derived_session_id = None;
        options.metadata.derived_session_id = None;
        return (request, options);
    }
    request.metadata.execution_session_id = None;
    options.metadata.execution_session_id = None;

    if has_explicit_session(&options.headers, payload) {
        request.metadata.derived_session_id = None;
        options.metadata.derived_session_id = None;
        return (request, options);
    }

    let retained_id = first_normalized_id([
        options.metadata.derived_session_id.as_deref(),
        request.metadata.derived_session_id.as_deref(),
    ]);
    request.metadata.derived_session_id = None;
    options.metadata.derived_session_id = None;

    let derived = retained_id.unwrap_or_else(|| {
        let caller = first_trimmed([
            options.metadata.caller_scope.as_deref(),
            request.metadata.caller_scope.as_deref(),
        ]);
        derive_id(&options.source_format, payload, caller)
    });
    if !derived.is_empty() {
        request.metadata.derived_session_id = Some(derived.clone());
        options.metadata.derived_session_id = Some(derived);
    }
    (request, options)
}

fn first_normalized_id<'a>(values: impl IntoIterator<Item = Option<&'a str>>) -> Option<String> {
    values
        .into_iter()
        .flatten()
        .map(normalize_explicit_id)
        .find(|value| !value.is_empty())
}

fn first_trimmed<'a>(values: impl IntoIterator<Item = Option<&'a str>>) -> &'a str {
    values
        .into_iter()
        .flatten()
        .map(str::trim)
        .find(|value| !value.is_empty())
        .unwrap_or_default()
}

fn has_explicit_session(headers: &Headers, payload: &[u8]) -> bool {
    const SESSION_HEADERS: [&str; 6] = [
        "X-Claude-Code-Session-Id",
        "X-Session-ID",
        "Session-Id",
        "Session_id",
        "X-Session-Affinity",
        "X-Client-Request-Id",
    ];
    if SESSION_HEADERS
        .iter()
        .any(|name| !header_value(headers, name).is_empty())
    {
        return true;
    }
    let Ok(root) = serde_json::from_slice::<Value>(payload) else {
        return false;
    };
    if [
        "session_id",
        "sessionId",
        "conversation_id",
        "prompt_cache_key",
    ]
    .iter()
    .any(|key| normalized_json_string(root.get(*key)).is_some())
    {
        return true;
    }
    if !claude_metadata_session_id(payload).is_empty() {
        return true;
    }
    if root
        .get("metadata")
        .and_then(|value| value.get("user_id"))
        .and_then(Value::as_str)
        .is_some_and(|value| !normalize_explicit_id(value.trim()).is_empty())
    {
        return true;
    }
    let Some(conversation) = root.get("conversation") else {
        return false;
    };
    conversation
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|value| !normalize_explicit_id(value).is_empty())
        || conversation
            .as_str()
            .is_some_and(|value| !normalize_explicit_id(value).is_empty())
}

fn normalized_json_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(normalize_explicit_id)
        .filter(|value| !value.is_empty())
}

fn header_value(headers: &Headers, name: &str) -> String {
    headers
        .iter()
        .filter(|(key, _)| key.eq_ignore_ascii_case(name))
        .flat_map(|(_, values)| values)
        .map(|value| normalize_explicit_id(value))
        .find(|value| !value.is_empty())
        .unwrap_or_default()
}

/// Builds a stable identity from leading instructions and the first complete
/// user input in the source protocol.
#[must_use]
pub fn derive_id(format: &Format, payload: &[u8], caller_scope: &str) -> String {
    let Ok(Value::Object(body)) = serde_json::from_slice::<Value>(payload) else {
        return String::new();
    };
    let mut root = CanonicalRoot {
        version: IDENTITY_VERSION,
        format: format.to_string(),
        caller_scope: caller_scope.trim().to_owned(),
        instructions: Vec::new(),
        user: Vec::new(),
        resource: String::new(),
    };
    if source_format_equal(format, &gemini()) {
        root.resource = string_field(&body, &["cachedContent", "cached_content"]);
        (root.instructions, root.user) = gemini_root(&body);
    } else if source_format_equal(format, &interactions()) {
        (root.instructions, root.user) = interactions_root(&body);
    } else if source_format_equal(format, &openai_response())
        || source_format_equal(format, &codex())
    {
        (root.instructions, root.user) = responses_root(&body);
    } else if source_format_equal(format, &claude()) {
        (root.instructions, root.user) = messages_root(&body, true);
    } else {
        (root.instructions, root.user) = messages_root(&body, false);
    }
    if root.user.is_empty() {
        return String::new();
    }
    hash_root(&root)
}

fn messages_root(
    body: &Map<String, Value>,
    include_top_level_system: bool,
) -> (Vec<String>, Vec<CanonicalPart>) {
    let mut instructions = Vec::new();
    if include_top_level_system {
        if let Some(system) = body.get("system") {
            append_instruction(&mut instructions, system);
        }
    }
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for message in messages.iter().filter_map(Value::as_object) {
            match normalized_string(message.get("role")).as_str() {
                "system" | "developer" => {
                    if let Some(content) = message.get("content") {
                        append_instruction(&mut instructions, content);
                    }
                }
                "user" => {
                    return (
                        instructions,
                        canonical_parts(message.get("content").unwrap_or(&Value::Null)),
                    )
                }
                _ => {}
            }
        }
    }
    (instructions, Vec::new())
}

fn responses_root(body: &Map<String, Value>) -> (Vec<String>, Vec<CanonicalPart>) {
    let mut instructions = Vec::new();
    if let Some(value) = body.get("instructions") {
        append_instruction(&mut instructions, value);
    }
    let Some(input) = body.get("input") else {
        return (instructions, Vec::new());
    };
    if input.is_string() {
        return (instructions, canonical_parts(input));
    }
    if let Some(items) = input.as_array() {
        for item in items.iter().filter_map(Value::as_object) {
            match normalized_string(item.get("role")).as_str() {
                "system" | "developer" => {
                    if let Some(content) = item.get("content") {
                        append_instruction(&mut instructions, content);
                    }
                }
                "user" => {
                    return (
                        instructions,
                        canonical_parts(item.get("content").unwrap_or(&Value::Null)),
                    )
                }
                _ => {}
            }
        }
    }
    (instructions, Vec::new())
}

fn gemini_root(body: &Map<String, Value>) -> (Vec<String>, Vec<CanonicalPart>) {
    let mut instructions = Vec::new();
    if let Some(value) = first_field(body, &["systemInstruction", "system_instruction"]) {
        append_instruction(&mut instructions, content_value(value));
    }
    if let Some(contents) = body.get("contents").and_then(Value::as_array) {
        for content in contents.iter().filter_map(Value::as_object) {
            if normalized_string(content.get("role")) == "user" {
                return (
                    instructions,
                    canonical_parts(content_value(&Value::Object(content.clone()))),
                );
            }
        }
    }
    (instructions, Vec::new())
}

fn interactions_root(body: &Map<String, Value>) -> (Vec<String>, Vec<CanonicalPart>) {
    let mut instructions = Vec::new();
    if let Some(value) = first_field(body, &["system_instruction", "systemInstruction"]) {
        append_instruction(&mut instructions, content_value(value));
    }
    let Some(input) = body.get("input") else {
        return (instructions, Vec::new());
    };
    if input.is_string() {
        return (instructions, canonical_parts(input));
    }
    for entry in flatten_interaction_entries(input) {
        if entry.is_string() {
            return (instructions, canonical_parts(&entry));
        }
        let Some(step) = entry.as_object() else {
            continue;
        };
        let role = normalized_string(step.get("role"));
        let step_type = normalized_string(step.get("type"));
        if matches!(role.as_str(), "system" | "developer")
            || matches!(
                step_type.as_str(),
                "system_instruction" | "developer_instruction"
            )
        {
            append_instruction(&mut instructions, content_value(&entry));
        } else if role == "user"
            || step_type == "user_input"
            || (matches!(step_type.as_str(), "message" | "") && role.is_empty())
        {
            return (instructions, canonical_parts(content_value(&entry)));
        }
    }
    (instructions, Vec::new())
}

fn flatten_interaction_entries(value: &Value) -> Vec<Value> {
    fn append_value(entries: &mut Vec<Value>, current: &Value, inherited_role: &str) {
        match current {
            Value::Array(children) => {
                for child in children {
                    append_value(entries, child, inherited_role);
                }
            }
            Value::Object(object) => {
                let own_role = normalized_string(object.get("role"));
                let role = if own_role.is_empty() {
                    inherited_role
                } else {
                    own_role.as_str()
                };
                if let Some(steps) = object.get("steps").and_then(Value::as_array) {
                    for child in steps {
                        append_value(entries, child, role);
                    }
                    return;
                }
                let mut entry = object.clone();
                if !role.is_empty() && own_role.is_empty() {
                    entry.insert("role".to_owned(), Value::String(role.to_owned()));
                }
                entries.push(Value::Object(entry));
            }
            _ => entries.push(current.clone()),
        }
    }
    let mut entries = Vec::new();
    append_value(&mut entries, value, "");
    entries
}

fn append_instruction(instructions: &mut Vec<String>, value: &Value) {
    let mut text = String::new();
    for part in canonical_parts(value) {
        if part.kind != "text" || part.value.is_empty() {
            continue;
        }
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&part.value);
    }
    if !text.is_empty() {
        instructions.push(text.chars().take(INSTRUCTION_RUNE_LIMIT).collect());
    }
}

fn canonical_parts(value: &Value) -> Vec<CanonicalPart> {
    let mut parts = Vec::new();
    append_canonical_parts(&mut parts, value);
    parts
}

fn append_canonical_parts(parts: &mut Vec<CanonicalPart>, value: &Value) {
    match value {
        Value::Null => {}
        Value::String(text) if !text.is_empty() => parts.push(CanonicalPart {
            kind: "text".to_owned(),
            mime: String::new(),
            value: text.clone(),
        }),
        Value::String(_) => {}
        Value::Array(children) => {
            for child in children {
                append_canonical_parts(parts, child);
            }
        }
        Value::Object(object) => {
            if let Some(Value::String(text)) = object.get("text") {
                append_canonical_parts(parts, &Value::String(text.clone()));
            } else if let Some(nested) = object.get("content") {
                append_canonical_parts(parts, nested);
            } else if let Some(nested) = object.get("parts") {
                append_canonical_parts(parts, nested);
            } else if let Some(image_url) = object.get("image_url") {
                append_media_part(parts, "image", image_url, "");
            } else if let Some(inline_data) = first_field(object, &["inlineData", "inline_data"]) {
                append_media_part(parts, "inline_data", inline_data, "");
            } else if let Some(file_data) = first_field(object, &["fileData", "file_data"]) {
                append_media_part(parts, "file", file_data, "");
            } else if let Some(source) = object.get("source") {
                append_media_part(
                    parts,
                    &normalized_string(object.get("type")),
                    source,
                    &normalized_string(object.get("media_type")),
                );
            } else if let Ok(encoded) = serde_json::to_string(&normalize_json_value(value)) {
                if !encoded.is_empty() {
                    parts.push(CanonicalPart {
                        kind: "json".to_owned(),
                        mime: String::new(),
                        value: encoded,
                    });
                }
            }
        }
        _ => {
            if let Ok(encoded) = serde_json::to_string(value) {
                if !encoded.is_empty() {
                    parts.push(CanonicalPart {
                        kind: "json".to_owned(),
                        mime: String::new(),
                        value: encoded,
                    });
                }
            }
        }
    }
}

fn append_media_part(
    parts: &mut Vec<CanonicalPart>,
    kind: &str,
    value: &Value,
    fallback_mime: &str,
) {
    let kind = if kind.trim().is_empty() {
        "media"
    } else {
        kind.trim()
    };
    match value {
        Value::String(media_value) if !media_value.is_empty() => parts.push(CanonicalPart {
            kind: kind.to_owned(),
            mime: fallback_mime.to_owned(),
            value: media_value.clone(),
        }),
        Value::Object(object) => {
            let mime = {
                let value = string_field(object, &["mimeType", "mime_type", "media_type"]);
                if value.is_empty() {
                    fallback_mime.to_owned()
                } else {
                    value
                }
            };
            let media_value = string_field(object, &["url", "uri", "fileUri", "file_uri", "data"]);
            if !media_value.is_empty() {
                parts.push(CanonicalPart {
                    kind: kind.to_owned(),
                    mime,
                    value: media_value,
                });
            }
        }
        _ => append_canonical_parts(parts, value),
    }
}

fn content_value(value: &Value) -> &Value {
    let Some(object) = value.as_object() else {
        return value;
    };
    object
        .get("content")
        .or_else(|| object.get("parts"))
        .or_else(|| object.get("text"))
        .unwrap_or(value)
}

fn normalize_json_value(value: &Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(key, _)| !key.trim().eq_ignore_ascii_case("cache_control"))
                .map(|(key, child)| (key.clone(), normalize_json_value(child)))
                .collect(),
        ),
        Value::Array(children) => Value::Array(children.iter().map(normalize_json_value).collect()),
        _ => value.clone(),
    }
}

fn hash_root(root: &CanonicalRoot) -> String {
    serde_json::to_vec(root).map_or_else(
        |_| String::new(),
        |encoded| format!("{IDENTITY_PREFIX}{}", hex_lower(&Sha256::digest(encoded))),
    )
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn first_field<'a>(object: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| object.get(*key))
}

fn string_field(object: &Map<String, Value>, keys: &[&str]) -> String {
    first_field(object, keys)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned()
}

fn normalized_string(value: Option<&Value>) -> String {
    value
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_lowercase()
}

fn source_format_equal(left: &Format, right: &Format) -> bool {
    left.as_str()
        .trim()
        .eq_ignore_ascii_case(right.as_str().trim())
}
