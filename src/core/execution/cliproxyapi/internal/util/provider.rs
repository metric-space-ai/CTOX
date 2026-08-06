// ref: internal/util/provider.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeSet;

const OPENAI_COMPATIBLE_PROVIDER_PREFIX: &str = "openai-compatible-";

/// Read-only boundary to the model registry. CTOX injects this view rather
/// than reaching through a package-global singleton as the Go implementation
/// does.
pub trait ModelRegistryView {
    type Error;

    fn model_providers(&self, model_name: &str) -> Vec<String>;
    fn first_available_model(&self, handler_type: &str) -> Result<String, Self::Error>;
}

/// Structural view of the compatibility-model fields used by this utility.
pub trait OpenAiCompatibilityModelView {
    fn alias(&self) -> &str;
}

/// Structural view of one OpenAI-compatible provider configuration.
pub trait OpenAiCompatibilityEntryView {
    type Model: OpenAiCompatibilityModelView;

    fn disabled(&self) -> bool;
    fn models(&self) -> &[Self::Model];
}

#[must_use]
pub fn openai_compatible_provider_key(name: &str) -> String {
    let name = go_simple_lowercase(name.trim());
    if name.is_empty()
        || name == "openai-compatibility"
        || name.starts_with(OPENAI_COMPATIBLE_PROVIDER_PREFIX)
    {
        if name.is_empty() {
            "openai-compatibility".to_owned()
        } else {
            name
        }
    } else {
        format!("{OPENAI_COMPATIBLE_PROVIDER_PREFIX}{name}")
    }
}

/// Returns all registered providers in registry order with exact, case-
/// sensitive de-duplication. The pinned implementation no longer performs the
/// legacy model-name heuristic described by its stale comment.
#[must_use]
pub fn get_provider_name<R: ModelRegistryView>(model_name: &str, registry: &R) -> Vec<String> {
    if model_name.is_empty() {
        return Vec::new();
    }

    let mut seen = BTreeSet::new();
    let mut providers = Vec::with_capacity(4);
    for provider in registry.model_providers(model_name) {
        if !provider.is_empty() && seen.insert(provider.clone()) {
            providers.push(provider);
        }
    }
    providers
}

#[must_use]
pub fn resolve_auto_model<R: ModelRegistryView>(model_name: &str, registry: &R) -> String {
    if model_name != "auto" {
        return model_name.to_owned();
    }

    registry
        .first_available_model("")
        .unwrap_or_else(|_| model_name.to_owned())
}

#[must_use]
pub fn is_openai_compatibility_alias<C: OpenAiCompatibilityEntryView>(
    model_name: &str,
    config: Option<&[C]>,
) -> bool {
    get_openai_compatibility_config(model_name, config).is_some()
}

#[must_use]
pub fn get_openai_compatibility_config<'a, C: OpenAiCompatibilityEntryView>(
    alias: &str,
    config: Option<&'a [C]>,
) -> Option<(&'a C, &'a C::Model)> {
    config?
        .iter()
        .filter(|entry| !entry.disabled())
        .find_map(|entry| {
            entry
                .models()
                .iter()
                .find(|model| model.alias() == alias)
                .map(|model| (entry, model))
        })
}

#[must_use]
pub fn in_array(haystack: &[String], needle: &str) -> bool {
    haystack.iter().any(|item| item == needle)
}

/// Byte-exact port of Go's string-slicing masker. A byte API is intentional:
/// Go may split a UTF-8 code point and therefore return a non-UTF-8 string.
#[must_use]
pub fn hide_api_key(api_key: &[u8]) -> Vec<u8> {
    let length = api_key.len();
    let (prefix, suffix) = if length > 8 {
        (4, 4)
    } else if length > 4 {
        (2, 2)
    } else if length > 2 {
        (1, 1)
    } else {
        return api_key.to_vec();
    };

    let mut hidden = Vec::with_capacity(prefix + 3 + suffix);
    hidden.extend_from_slice(&api_key[..prefix]);
    hidden.extend_from_slice(b"...");
    hidden.extend_from_slice(&api_key[length - suffix..]);
    hidden
}

#[must_use]
pub fn mask_authorization_header(value: &str) -> Vec<u8> {
    let trimmed = value.trim();
    if let Some((auth_type, credential)) = trimmed.split_once(' ') {
        let mut masked = Vec::with_capacity(auth_type.len() + credential.len() + 4);
        masked.extend_from_slice(auth_type.as_bytes());
        masked.push(b' ');
        masked.extend_from_slice(&hide_api_key(credential.as_bytes()));
        masked
    } else {
        hide_api_key(value.as_bytes())
    }
}

#[must_use]
pub fn mask_sensitive_header_value(key: &str, value: &str) -> Vec<u8> {
    let lower_key = go_simple_lowercase(key.trim());
    if lower_key.contains("authorization") {
        mask_authorization_header(value)
    } else if ["api-key", "apikey", "token", "secret"]
        .iter()
        .any(|needle| lower_key.contains(needle))
    {
        hide_api_key(value.as_bytes())
    } else {
        value.as_bytes().to_vec()
    }
}

#[must_use]
pub fn mask_sensitive_query(raw: &str) -> String {
    if raw.is_empty() {
        return String::new();
    }

    let mut parts = raw.split('&').map(str::to_owned).collect::<Vec<_>>();
    let mut changed = false;
    for part in &mut parts {
        if part.is_empty() {
            continue;
        }

        let (key_part, value_part) = part.split_once('=').unwrap_or((part.as_str(), ""));
        let decoded_key = query_unescape(key_part).unwrap_or_else(|| key_part.as_bytes().to_vec());
        if !should_mask_query_param(&decoded_key) {
            continue;
        }

        let decoded_value =
            query_unescape(value_part).unwrap_or_else(|| value_part.as_bytes().to_vec());
        let trimmed_value = trim_space_bytes(&decoded_value);
        let masked = hide_api_key(trimmed_value);
        *part = format!("{key_part}={}", query_escape(&masked));
        changed = true;
    }

    if changed {
        parts.join("&")
    } else {
        raw.to_owned()
    }
}

fn should_mask_query_param(key: &[u8]) -> bool {
    let normalized = go_simple_lowercase(&String::from_utf8_lossy(trim_space_bytes(key)));
    let normalized = normalized.strip_suffix("[]").unwrap_or(&normalized);
    !normalized.is_empty()
        && (normalized == "key"
            || ["api-key", "apikey", "api_key", "token", "secret"]
                .iter()
                .any(|needle| normalized.contains(needle)))
}

/// Go's `strings.ToLower` uses Unicode simple case mappings (one rune in, at
/// most one rune out). Rust exposes full mappings, which can expand a scalar;
/// taking the first mapped scalar preserves Go behavior for cases such as
/// U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE.
fn go_simple_lowercase(value: &str) -> String {
    value
        .chars()
        .map(|character| character.to_lowercase().next().unwrap_or(character))
        .collect()
}

fn query_unescape(value: &str) -> Option<Vec<u8>> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' => {
                let high = *bytes.get(index + 1)?;
                let low = *bytes.get(index + 2)?;
                decoded.push(hex_value(high)? << 4 | hex_value(low)?);
                index += 3;
            }
            b'+' => {
                decoded.push(b' ');
                index += 1;
            }
            byte => {
                decoded.push(byte);
                index += 1;
            }
        }
    }
    Some(decoded)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn query_escape(value: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut escaped = String::with_capacity(value.len());
    for &byte in value {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            escaped.push(byte as char);
        } else if byte == b' ' {
            escaped.push('+');
        } else {
            escaped.push('%');
            escaped.push(HEX[(byte >> 4) as usize] as char);
            escaped.push(HEX[(byte & 0x0f) as usize] as char);
        }
    }
    escaped
}

fn trim_space_bytes(value: &[u8]) -> &[u8] {
    match std::str::from_utf8(value) {
        Ok(value) => value.trim().as_bytes(),
        Err(_) => {
            let start = value
                .iter()
                .position(|byte| !byte.is_ascii_whitespace())
                .unwrap_or(value.len());
            let end = value
                .iter()
                .rposition(|byte| !byte.is_ascii_whitespace())
                .map_or(start, |index| index + 1);
            &value[start..end]
        }
    }
}
