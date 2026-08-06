// ref: internal/signature/claude_validation.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};
use serde_json::Value;

pub const MAX_CLAUDE_THINKING_SIGNATURE_LEN: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClaudeSignatureValidationOptions {
    pub prefix_only: bool,
    pub base64_only: bool,
    pub allow_empty_signature_with_empty_text: bool,
    pub strict: bool,
}

pub fn is_valid_claude_thinking_signature(raw_signature: &str) -> bool {
    is_valid_claude_thinking_signature_with_options(
        raw_signature,
        ClaudeSignatureValidationOptions::default(),
    )
}

pub fn is_valid_claude_thinking_signature_with_options(
    raw_signature: &str,
    options: ClaudeSignatureValidationOptions,
) -> bool {
    if options.prefix_only {
        return has_claude_thinking_signature_prefix(raw_signature);
    }
    if options.base64_only {
        return has_decodable_claude_thinking_signature(raw_signature);
    }
    normalize_claude_thinking_signature(raw_signature, options).is_some()
}

pub fn has_claude_thinking_signature_prefix(raw_signature: &str) -> bool {
    matches!(
        strip_provider_prefix(raw_signature).as_bytes().first(),
        Some(b'E' | b'R')
    )
}

pub fn has_decodable_claude_thinking_signature(raw_signature: &str) -> bool {
    let signature = strip_provider_prefix(raw_signature);
    if signature.is_empty() || signature.len() > MAX_CLAUDE_THINKING_SIGNATURE_LEN {
        return false;
    }
    match signature.as_bytes().first() {
        Some(b'E') => general_purpose::STANDARD
            .decode(signature)
            .is_ok_and(|decoded| !decoded.is_empty()),
        Some(b'R') => general_purpose::STANDARD
            .decode(signature)
            .ok()
            .filter(|decoded| decoded.first() == Some(&b'E'))
            .and_then(|inner| general_purpose::STANDARD.decode(inner).ok())
            .is_some_and(|decoded| !decoded.is_empty()),
        _ => false,
    }
}

/// Normalizes an E/R Claude signature to Antigravity's double-layer R form.
/// Basic mode checks the decoded 0x12 marker; strict mode additionally requires
/// the known Claude field-2/container/channel protobuf tree.
pub fn normalize_claude_bypass_thinking_signature(
    raw_signature: &str,
    strict: bool,
) -> Option<String> {
    normalize_claude_thinking_signature(
        raw_signature,
        ClaudeSignatureValidationOptions {
            strict,
            ..ClaudeSignatureValidationOptions::default()
        },
    )
}

/// Returns the double-layer R form expected by Antigravity bypass mode.
pub fn normalize_claude_thinking_signature(
    raw_signature: &str,
    options: ClaudeSignatureValidationOptions,
) -> Option<String> {
    let signature = strip_provider_prefix(raw_signature);
    if signature.is_empty() || signature.len() > MAX_CLAUDE_THINKING_SIGNATURE_LEN {
        return None;
    }
    match signature.as_bytes().first() {
        Some(b'E') => validate_single_layer(signature, options.strict)
            .then(|| general_purpose::STANDARD.encode(signature.as_bytes())),
        Some(b'R') => {
            let inner = general_purpose::STANDARD.decode(signature).ok()?;
            let inner = std::str::from_utf8(&inner).ok()?;
            (inner.starts_with('E') && validate_single_layer(inner, options.strict))
                .then(|| signature.to_owned())
        }
        _ => None,
    }
}

pub fn normalize_claude_provider_native_thinking_signature(raw_signature: &str) -> Option<String> {
    normalize_claude_provider_native_thinking_signature_with_options(
        raw_signature,
        ClaudeSignatureValidationOptions::default(),
    )
}

pub fn normalize_claude_provider_native_thinking_signature_with_options(
    raw_signature: &str,
    options: ClaudeSignatureValidationOptions,
) -> Option<String> {
    let signature = strip_provider_prefix(raw_signature);
    if signature.is_empty() || signature.len() > MAX_CLAUDE_THINKING_SIGNATURE_LEN {
        return None;
    }
    match signature.as_bytes()[0] {
        b'E' => validate_single_layer(signature, options.strict).then(|| signature.to_owned()),
        b'R' => {
            let inner = general_purpose::STANDARD.decode(signature).ok()?;
            let inner = std::str::from_utf8(&inner).ok()?;
            validate_single_layer(inner, options.strict).then(|| inner.to_owned())
        }
        _ => None,
    }
}

pub fn validate_claude_thinking_signatures(
    payload: &[u8],
    options: ClaudeSignatureValidationOptions,
) -> Result<(), String> {
    let Ok(root) = serde_json::from_slice::<Value>(payload) else {
        return Ok(());
    };
    let Some(messages) = root.get("messages").and_then(Value::as_array) else {
        return Ok(());
    };
    for (message_index, message) in messages.iter().enumerate() {
        let Some(content) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for (content_index, part) in content.iter().enumerate() {
            if part.get("type").and_then(Value::as_str) != Some("thinking") {
                continue;
            }
            let signature = part
                .get("signature")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim();
            if signature.is_empty() {
                return Err(format!(
                    "messages[{message_index}].content[{content_index}]: missing thinking signature"
                ));
            }
            if normalize_claude_thinking_signature(signature, options).is_none() {
                return Err(format!(
                    "messages[{message_index}].content[{content_index}]: invalid thinking signature"
                ));
            }
        }
    }
    Ok(())
}

pub fn is_valid_claude_cais_signature(raw_signature: &str) -> bool {
    inspect_claude_cais_signature(raw_signature).is_ok()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaudeCaisSignatureInfo {
    pub first_byte: u8,
    pub envelope_version: u64,
    pub channel_id: u64,
    pub model_text: String,
    pub block_kind: String,
    pub context_id: String,
    pub signature_len: usize,
}

pub fn inspect_claude_cais_signature(
    raw_signature: &str,
) -> Result<ClaudeCaisSignatureInfo, String> {
    let signature = strip_provider_prefix(raw_signature);
    if signature.is_empty() {
        return Err("empty signature".to_owned());
    }
    if signature.len() > MAX_CLAUDE_THINKING_SIGNATURE_LEN {
        return Err("signature exceeds maximum length".to_owned());
    }
    if !signature.starts_with('C') {
        return Err("invalid Claude CAIS signature: expected 'C' prefix".to_owned());
    }
    let payload = general_purpose::STANDARD
        .decode(signature)
        .map_err(|_| "invalid Claude CAIS signature: base64 decode failed".to_owned())?;
    if payload.first() != Some(&0x08) {
        return Err("invalid Claude CAIS signature: expected first byte 0x08".to_owned());
    }
    let envelope_version = required_varint(&payload, 1, "top-level envelope version")?;
    if let Some(value) = field(&payload, 3) {
        require_varint(value, "top-level trailer")?;
    }
    let Some(ProtoValue::Bytes(container)) = field(&payload, 2) else {
        return Err("invalid Claude CAIS signature: missing top-level container".to_owned());
    };
    let Some(ProtoValue::Bytes(channel)) = field(container, 1) else {
        return Err("invalid Claude CAIS signature: missing channel block".to_owned());
    };
    let channel_id = required_varint(channel, 1, "channel_id")?;
    if let Some(value) = field(channel, 3) {
        require_varint(value, "channel version")?;
    }
    if let Some(value) = field(channel, 7) {
        require_varint(value, "channel field 7")?;
    }
    let signature_len = match field(channel, 5) {
        Some(ProtoValue::Bytes(bytes)) if !bytes.is_empty() => bytes.len(),
        Some(ProtoValue::Bytes(_)) => {
            return Err("invalid Claude CAIS signature: empty signature bytes".to_owned())
        }
        _ => return Err("invalid Claude CAIS signature: missing signature bytes".to_owned()),
    };
    let model_text = required_utf8(channel, 6, "model_text")?;
    if !model_text.starts_with("claude-") {
        return Err("invalid Claude CAIS signature: model_text must start with claude-".to_owned());
    }
    let block_kind = optional_utf8(channel, 8, "block kind")?.unwrap_or_default();
    let context_id = optional_utf8(channel, 11, "context id")?.unwrap_or_default();
    if !context_id.is_empty() && !is_canonical_uuid(&context_id) {
        return Err("invalid Claude CAIS signature: context id is not a canonical UUID".to_owned());
    }
    Ok(ClaudeCaisSignatureInfo {
        first_byte: payload[0],
        envelope_version,
        channel_id,
        model_text,
        block_kind,
        context_id,
        signature_len,
    })
}

fn required_varint(message: &[u8], number: u64, label: &str) -> Result<u64, String> {
    field(message, number)
        .ok_or_else(|| format!("invalid Claude CAIS signature: missing {label}"))
        .and_then(|value| require_varint(value, label))
}

fn require_varint(value: ProtoValue<'_>, label: &str) -> Result<u64, String> {
    match value {
        ProtoValue::Varint(value) => Ok(value),
        _ => Err(format!(
            "invalid Claude CAIS signature: {label} must be varint"
        )),
    }
}

fn required_utf8(message: &[u8], number: u64, label: &str) -> Result<String, String> {
    optional_utf8(message, number, label)?
        .ok_or_else(|| format!("invalid Claude CAIS signature: missing {label}"))
}

fn optional_utf8(message: &[u8], number: u64, label: &str) -> Result<Option<String>, String> {
    match field(message, number) {
        Some(ProtoValue::Bytes(bytes)) => std::str::from_utf8(bytes)
            .map(|value| Some(value.to_owned()))
            .map_err(|_| format!("invalid Claude CAIS signature: {label} must be valid UTF-8")),
        Some(_) => Err(format!(
            "invalid Claude CAIS signature: {label} must be bytes"
        )),
        None => Ok(None),
    }
}

fn is_canonical_uuid(value: &str) -> bool {
    if value.len() != 36 {
        return false;
    }
    value.bytes().enumerate().all(|(index, byte)| match index {
        8 | 13 | 18 | 23 => byte == b'-',
        _ => byte.is_ascii_hexdigit(),
    })
}

fn validate_single_layer(signature: &str, strict: bool) -> bool {
    let Ok(payload) = general_purpose::STANDARD.decode(signature) else {
        return false;
    };
    if payload.first() != Some(&0x12) {
        return false;
    }
    if !strict {
        return true;
    }
    let Some(ProtoValue::Bytes(container)) = field(&payload, 2) else {
        return false;
    };
    let Some(ProtoValue::Bytes(channel)) = field(container, 1) else {
        return false;
    };
    matches!(field(channel, 1), Some(ProtoValue::Varint(_)))
        && match field(channel, 6) {
            Some(ProtoValue::Bytes(bytes)) => std::str::from_utf8(bytes).is_ok(),
            Some(_) => false,
            None => true,
        }
}

fn strip_provider_prefix(raw_signature: &str) -> &str {
    let signature = raw_signature.trim();
    signature
        .split_once('#')
        .map_or(signature, |(_, payload)| payload.trim())
}

#[derive(Clone, Copy, Debug)]
enum ProtoValue<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Other,
}

fn field(message: &[u8], wanted: u64) -> Option<ProtoValue<'_>> {
    let mut offset = 0;
    let mut found = None;
    while offset < message.len() {
        let tag = consume_varint(message, &mut offset)?;
        let number = tag >> 3;
        let wire_type = tag & 7;
        let value = match wire_type {
            0 => ProtoValue::Varint(consume_varint(message, &mut offset)?),
            1 => {
                offset = offset.checked_add(8)?;
                if offset > message.len() {
                    return None;
                }
                ProtoValue::Other
            }
            2 => {
                let length = consume_varint(message, &mut offset)? as usize;
                let end = offset.checked_add(length)?;
                let bytes = message.get(offset..end)?;
                offset = end;
                ProtoValue::Bytes(bytes)
            }
            5 => {
                offset = offset.checked_add(4)?;
                if offset > message.len() {
                    return None;
                }
                ProtoValue::Other
            }
            _ => return None,
        };
        if number == wanted {
            found = Some(value);
        }
    }
    found
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaudeSignatureTree {
    pub encoding_layers: usize,
    pub channel_id: u64,
    pub field2: Option<u64>,
    pub routing_class: String,
    pub infrastructure_class: String,
    pub schema_features: String,
    pub model_text: String,
    pub legacy_route_hint: String,
    pub has_field7: bool,
}

pub fn inspect_claude_double_layer_signature(
    signature: &str,
) -> Result<ClaudeSignatureTree, String> {
    let decoded = general_purpose::STANDARD
        .decode(signature)
        .map_err(|error| format!("invalid double-layer signature: {error}"))?;
    if decoded.first() != Some(&b'E') {
        return Err("invalid double-layer signature: inner does not start with 'E'".into());
    }
    let inner = std::str::from_utf8(&decoded)
        .map_err(|error| format!("invalid double-layer signature: {error}"))?;
    inspect_claude_single_layer_signature_with_layers(inner, 2)
}

pub fn inspect_claude_single_layer_signature(
    signature: &str,
) -> Result<ClaudeSignatureTree, String> {
    inspect_claude_single_layer_signature_with_layers(signature, 1)
}

fn inspect_claude_single_layer_signature_with_layers(
    signature: &str,
    encoding_layers: usize,
) -> Result<ClaudeSignatureTree, String> {
    let payload = general_purpose::STANDARD
        .decode(signature)
        .map_err(|error| format!("invalid single-layer signature: {error}"))?;
    inspect_claude_signature_payload(&payload, encoding_layers)
}

pub fn inspect_claude_signature_payload(
    payload: &[u8],
    encoding_layers: usize,
) -> Result<ClaudeSignatureTree, String> {
    if payload.first() != Some(&0x12) {
        return Err("invalid Claude signature: expected first byte 0x12".into());
    }
    let container = match field(payload, 2) {
        Some(ProtoValue::Bytes(value)) => value,
        _ => return Err("invalid Claude signature: missing field 2 container".into()),
    };
    let channel = match field(container, 1) {
        Some(ProtoValue::Bytes(value)) => value,
        _ => return Err("invalid Claude signature: missing channel block".into()),
    };
    let channel_id = match field(channel, 1) {
        Some(ProtoValue::Varint(value)) => value,
        _ => return Err("invalid Claude signature: missing channel_id".into()),
    };
    let field2 = match field(channel, 2) {
        Some(ProtoValue::Varint(value)) => Some(value),
        Some(_) => return Err("invalid Claude signature: field2 must be varint".into()),
        None => None,
    };
    let model_text = match field(channel, 6) {
        Some(ProtoValue::Bytes(value)) => std::str::from_utf8(value)
            .map_err(|_| "invalid Claude signature: model_text is not UTF-8")?
            .to_owned(),
        Some(_) => return Err("invalid Claude signature: model_text must be bytes".into()),
        None => String::new(),
    };
    let has_field7 = match field(channel, 7) {
        Some(ProtoValue::Varint(_)) => true,
        Some(_) => return Err("invalid Claude signature: field7 must be varint".into()),
        None => false,
    };
    let routing_class = match channel_id {
        11 => "routing_class_11",
        12 => "routing_class_12",
        _ => "unknown",
    }
    .to_owned();
    let infrastructure_class = match field2 {
        None => "infra_default",
        Some(1) => "infra_aws",
        Some(2) => "infra_google",
        Some(_) => "infra_unknown",
    }
    .to_owned();
    let schema_features = if !model_text.is_empty() {
        "extended_model_tagged_schema"
    } else if !has_field7 && (70..=72).contains(&channel.len()) {
        "compact_schema"
    } else {
        "unknown_schema_features"
    }
    .to_owned();
    let legacy_route_hint = if channel_id == 11 {
        match (field2, encoding_layers) {
            (None, _) => "legacy_default_group",
            (Some(1), _) => "legacy_aws_group",
            (Some(2), 2) => "legacy_vertex_direct",
            (Some(2), 1) => "legacy_vertex_proxy",
            _ => "",
        }
    } else {
        ""
    }
    .to_owned();
    Ok(ClaudeSignatureTree {
        encoding_layers,
        channel_id,
        field2,
        routing_class,
        infrastructure_class,
        schema_features,
        model_text,
        legacy_route_hint,
        has_field7,
    })
}

fn consume_varint(message: &[u8], offset: &mut usize) -> Option<u64> {
    let mut value = 0_u64;
    for shift in (0..=63).step_by(7) {
        let byte = *message.get(*offset)?;
        *offset += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

#[cfg(test)]
pub(crate) fn test_claude_signature() -> String {
    let mut channel = vec![0x08, 0x0c, 0x10, 0x02, 0x32, 0x11];
    channel.extend_from_slice(b"claude-sonnet-4-6");
    let mut container = vec![0x0a, channel.len() as u8];
    container.extend_from_slice(&channel);
    let mut payload = vec![0x12, container.len() as u8];
    payload.extend_from_slice(&container);
    payload.extend_from_slice(&[0x18, 0x01]);
    general_purpose::STANDARD.encode(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn append_bytes(target: &mut Vec<u8>, field: u8, value: &[u8]) {
        target.push((field << 3) | 2);
        target.push(value.len() as u8);
        target.extend_from_slice(value);
    }

    fn append_varint(target: &mut Vec<u8>, field: u8, value: u8) {
        target.push(field << 3);
        target.push(value);
    }

    #[test]
    fn validates_and_normalizes_claude_e_and_r_forms() {
        let single = test_claude_signature();
        let double = general_purpose::STANDARD.encode(single.as_bytes());
        assert!(is_valid_claude_thinking_signature(&single));
        assert_eq!(
            normalize_claude_provider_native_thinking_signature(&double),
            Some(single)
        );
    }

    #[test]
    fn rejects_shallow_e_prefix_without_claude_tree() {
        assert!(!is_valid_claude_thinking_signature("EAAAAA=="));
    }

    #[test]
    fn basic_bypass_accepts_marker_while_strict_requires_tree() {
        let shallow = general_purpose::STANDARD.encode([0x12, 0x00]);
        assert!(has_claude_thinking_signature_prefix(&shallow));
        assert!(has_decodable_claude_thinking_signature(&shallow));
        assert!(normalize_claude_bypass_thinking_signature(&shallow, false).is_some());
        assert!(normalize_claude_bypass_thinking_signature(&shallow, true).is_none());
        assert!(is_valid_claude_thinking_signature(&shallow));
        assert!(!is_valid_claude_thinking_signature_with_options(
            &shallow,
            ClaudeSignatureValidationOptions {
                strict: true,
                ..ClaudeSignatureValidationOptions::default()
            }
        ));
    }

    #[test]
    fn validates_structural_cais_envelope() {
        let mut channel = Vec::new();
        append_varint(&mut channel, 1, 16);
        append_bytes(&mut channel, 5, &[1, 2, 3]);
        append_bytes(&mut channel, 6, b"claude-opus-5");
        append_bytes(&mut channel, 8, b"thinking");
        append_bytes(&mut channel, 11, b"123e4567-e89b-12d3-a456-426614174000");
        let mut container = Vec::new();
        append_bytes(&mut container, 1, &channel);
        let mut payload = Vec::new();
        append_varint(&mut payload, 1, 2);
        append_bytes(&mut payload, 2, &container);
        let signature = general_purpose::STANDARD.encode(payload);
        assert!(is_valid_claude_cais_signature(&signature));
        let info = inspect_claude_cais_signature(&signature).unwrap();
        assert_eq!(info.envelope_version, 2);
        assert_eq!(info.channel_id, 16);
        assert_eq!(info.model_text, "claude-opus-5");
        assert_eq!(info.block_kind, "thinking");
        assert_eq!(info.signature_len, 3);

        assert!(!is_valid_claude_cais_signature("EAAAAA=="));
    }

    #[test]
    fn validates_every_thinking_block_with_index_context() {
        let signature = test_claude_signature();
        let valid = serde_json::json!({
            "messages": [{"content": [
                {"type":"text", "text":"hello"},
                {"type":"thinking", "thinking":"work", "signature":signature}
            ]}]
        });
        validate_claude_thinking_signatures(
            &serde_json::to_vec(&valid).unwrap(),
            ClaudeSignatureValidationOptions {
                strict: true,
                ..Default::default()
            },
        )
        .unwrap();

        let missing = serde_json::json!({
            "messages": [{"content": [{"type":"thinking", "thinking":"work"}]}]
        });
        let error = validate_claude_thinking_signatures(
            &serde_json::to_vec(&missing).unwrap(),
            ClaudeSignatureValidationOptions::default(),
        )
        .unwrap_err();
        assert!(error.contains("messages[0].content[0]"));
    }
}
