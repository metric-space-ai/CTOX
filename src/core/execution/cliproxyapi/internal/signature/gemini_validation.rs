// ref: internal/signature/gemini_validation.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};
use serde_json::Value;

use super::is_valid_claude_cais_signature;

pub const MAX_GEMINI_THOUGHT_SIGNATURE_LEN: usize = 32 * 1024 * 1024;
pub const GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR: &str = "skip_thought_signature_validator";
pub const GEMINI_CONTEXT_ENGINEERING_BYPASS: &str = "context_engineering_is_the_way_to_go";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GeminiThoughtSignatureValidationOptions {
    pub allow_bypass_sentinel: bool,
    pub require_known_envelope: bool,
    pub require_observed_marker: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GeminiThoughtSignatureEnvelope {
    #[default]
    Unknown,
    ProtobufField2,
    AsciiUuid,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GeminiThoughtSignatureInfo {
    pub is_bypass_sentinel: bool,
    pub bypass_sentinel: String,
    pub decoded_len: usize,
    pub first_byte: u8,
    pub has_observed_marker: bool,
    pub known_envelope: bool,
    pub envelope: GeminiThoughtSignatureEnvelope,
    pub record_count: usize,
    pub opaque_payload_len: usize,
}

pub fn is_gemini_thought_signature_bypass(raw_signature: &str) -> bool {
    matches!(
        raw_signature.trim(),
        GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR | GEMINI_CONTEXT_ENGINEERING_BYPASS
    )
}

pub fn is_valid_gemini_thought_signature(
    raw_signature: &str,
    options: GeminiThoughtSignatureValidationOptions,
) -> bool {
    inspect_gemini_thought_signature(raw_signature, options).is_ok()
}

pub fn inspect_gemini_thought_signature(
    raw_signature: &str,
    options: GeminiThoughtSignatureValidationOptions,
) -> Result<GeminiThoughtSignatureInfo, String> {
    let signature = raw_signature.trim();
    if signature.is_empty() {
        return Err("empty Gemini thought signature".to_owned());
    }
    if is_valid_claude_cais_signature(signature) {
        return Err("invalid Gemini thought signature: detected Claude CAIS signature".to_owned());
    }
    if is_gemini_thought_signature_bypass(signature) {
        if !options.allow_bypass_sentinel {
            return Err("Gemini thought signature bypass sentinel is not allowed".to_owned());
        }
        return Ok(GeminiThoughtSignatureInfo {
            is_bypass_sentinel: true,
            bypass_sentinel: signature.to_owned(),
            ..GeminiThoughtSignatureInfo::default()
        });
    }
    if signature.len() > MAX_GEMINI_THOUGHT_SIGNATURE_LEN {
        return Err(format!(
            "Gemini thought signature exceeds maximum length ({MAX_GEMINI_THOUGHT_SIGNATURE_LEN} bytes)"
        ));
    }
    let decoded = general_purpose::STANDARD
        .decode(signature)
        .or_else(|_| general_purpose::STANDARD_NO_PAD.decode(signature))
        .map_err(|_| "invalid Gemini thought signature: base64 decode failed".to_owned())?;
    let Some(first_byte) = decoded.first().copied() else {
        return Err("invalid Gemini thought signature: empty decoded payload".to_owned());
    };
    let envelope = classify_envelope(&decoded);
    let known_envelope = envelope == GeminiThoughtSignatureEnvelope::ProtobufField2;
    let (record_count, opaque_payload_len) = inspect_envelope(&decoded, envelope);
    if options.require_known_envelope && !known_envelope {
        return Err(format!(
            "invalid Gemini thought signature: unknown envelope {envelope:?}"
        ));
    }
    if options.require_observed_marker && first_byte != 0x12 {
        return Err(format!(
            "invalid Gemini thought signature: expected observed marker 0x12, got 0x{first_byte:02x}"
        ));
    }
    Ok(GeminiThoughtSignatureInfo {
        decoded_len: decoded.len(),
        first_byte,
        has_observed_marker: first_byte == 0x12,
        known_envelope,
        envelope,
        record_count,
        opaque_payload_len,
        ..GeminiThoughtSignatureInfo::default()
    })
}

pub fn validate_gemini_thought_signatures(
    input: &[u8],
    options: GeminiThoughtSignatureValidationOptions,
) -> Result<(), String> {
    let Ok(document) = std::str::from_utf8(input) else {
        return Ok(());
    };
    let root = gjson::parse(document);
    let direct_contents = root.get("contents");
    let (contents, path) = if direct_contents.exists() {
        (direct_contents, "contents")
    } else {
        (root.get("request.contents"), "request.contents")
    };
    if contents.kind() != gjson::Kind::Array {
        return Ok(());
    }
    for (content_index, content) in contents.array().into_iter().enumerate() {
        let parts = content.get("parts");
        if parts.kind() != gjson::Kind::Array {
            continue;
        }
        let is_model = content
            .get("role")
            .str()
            .trim()
            .eq_ignore_ascii_case("model");
        let mut first_function_call_seen = false;
        for (part_index, part) in parts.array().into_iter().enumerate() {
            let has_call = part.get("functionCall").exists();
            let is_first_call = is_model && has_call && !first_function_call_seen;
            if is_model && has_call {
                first_function_call_seen = true;
            }
            let signature = first_gjson_signature(&part);
            if !has_call && signature.is_none() {
                continue;
            }
            let part_path = format!("{path}[{content_index}].parts[{part_index}]");
            if part.get("functionResponse").exists() && signature.is_some() {
                return Err(format!(
                    "{part_path}: functionResponse must not carry thoughtSignature"
                ));
            }
            let raw = signature.as_ref().map_or("", |value| value.str().trim());
            if raw.is_empty() {
                if is_first_call {
                    return Err(format!(
                        "{part_path}: missing thoughtSignature on first functionCall"
                    ));
                }
                if signature.is_some() {
                    return Err(format!("{part_path}: empty thoughtSignature"));
                }
                continue;
            }
            if is_gemini_thought_signature_bypass(raw) && !is_first_call {
                return Err(format!(
                    "{part_path}: Gemini bypass sentinel is allowed only on the first model functionCall"
                ));
            }
            if !has_normalized_gjson_signature(&part, raw) {
                return Err(format!(
                    "{part_path}: thoughtSignature must use one canonical top-level field"
                ));
            }
            inspect_gemini_thought_signature(raw, options)
                .map_err(|error| format!("{part_path}: {error}"))?;
        }
    }
    Ok(())
}

const GEMINI_SIGNATURE_PATHS: [&str; 7] = [
    "thoughtSignature",
    "thought_signature",
    "functionCall.thoughtSignature",
    "functionCall.thought_signature",
    "functionResponse.thoughtSignature",
    "functionResponse.thought_signature",
    "extra_content.google.thought_signature",
];

fn first_gjson_signature<'a>(part: &'a gjson::Value<'a>) -> Option<gjson::Value<'a>> {
    GEMINI_SIGNATURE_PATHS.iter().find_map(|path| {
        let result = part.get(path);
        result.exists().then_some(result)
    })
}

fn has_normalized_gjson_signature(part: &gjson::Value<'_>, replay_signature: &str) -> bool {
    let mut canonical_count = 0;
    part.each(|key, _| {
        if key.str() == "thoughtSignature" {
            canonical_count += 1;
        }
        true
    });
    let canonical = part.get("thoughtSignature");
    canonical_count == 1
        && canonical.kind() == gjson::Kind::String
        && canonical.str() == replay_signature
        && GEMINI_SIGNATURE_PATHS[1..]
            .iter()
            .all(|path| !part.get(path).exists())
}

pub fn validate_gemini_function_call_pairing(input: &[u8]) -> Result<(), String> {
    let Ok(root) = serde_json::from_slice::<Value>(input) else {
        return Ok(());
    };
    let (contents, path) = gemini_contents(&root);
    let Some(contents) = contents.and_then(Value::as_array) else {
        return Ok(());
    };
    let mut pending: Vec<FunctionCallRef> = Vec::new();
    for (content_index, content) in contents.iter().enumerate() {
        let Some(parts) = content.get("parts").and_then(Value::as_array) else {
            if !pending.is_empty() {
                return Err(format!(
                    "{path}[{content_index}]: content appears before {} pending functionResponse part(s)",
                    pending.len()
                ));
            }
            continue;
        };
        let mut calls = Vec::new();
        let mut responses = Vec::new();
        for (part_index, part) in parts.iter().enumerate() {
            let part_path = format!("{path}[{content_index}].parts[{part_index}]");
            if let Some(call) = part.get("functionCall") {
                let name = json_string(call, "name");
                if name.is_empty() {
                    return Err(format!("{part_path}: missing functionCall.name"));
                }
                calls.push(FunctionCallRef {
                    id: json_string(call, "id"),
                    name,
                    path: part_path.clone(),
                });
            }
            if let Some(response) = part.get("functionResponse") {
                responses.push((response, part_path));
            }
        }
        if !calls.is_empty() && !responses.is_empty() {
            return Err(format!(
                "{path}[{content_index}]: functionCall and functionResponse parts must not be interleaved in the same content"
            ));
        }
        if !calls.is_empty() {
            if !pending.is_empty() {
                return Err(format!(
                    "{path}[{content_index}]: functionCall appears before {} pending functionResponse part(s)",
                    pending.len()
                ));
            }
            pending = calls;
            continue;
        }
        if responses.is_empty() {
            if !pending.is_empty() {
                return Err(format!(
                    "{path}[{content_index}]: content appears before {} pending functionResponse part(s)",
                    pending.len()
                ));
            }
            continue;
        }
        if pending.is_empty() {
            return Err(format!(
                "{path}[{content_index}]: functionResponse without preceding functionCall"
            ));
        }
        if responses.len() != pending.len() {
            return Err(format!(
                "{path}[{content_index}]: functionResponse count {} does not match pending functionCall count {}",
                responses.len(),
                pending.len()
            ));
        }
        for ((response, response_path), call) in responses.into_iter().zip(&pending) {
            let response_id = json_string(response, "id");
            let response_name = json_string(response, "name");
            if !call.id.is_empty() && response_id.is_empty() {
                return Err(format!(
                    "{response_path}: missing functionResponse.id for {}",
                    call.path
                ));
            }
            if !call.id.is_empty() && response_id != call.id {
                return Err(format!(
                    "{response_path}: functionResponse.id {response_id:?} does not match functionCall.id {:?} at {}",
                    call.id, call.path
                ));
            }
            if response_name.is_empty() {
                return Err(format!("{response_path}: missing functionResponse.name"));
            }
            if response_name != call.name {
                return Err(format!(
                    "{response_path}: functionResponse.name {response_name:?} does not match functionCall.name {:?} at {}",
                    call.name, call.path
                ));
            }
        }
        pending.clear();
    }
    Ok(())
}

#[derive(Debug)]
struct FunctionCallRef {
    id: String,
    name: String,
    path: String,
}

fn json_string(value: &Value, key: &str) -> String {
    match value.get(key) {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Number(value)) => value.to_string(),
        Some(Value::Bool(value)) => value.to_string(),
        _ => String::new(),
    }
}

fn gemini_contents(root: &Value) -> (Option<&Value>, &'static str) {
    if let Some(contents) = root.get("contents") {
        (Some(contents), "contents")
    } else {
        (root.pointer("/request/contents"), "request.contents")
    }
}

fn classify_envelope(decoded: &[u8]) -> GeminiThoughtSignatureEnvelope {
    if is_ascii_uuid(decoded) {
        GeminiThoughtSignatureEnvelope::AsciiUuid
    } else if inspect_field2_envelope(decoded).is_some() {
        GeminiThoughtSignatureEnvelope::ProtobufField2
    } else {
        GeminiThoughtSignatureEnvelope::Unknown
    }
}

fn inspect_envelope(decoded: &[u8], envelope: GeminiThoughtSignatureEnvelope) -> (usize, usize) {
    if envelope == GeminiThoughtSignatureEnvelope::ProtobufField2 {
        if let Some(value) = inspect_field2_envelope(decoded) {
            return (1, value.len());
        }
    }
    (0, 0)
}

pub(crate) fn recognized_gemini_provider_signature(raw_signature: &str) -> bool {
    !is_valid_claude_cais_signature(raw_signature)
        && is_valid_gemini_thought_signature(
            raw_signature,
            GeminiThoughtSignatureValidationOptions {
                require_known_envelope: true,
                ..GeminiThoughtSignatureValidationOptions::default()
            },
        )
}

fn inspect_field2_envelope(decoded: &[u8]) -> Option<&[u8]> {
    let outer = consume_length_delimited_exact(decoded, 2)?;
    let value = consume_length_delimited_exact(outer, 1)?;
    (!value.is_empty() && (value[0] == 0x01 || is_ascii_uuid(value))).then_some(value)
}

fn consume_length_delimited_exact(input: &[u8], expected_field: u64) -> Option<&[u8]> {
    let (tag, tag_len) = consume_varint(input)?;
    if tag >> 3 != expected_field || tag & 7 != 2 {
        return None;
    }
    let (length, length_len) = consume_varint(input.get(tag_len..)?)?;
    let start = tag_len.checked_add(length_len)?;
    let end = start.checked_add(usize::try_from(length).ok()?)?;
    (end == input.len()).then(|| &input[start..end])
}

fn consume_varint(input: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0_u64;
    for (index, byte) in input.iter().copied().take(10).enumerate() {
        value |= u64::from(byte & 0x7f) << (index * 7);
        if byte & 0x80 == 0 {
            return Some((value, index + 1));
        }
    }
    None
}

fn is_ascii_uuid(input: &[u8]) -> bool {
    input.len() == 36
        && input.iter().copied().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
}
