// ref: internal/translator/antigravity/claude/signature_validation.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};
use serde_json::Value;

use crate::internal::cache::signature_bypass_strict_mode;
use crate::internal::signature::{
    compatible_gemini_signature, has_claude_thinking_signature_prefix,
    inspect_claude_double_layer_signature, inspect_claude_signature_payload,
    inspect_claude_single_layer_signature, normalize_claude_bypass_thinking_signature,
    ClaudeSignatureTree,
};

const MAX_GEMINI_THOUGHT_SIGNATURE_LEN: usize = 32 * 1024 * 1024;
const GEMINI_CLAUDE_CARRIER_PREFIX: &str = "cpa-gemini-carrier-v1:";
const GEMINI_CLAUDE_CARRIER_NEXT: &str = "next";
const GEMINI_CLAUDE_CARRIER_PREVIOUS: &str = "previous";
const GEMINI_CLAUDE_CARRIER_STANDALONE: &str = "standalone";
const GEMINI_CLAUDE_CARRIER_TEXT: &str = "text";
const GEMINI_CLAUDE_CARRIER_FUNCTION: &str = "function";
const GEMINI_CLAUDE_CARRIER_ANY: &str = "any";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeminiClaudeCarrier {
    pub signature: String,
    pub direction: String,
    pub target_kind: String,
    pub marked: bool,
}

pub fn encode_gemini_claude_carrier_signature(
    raw_signature: &str,
    direction: &str,
    target_kind: &str,
) -> String {
    let raw_signature = raw_signature.trim();
    if raw_signature.is_empty() {
        return String::new();
    }
    format!(
        "{GEMINI_CLAUDE_CARRIER_PREFIX}{direction}:{target_kind}:{}",
        general_purpose::STANDARD_NO_PAD.encode(raw_signature)
    )
}

pub fn decode_gemini_claude_carrier_signature(raw_signature: &str) -> Option<GeminiClaudeCarrier> {
    let raw_signature = raw_signature.trim();
    let Some(marked_payload) = raw_signature.strip_prefix(GEMINI_CLAUDE_CARRIER_PREFIX) else {
        return Some(GeminiClaudeCarrier {
            signature: raw_signature.to_owned(),
            direction: String::new(),
            target_kind: String::new(),
            marked: false,
        });
    };
    if raw_signature.len() > (MAX_GEMINI_THOUGHT_SIGNATURE_LEN * 4 / 3) + 1024 {
        return None;
    }
    let mut fields = marked_payload.splitn(3, ':');
    let direction = fields.next()?;
    let target_kind = fields.next()?;
    let encoded = fields.next()?;
    if !matches!(
        direction,
        GEMINI_CLAUDE_CARRIER_NEXT
            | GEMINI_CLAUDE_CARRIER_PREVIOUS
            | GEMINI_CLAUDE_CARRIER_STANDALONE
    ) || !matches!(
        target_kind,
        GEMINI_CLAUDE_CARRIER_TEXT | GEMINI_CLAUDE_CARRIER_FUNCTION | GEMINI_CLAUDE_CARRIER_ANY
    ) {
        return None;
    }
    let decoded = general_purpose::STANDARD_NO_PAD.decode(encoded).ok()?;
    if decoded.is_empty() {
        return None;
    }
    let decoded = String::from_utf8(decoded).ok()?;
    if decoded.starts_with(GEMINI_CLAUDE_CARRIER_PREFIX) {
        return None;
    }
    let signature = compatible_gemini_signature(&decoded)?;
    Some(GeminiClaudeCarrier {
        signature,
        direction: direction.to_owned(),
        target_kind: target_kind.to_owned(),
        marked: true,
    })
}

fn semantic_target_kind(block: &Value) -> Option<&'static str> {
    match block.get("type").and_then(Value::as_str) {
        Some("text") => Some(GEMINI_CLAUDE_CARRIER_TEXT),
        Some("tool_use") => Some(GEMINI_CLAUDE_CARRIER_FUNCTION),
        Some("thinking")
            if !block
                .get("thinking")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .is_empty() =>
        {
            Some(GEMINI_CLAUDE_CARRIER_TEXT)
        }
        _ => None,
    }
}

fn carrier_matches_adjacent(
    blocks: &[Value],
    index: usize,
    direction: &str,
    target_kind: &str,
) -> bool {
    let step = if direction == GEMINI_CLAUDE_CARRIER_PREVIOUS {
        -1_isize
    } else {
        1
    };
    let mut adjacent = index as isize + step;
    while let Some(block) = usize::try_from(adjacent)
        .ok()
        .filter(|position| *position < blocks.len())
        .and_then(|position| blocks.get(position))
    {
        if let Some(kind) = semantic_target_kind(block) {
            return target_kind == GEMINI_CLAUDE_CARRIER_ANY || target_kind == kind;
        }
        let is_empty_thinking = block.get("type").and_then(Value::as_str) == Some("thinking")
            && block
                .get("thinking")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .is_empty();
        if !is_empty_thinking {
            return false;
        }
        adjacent += step;
    }
    false
}

/// Preserves only Claude-facing thinking carriers that are replayable by
/// Gemini and whose directional marker matches adjacent semantic content.
/// Unchanged and invalid payloads retain byte identity.
pub fn strip_invalid_gemini_signature_thinking_blocks(payload: &[u8]) -> Vec<u8> {
    let Ok(mut root) = serde_json::from_slice::<Value>(payload) else {
        return payload.to_vec();
    };
    let Some(messages) = root.get_mut("messages").and_then(Value::as_array_mut) else {
        return payload.to_vec();
    };
    let mut changed = false;
    for message in messages {
        let assistant_message = message
            .get("role")
            .and_then(Value::as_str)
            .is_some_and(|role| role.eq_ignore_ascii_case("assistant"));
        let Some(content) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        let original = content.clone();
        let mut kept = Vec::with_capacity(original.len());
        let mut content_changed = false;
        let mut pending_target = String::new();
        for (index, block) in original.iter().enumerate() {
            if block.get("type").and_then(Value::as_str) == Some("thinking") {
                let raw_signature = block
                    .get("signature")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim();
                let thinking_text = block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim();
                if raw_signature.is_empty()
                    && !thinking_text.is_empty()
                    && matches!(
                        pending_target.as_str(),
                        GEMINI_CLAUDE_CARRIER_ANY | GEMINI_CLAUDE_CARRIER_TEXT
                    )
                {
                    pending_target.clear();
                    kept.push(block.clone());
                    continue;
                }
                let Some(carrier) = decode_gemini_claude_carrier_signature(raw_signature) else {
                    pending_target.clear();
                    content_changed = true;
                    continue;
                };
                let invalid_marked_placement = carrier.marked
                    && match carrier.direction.as_str() {
                        GEMINI_CLAUDE_CARRIER_NEXT | GEMINI_CLAUDE_CARRIER_PREVIOUS => {
                            !carrier_matches_adjacent(
                                &original,
                                index,
                                &carrier.direction,
                                &carrier.target_kind,
                            )
                        }
                        GEMINI_CLAUDE_CARRIER_STANDALONE => {
                            !thinking_text.is_empty()
                                && carrier.target_kind == GEMINI_CLAUDE_CARRIER_FUNCTION
                        }
                        _ => false,
                    };
                let invalid_previous_with_text = carrier.marked
                    && !thinking_text.is_empty()
                    && carrier.direction == GEMINI_CLAUDE_CARRIER_PREVIOUS;
                if !assistant_message
                    || invalid_marked_placement
                    || invalid_previous_with_text
                    || compatible_gemini_signature(&carrier.signature).is_none()
                {
                    pending_target.clear();
                    content_changed = true;
                    continue;
                }
                if carrier.marked && carrier.direction == GEMINI_CLAUDE_CARRIER_NEXT {
                    pending_target.clone_from(&carrier.target_kind);
                } else {
                    pending_target.clear();
                }
            } else {
                pending_target.clear();
            }
            kept.push(block.clone());
        }
        if content_changed {
            *content = kept;
            changed = true;
        }
    }
    if !changed {
        return payload.to_vec();
    }
    serde_json::to_vec(&root).unwrap_or_else(|_| payload.to_vec())
}

/// Preserves legacy Claude thinking blocks by E/R prefix only. This deliberately
/// mirrors the executor's shallow first cleanup before cache/bypass policy is
/// selected.
pub fn strip_empty_signature_thinking_blocks(payload: &[u8]) -> Vec<u8> {
    strip_claude_thinking_blocks(payload, has_claude_thinking_signature_prefix)
}

/// Applies the configured basic/strict Antigravity bypass validator and removes
/// thinking blocks that cannot be replayed.
pub fn strip_invalid_bypass_signature_thinking_blocks(payload: &[u8]) -> Vec<u8> {
    let strict = signature_bypass_strict_mode();
    strip_claude_thinking_blocks(payload, |signature| {
        normalize_claude_bypass_thinking_signature(signature, strict).is_some()
    })
}

pub fn validate_claude_bypass_signatures(payload: &[u8]) -> Result<(), String> {
    let root: Value = serde_json::from_slice(payload).map_err(|error| error.to_string())?;
    let strict = signature_bypass_strict_mode();
    for (message_index, message) in root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        for (content_index, part) in message
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
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
            if normalize_claude_bypass_thinking_signature(signature, strict).is_none() {
                return Err(format!(
                    "messages[{message_index}].content[{content_index}]: invalid thinking signature"
                ));
            }
        }
    }
    Ok(())
}

pub fn normalize_claude_bypass_signature(raw_signature: &str) -> Option<String> {
    normalize_claude_bypass_thinking_signature(raw_signature, signature_bypass_strict_mode())
}

fn strip_claude_thinking_blocks(payload: &[u8], mut valid: impl FnMut(&str) -> bool) -> Vec<u8> {
    let Ok(mut root) = serde_json::from_slice::<Value>(payload) else {
        return payload.to_vec();
    };
    let Some(messages) = root.get_mut("messages").and_then(Value::as_array_mut) else {
        return payload.to_vec();
    };
    let mut changed = false;
    for message in messages {
        let Some(content) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        let before = content.len();
        content.retain(|part| {
            part.get("type").and_then(Value::as_str) != Some("thinking")
                || valid(
                    part.get("signature")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                )
        });
        changed |= content.len() != before;
    }
    if !changed {
        payload.to_vec()
    } else {
        serde_json::to_vec(&root).unwrap_or_else(|_| payload.to_vec())
    }
}

pub fn inspect_double_layer_signature(signature: &str) -> Result<ClaudeSignatureTree, String> {
    inspect_claude_double_layer_signature(signature)
}

pub fn inspect_single_layer_signature(signature: &str) -> Result<ClaudeSignatureTree, String> {
    inspect_claude_single_layer_signature(signature)
}

pub fn inspect_signature_payload(
    payload: &[u8],
    encoding_layers: usize,
) -> Result<ClaudeSignatureTree, String> {
    inspect_claude_signature_payload(payload, encoding_layers)
}
