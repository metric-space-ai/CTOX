// ref: internal/runtime/executor/helps/codex_input_ids.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{HashMap, HashSet};

use serde_json::value::RawValue;
use serde_json::Value;
use sha2::{Digest, Sha256};

const CODEX_INPUT_ITEM_ID_LIMIT: usize = 64;
const CODEX_MESSAGE_ITEM_ID_PREFIX: &str = "msg";

/// Normalizes message IDs for Codex, removes encrypted reasoning items whose
/// IDs exceed the Codex limit, and deterministically shortens other overlong
/// input item IDs.
///
/// The outer payload and unchanged input items retain their original bytes.
/// This mirrors upstream's `gjson`/`sjson` boundary without a lossy whole-body
/// `serde_json::Value` round trip.
pub fn sanitize_codex_input_item_ids(body: &[u8]) -> Vec<u8> {
    let Some(input_raw) = top_level_raw_field(body, "input") else {
        return body.to_vec();
    };
    let Ok(items) = serde_json::from_str::<Vec<Box<RawValue>>>(input_raw.get()) else {
        return body.to_vec();
    };

    let parsed = items
        .iter()
        .map(|item| serde_json::from_str::<Value>(item.get()).ok())
        .collect::<Vec<_>>();
    let mut occupied = HashSet::with_capacity(items.len());
    for item in parsed.iter().flatten() {
        if should_drop_codex_encrypted_reasoning_item(item) {
            continue;
        }
        if let Some(id) = item.get("id").and_then(Value::as_str) {
            let normalized = normalize_codex_input_item_id(item, id);
            if normalized.chars().count() <= CODEX_INPUT_ITEM_ID_LIMIT {
                occupied.insert(normalized);
            }
        }
    }

    let mut mapped = HashMap::<String, String>::with_capacity(items.len());
    let mut rebuilt = Vec::with_capacity(items.len());
    let mut changed = false;
    for (raw, parsed) in items.iter().zip(&parsed) {
        let Some(item) = parsed else {
            rebuilt.push(raw.get().to_owned());
            continue;
        };
        if should_drop_codex_encrypted_reasoning_item(item) {
            changed = true;
            continue;
        }

        let mut next = item.clone();
        if let Some(original_id) = item.get("id").and_then(Value::as_str) {
            let mut id = normalize_codex_input_item_id(item, original_id);
            if id.chars().count() > CODEX_INPUT_ITEM_ID_LIMIT {
                id = if let Some(existing) = mapped.get(&id) {
                    existing.clone()
                } else {
                    let mut attempt = 0;
                    let shortened = loop {
                        let candidate = shorten_codex_input_item_id_with_attempt(&id, attempt);
                        if !occupied.contains(&candidate) {
                            break candidate;
                        }
                        attempt += 1;
                    };
                    mapped.insert(id.clone(), shortened.clone());
                    occupied.insert(shortened.clone());
                    shortened
                };
            }
            if id != original_id {
                if let Some(object) = next.as_object_mut() {
                    object.insert("id".to_owned(), Value::String(id));
                    rebuilt.push(
                        serde_json::to_string(&next).unwrap_or_else(|_| raw.get().to_owned()),
                    );
                    changed = true;
                    continue;
                }
            }
        }
        rebuilt.push(raw.get().to_owned());
    }

    if !changed {
        return body.to_vec();
    }
    let replacement = format!("[{}]", rebuilt.join(","));
    let input_start = input_raw.get().as_ptr() as usize - body.as_ptr() as usize;
    let input_end = input_start + input_raw.get().len();
    let mut output = Vec::with_capacity(body.len() - (input_end - input_start) + replacement.len());
    output.extend_from_slice(&body[..input_start]);
    output.extend_from_slice(replacement.as_bytes());
    output.extend_from_slice(&body[input_end..]);
    output
}

fn top_level_raw_field<'a>(body: &'a [u8], field: &str) -> Option<&'a RawValue> {
    serde_json::from_slice::<HashMap<&'a str, &'a RawValue>>(body)
        .ok()?
        .get(field)
        .copied()
}

fn normalize_codex_input_item_id(item: &Value, id: &str) -> String {
    if item.get("type").and_then(Value::as_str) != Some("message")
        || id.is_empty()
        || id.starts_with(CODEX_MESSAGE_ITEM_ID_PREFIX)
    {
        return id.to_owned();
    }
    format!("{CODEX_MESSAGE_ITEM_ID_PREFIX}_{id}")
}

fn should_drop_codex_encrypted_reasoning_item(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("reasoning")
        && item
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|id| id.chars().count() > CODEX_INPUT_ITEM_ID_LIMIT)
        && item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty())
}

pub(super) fn shorten_codex_input_item_id_with_attempt(id: &str, attempt: usize) -> String {
    if id.chars().count() <= CODEX_INPUT_ITEM_ID_LIMIT {
        return id.to_owned();
    }
    let mut hash_input = id.as_bytes().to_vec();
    if attempt > 0 {
        hash_input.push(0);
        hash_input.extend_from_slice(attempt.to_string().as_bytes());
    }
    let digest = Sha256::digest(hash_input);
    let suffix = format!("_{}", hex(&digest[..8]));
    let prefix_len = CODEX_INPUT_ITEM_ID_LIMIT - suffix.len();
    let prefix = id.chars().take(prefix_len).collect::<String>();
    format!("{prefix}{suffix}")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
