// ref: internal/runtime/executor/openai_responses_signature.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

//! Sanitizes reasoning history before forwarding an OpenAI Responses request.
//!
//! The Go implementation rebuilds only the `input` array once an edit is
//! required. `Cow` preserves the same allocation-free, byte-identical no-op
//! behavior while keeping all authority request-local.

use std::borrow::Cow;

use serde_json::Value;

use crate::internal::signature::is_valid_gpt_reasoning_signature;

#[must_use]
pub fn sanitize_openai_responses_reasoning_encrypted_content<'a>(
    _provider: &str,
    body: &'a [u8],
) -> Cow<'a, [u8]> {
    let Ok(mut root) = serde_json::from_slice::<Value>(body) else {
        return Cow::Borrowed(body);
    };
    let store = root.get("store").and_then(Value::as_bool).unwrap_or(false);
    let Some(input) = root.get_mut("input").and_then(Value::as_array_mut) else {
        return Cow::Borrowed(body);
    };

    let mut changed = false;
    for item in input {
        if item.get("type").and_then(Value::as_str).map(str::trim) != Some("reasoning") {
            continue;
        }
        let encrypted = item.get("encrypted_content");
        let valid = encrypted
            .and_then(Value::as_str)
            .is_some_and(|value| value == value.trim() && is_valid_gpt_reasoning_signature(value));
        let encrypted_present = encrypted.is_some();
        let Some(object) = item.as_object_mut() else {
            continue;
        };

        if encrypted_present && !valid {
            object.remove("encrypted_content");
            changed = true;
        }
        if !store && object.contains_key("id") && (!encrypted_present || !valid) {
            object.remove("id");
            changed = true;
        }
    }

    if !changed {
        return Cow::Borrowed(body);
    }
    serde_json::to_vec(&root)
        .map(Cow::Owned)
        .unwrap_or(Cow::Borrowed(body))
}
