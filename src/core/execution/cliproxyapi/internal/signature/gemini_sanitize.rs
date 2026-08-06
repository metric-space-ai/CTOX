// ref: internal/signature/gemini_sanitize.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::compatible_gemini_signature;

pub const GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR: &str = "skip_thought_signature_validator";

pub fn sanitize_gemini_request_thought_signatures(payload: &[u8]) -> Vec<u8> {
    let Ok(mut root) = serde_json::from_slice::<Value>(payload) else {
        return payload.to_vec();
    };
    let Some(contents) = root.get_mut("contents").and_then(Value::as_array_mut) else {
        return payload.to_vec();
    };
    let mut changed = false;

    for content in contents {
        let is_model = content.get("role").and_then(Value::as_str) == Some("model");
        let Some(parts) = content.get_mut("parts").and_then(Value::as_array_mut) else {
            continue;
        };
        let mut first_function_call_seen = false;
        for part in parts {
            let has_function_response = part.get("functionResponse").is_some();
            let has_function_call = part.get("functionCall").is_some();
            let signature = part_signature(part).map(str::to_owned);

            if has_function_response {
                if signature.is_some() {
                    changed |= remove_signature_fields(part);
                }
                continue;
            }
            if !is_model {
                continue;
            }

            let first_function_call = has_function_call && !first_function_call_seen;
            if has_function_call {
                first_function_call_seen = true;
            }
            if !has_function_call && signature.is_none() {
                continue;
            }

            let replay = if first_function_call {
                signature
                    .as_deref()
                    .and_then(compatible_gemini_signature)
                    .unwrap_or_else(|| GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR.to_owned())
                    .into()
            } else {
                signature
                    .as_deref()
                    .filter(|value| {
                        value.trim() != GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR
                            && !matches!(
                                value.trim().split_once('#'),
                                Some((
                                    "gemini" | "google",
                                    GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR
                                ))
                            )
                    })
                    .and_then(compatible_gemini_signature)
            };

            match replay {
                Some(replay) => {
                    if !has_canonical_signature_only(part, &replay) {
                        remove_signature_fields(part);
                        if let Some(object) = part.as_object_mut() {
                            object.insert("thoughtSignature".to_owned(), Value::String(replay));
                            changed = true;
                        }
                    }
                }
                None if signature.is_some() => changed |= remove_signature_fields(part),
                None => {}
            }
        }
    }

    if !changed {
        payload.to_vec()
    } else {
        serde_json::to_vec(&root).unwrap_or_else(|_| payload.to_vec())
    }
}

fn part_signature(part: &Value) -> Option<&str> {
    part.get("thoughtSignature")
        .or_else(|| part.get("thought_signature"))
        .or_else(|| part.pointer("/functionCall/thoughtSignature"))
        .or_else(|| part.pointer("/functionCall/thought_signature"))
        .or_else(|| part.pointer("/functionResponse/thoughtSignature"))
        .or_else(|| part.pointer("/functionResponse/thought_signature"))
        .or_else(|| part.pointer("/extra_content/google/thought_signature"))
        .and_then(Value::as_str)
}

fn has_canonical_signature_only(part: &Value, expected: &str) -> bool {
    part.get("thoughtSignature").and_then(Value::as_str) == Some(expected)
        && part.get("thought_signature").is_none()
        && part.pointer("/functionCall/thoughtSignature").is_none()
        && part.pointer("/functionCall/thought_signature").is_none()
        && part.pointer("/functionResponse/thoughtSignature").is_none()
        && part
            .pointer("/functionResponse/thought_signature")
            .is_none()
        && part
            .pointer("/extra_content/google/thought_signature")
            .is_none()
}

fn remove_signature_fields(part: &mut Value) -> bool {
    let Some(object) = part.as_object_mut() else {
        return false;
    };
    let mut changed = object.remove("thoughtSignature").is_some();
    changed |= object.remove("thought_signature").is_some();
    for key in ["functionCall", "functionResponse"] {
        if let Some(nested) = object.get_mut(key).and_then(Value::as_object_mut) {
            changed |= nested.remove("thoughtSignature").is_some();
            changed |= nested.remove("thought_signature").is_some();
        }
    }
    if let Some(google) = object
        .get_mut("extra_content")
        .and_then(Value::as_object_mut)
        .and_then(|extra| extra.get_mut("google"))
        .and_then(Value::as_object_mut)
    {
        changed |= google.remove("thought_signature").is_some();
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_bypass_only_to_first_parallel_call_and_drops_response_signature() {
        let output = sanitize_gemini_request_thought_signatures(
            br#"{"contents":[{"role":"model","parts":[{"functionCall":{"name":"a"}},{"functionCall":{"name":"b"}}]},{"role":"user","parts":[{"functionResponse":{"name":"a","thoughtSignature":"bad"},"thoughtSignature":"bad"}]}]}"#,
        );
        let output: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(
            output["contents"][0]["parts"][0]["thoughtSignature"],
            GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR
        );
        assert!(output["contents"][0]["parts"][1]
            .get("thoughtSignature")
            .is_none());
        assert!(output["contents"][1]["parts"][0]
            .get("thoughtSignature")
            .is_none());
    }
}
