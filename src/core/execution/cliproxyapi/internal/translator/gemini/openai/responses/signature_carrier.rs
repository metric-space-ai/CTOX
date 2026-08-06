// ref: internal/translator/gemini/openai/responses/signature_carrier.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};
use serde_json::Value;

use crate::internal::signature::compatible_gemini_signature;

pub(super) const CARRIER_PREFIX: &str = "cpa-gemini-responses-carrier-v1:";
pub(super) const DIRECTION_FIELD: &str = "_cpa_reasoning_direction";
pub(super) const TARGET_FIELD: &str = "_cpa_reasoning_target";
pub(super) const SIGNATURE_FIELD: &str = "_cpa_reasoning_signature";
pub(super) const SUMMARY_FIELD: &str = "_cpa_reasoning_summary";

const MAX_GEMINI_SIGNATURE_LEN: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CarrierDirection {
    Next,
    Previous,
    Standalone,
}

impl CarrierDirection {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "next" => Some(Self::Next),
            "previous" => Some(Self::Previous),
            "standalone" => Some(Self::Standalone),
            _ => None,
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Next => "next",
            Self::Previous => "previous",
            Self::Standalone => "standalone",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CarrierTarget {
    Text,
    Function,
    Any,
}

impl CarrierTarget {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "text" => Some(Self::Text),
            "function" => Some(Self::Function),
            "any" => Some(Self::Any),
            _ => None,
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Function => "function",
            Self::Any => "any",
        }
    }

    fn matches(self, actual: Self) -> bool {
        self == Self::Any || self == actual
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DecodedCarrier {
    signature: String,
    direction: CarrierDirection,
    target: CarrierTarget,
}

#[cfg(test)]
pub(super) fn encode_carrier(
    signature: &str,
    direction: CarrierDirection,
    target: CarrierTarget,
) -> String {
    let signature = signature.trim();
    if signature.is_empty() {
        return String::new();
    }
    format!(
        "{CARRIER_PREFIX}{}:{}:{}",
        direction.as_str(),
        target.as_str(),
        general_purpose::STANDARD_NO_PAD.encode(signature)
    )
}

fn decode_carrier(raw: &str) -> Result<Option<DecodedCarrier>, ()> {
    let raw = raw.trim();
    let Some(encoded) = raw.strip_prefix(CARRIER_PREFIX) else {
        return Ok(None);
    };
    if raw.len() > (MAX_GEMINI_SIGNATURE_LEN * 4 / 3) + 1024 {
        return Err(());
    }
    let mut fields = encoded.splitn(3, ':');
    let direction = CarrierDirection::parse(fields.next().ok_or(())?).ok_or(())?;
    let target = CarrierTarget::parse(fields.next().ok_or(())?).ok_or(())?;
    let payload = fields.next().ok_or(())?;
    let decoded = general_purpose::STANDARD_NO_PAD
        .decode(payload)
        .map_err(|_| ())?;
    if decoded.is_empty() {
        return Err(());
    }
    let signature = String::from_utf8(decoded).map_err(|_| ())?;
    if signature.starts_with(CARRIER_PREFIX) {
        return Err(());
    }
    Ok(Some(DecodedCarrier {
        signature,
        direction,
        target,
    }))
}

pub(super) fn normalize_carriers(items: &[Value]) -> (Vec<Value>, bool) {
    let mut normalized = Vec::with_capacity(items.len());
    let mut has_valid = false;
    for (index, original) in items.iter().enumerate() {
        let mut item = original.clone();
        strip_internal_metadata(&mut item);
        if item.get("type").and_then(Value::as_str) != Some("reasoning") {
            normalized.push(item);
            continue;
        }
        let raw = item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        let decoded = match decode_carrier(raw) {
            Ok(None) => {
                has_valid |= compatible_gemini_signature(raw).is_some();
                normalized.push(item);
                continue;
            }
            Ok(Some(decoded)) => decoded,
            Err(()) => {
                retain_summary_without_signature(item, &mut normalized);
                continue;
            }
        };
        let Some(signature) = compatible_gemini_signature(&decoded.signature) else {
            retain_summary_without_signature(item, &mut normalized);
            continue;
        };
        if decoded.direction != CarrierDirection::Standalone
            && !matches_adjacent(items, index, decoded.direction, decoded.target)
        {
            retain_summary_without_signature(item, &mut normalized);
            continue;
        }
        let detached = is_detached(&item);
        let has_summary = summary_text(&item).is_some_and(|value| !value.trim().is_empty());
        let valid_summary = has_summary
            && ((decoded.direction == CarrierDirection::Standalone
                && matches!(decoded.target, CarrierTarget::Text | CarrierTarget::Any))
                || decoded.direction == CarrierDirection::Next);
        if !detached && !valid_summary {
            retain_summary_without_signature(item, &mut normalized);
            continue;
        }
        if let Some(object) = item.as_object_mut() {
            object.insert("encrypted_content".to_owned(), Value::String(signature));
            object.insert(
                DIRECTION_FIELD.to_owned(),
                Value::String(decoded.direction.as_str().to_owned()),
            );
            object.insert(
                TARGET_FIELD.to_owned(),
                Value::String(decoded.target.as_str().to_owned()),
            );
        }
        has_valid = true;
        normalized.push(item);
    }
    (normalized, has_valid)
}

fn retain_summary_without_signature(mut item: Value, normalized: &mut Vec<Value>) {
    if summary_text(&item).is_some_and(|value| !value.trim().is_empty()) {
        if let Some(object) = item.as_object_mut() {
            object.remove("encrypted_content");
        }
        normalized.push(item);
    }
}

fn strip_internal_metadata(item: &mut Value) {
    if let Some(object) = item.as_object_mut() {
        for field in [
            DIRECTION_FIELD,
            TARGET_FIELD,
            SIGNATURE_FIELD,
            SUMMARY_FIELD,
        ] {
            object.remove(field);
        }
    }
}

fn matches_adjacent(
    items: &[Value],
    index: usize,
    direction: CarrierDirection,
    target: CarrierTarget,
) -> bool {
    let mut adjacent = index as isize
        + if direction == CarrierDirection::Previous {
            -1
        } else {
            1
        };
    while adjacent >= 0 && (adjacent as usize) < items.len() {
        let item = &items[adjacent as usize];
        if let Some(actual) = semantic_target(item) {
            return target.matches(actual);
        }
        if !is_detached(item) {
            return false;
        }
        adjacent += if direction == CarrierDirection::Previous {
            -1
        } else {
            1
        };
    }
    false
}

fn semantic_target(item: &Value) -> Option<CarrierTarget> {
    match item.get("type").and_then(Value::as_str) {
        Some("function_call") => Some(CarrierTarget::Function),
        Some("reasoning") if summary_text(item).is_some_and(|text| !text.trim().is_empty()) => {
            Some(CarrierTarget::Text)
        }
        _ if assistant_visible_text(item) => Some(CarrierTarget::Text),
        _ => None,
    }
}

fn assistant_visible_text(item: &Value) -> bool {
    let role = item.get("role").and_then(Value::as_str).unwrap_or_default();
    match item.get("content") {
        Some(Value::String(_)) => {
            matches!(role.to_ascii_lowercase().as_str(), "assistant" | "model")
        }
        Some(Value::Array(parts)) => parts
            .iter()
            .any(|part| part.get("type").and_then(Value::as_str) == Some("output_text")),
        _ => false,
    }
}

fn summary_text(item: &Value) -> Option<&str> {
    item.pointer("/summary/0/text").and_then(Value::as_str)
}

fn is_detached(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("reasoning")
        && item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
        && summary_text(item).is_none_or(|value| value.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn signature() -> String {
        let opaque = [0x01, 0x0c, 0x39, 0xd6, 0xc7, 0xaa];
        let mut inner = vec![0x0a, opaque.len() as u8];
        inner.extend_from_slice(&opaque);
        let mut outer = vec![0x12, inner.len() as u8];
        outer.extend_from_slice(&inner);
        general_purpose::STANDARD.encode(outer)
    }

    #[test]
    fn codec_round_trips_and_rejects_nested_or_malformed_carriers() {
        let native = signature();
        for (direction, target) in [
            (CarrierDirection::Next, CarrierTarget::Text),
            (CarrierDirection::Previous, CarrierTarget::Function),
            (CarrierDirection::Standalone, CarrierTarget::Any),
        ] {
            let encoded = encode_carrier(&native, direction, target);
            assert_eq!(
                decode_carrier(&encoded),
                Ok(Some(DecodedCarrier {
                    signature: native.clone(),
                    direction,
                    target
                }))
            );
        }
        assert_eq!(
            decode_carrier(&format!("{CARRIER_PREFIX}next:text:!")),
            Err(())
        );
        let nested = encode_carrier(
            &encode_carrier(&native, CarrierDirection::Next, CarrierTarget::Text),
            CarrierDirection::Previous,
            CarrierTarget::Text,
        );
        assert_eq!(decode_carrier(&nested), Err(()));
    }

    #[test]
    fn normalization_unwraps_valid_adjacent_and_drops_malformed_detached() {
        let carrier = encode_carrier(&signature(), CarrierDirection::Next, CarrierTarget::Text);
        let items = json!([
            {"type":"reasoning","encrypted_content":carrier,"summary":[]},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},
            {"type":"reasoning","encrypted_content":format!("{CARRIER_PREFIX}next:text:!"),"summary":[]}
        ]);
        let (normalized, valid) = normalize_carriers(items.as_array().unwrap());
        assert!(valid);
        assert_eq!(normalized.len(), 2);
        assert_eq!(normalized[0][DIRECTION_FIELD], "next");
        assert_eq!(normalized[0][TARGET_FIELD], "text");
        assert_eq!(normalized[0]["encrypted_content"], signature());
    }
}
