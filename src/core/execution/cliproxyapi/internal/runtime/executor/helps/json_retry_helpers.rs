// ref: internal/runtime/executor/helps/json_retry_helpers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

use chrono::TimeDelta;
use serde_json::Value;

/// Retry error bodies are diagnostic input, not an unbounded allocation lane.
pub const MAX_RETRY_ERROR_BODY_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryDelayError {
    BodyTooLarge,
    InvalidDuration,
    MissingRetryInfo,
}

impl fmt::Display for RetryDelayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::BodyTooLarge => "retry response body exceeds limit",
            Self::InvalidDuration => "failed to parse duration",
            Self::MissingRetryInfo => "no RetryInfo found",
        })
    }
}

impl std::error::Error for RetryDelayError {}

/// Removes a top-level or nested JSON field. Owned input makes the upstream
/// byte-slice no-op guarantee explicit: absent, empty-key, empty-body, and
/// invalid-JSON paths return the same allocation unchanged.
#[must_use]
pub fn delete_json_field(mut body: Vec<u8>, key: &str) -> Vec<u8> {
    if key.is_empty() || body.is_empty() {
        return body;
    }
    let Ok(mut document) = serde_json::from_slice::<Value>(&body) else {
        return body;
    };
    if !delete_path(&mut document, &key.split('.').collect::<Vec<_>>()) {
        return body;
    }
    if let Ok(updated) = serde_json::to_vec(&document) {
        body = updated;
    }
    body
}

/// Extracts the retry delay from a Google API 429 response. A signed
/// `TimeDelta` preserves Go `time.Duration` semantics, including negative
/// values, without permitting arithmetic overflow.
pub fn parse_retry_delay(error_body: &[u8]) -> Result<TimeDelta, RetryDelayError> {
    if error_body.len() > MAX_RETRY_ERROR_BODY_BYTES {
        return Err(RetryDelayError::BodyTooLarge);
    }
    let document = serde_json::from_slice::<Value>(error_body).ok();
    if let Some(details) = document
        .as_ref()
        .and_then(|root| root.pointer("/error/details"))
        .and_then(Value::as_array)
    {
        for detail in details {
            if detail.get("@type").and_then(Value::as_str)
                != Some("type.googleapis.com/google.rpc.RetryInfo")
            {
                continue;
            }
            let Some(raw) = detail
                .get("retryDelay")
                .and_then(Value::as_str)
                .filter(|raw| !raw.is_empty())
            else {
                continue;
            };
            return parse_go_duration(raw).ok_or(RetryDelayError::InvalidDuration);
        }
        for detail in details {
            if detail.get("@type").and_then(Value::as_str)
                != Some("type.googleapis.com/google.rpc.ErrorInfo")
            {
                continue;
            }
            let Some(raw) = detail
                .pointer("/metadata/quotaResetDelay")
                .and_then(Value::as_str)
                .filter(|raw| !raw.is_empty())
            else {
                continue;
            };
            if let Some(duration) = parse_go_duration(raw) {
                return Ok(duration);
            }
        }
    }

    let message = document
        .as_ref()
        .and_then(|root| root.pointer("/error/message"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if let Some(seconds) = retry_after_seconds(message) {
        return TimeDelta::try_seconds(seconds).ok_or(RetryDelayError::InvalidDuration);
    }
    if let Some(raw) = retry_after_human_duration(&message.to_ascii_lowercase()) {
        if let Some(duration) =
            parse_go_duration(raw).filter(|duration| *duration > TimeDelta::zero())
        {
            return Ok(duration);
        }
    }
    Err(RetryDelayError::MissingRetryInfo)
}

fn delete_path(document: &mut Value, segments: &[&str]) -> bool {
    let Some((head, tail)) = segments.split_first() else {
        return false;
    };
    if tail.is_empty() {
        return document
            .as_object_mut()
            .is_some_and(|object| object.remove(*head).is_some());
    }
    document
        .as_object_mut()
        .and_then(|object| object.get_mut(*head))
        .is_some_and(|child| delete_path(child, tail))
}

fn retry_after_seconds(message: &str) -> Option<i64> {
    for (index, _) in message.match_indices("after") {
        let remainder = message.get(index + "after".len()..)?;
        let whitespace = remainder
            .bytes()
            .take_while(|byte| byte.is_ascii_whitespace())
            .count();
        if whitespace == 0 {
            continue;
        }
        let remainder = &remainder[whitespace..];
        let digits = remainder.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && remainder.as_bytes().get(digits) == Some(&b's') {
            return remainder[..digits].parse().ok();
        }
    }
    None
}

fn retry_after_human_duration(message: &str) -> Option<&str> {
    for (index, _) in message.match_indices("after") {
        let remainder = message.get(index + "after".len()..)?;
        let whitespace = remainder
            .bytes()
            .take_while(|byte| byte.is_ascii_whitespace())
            .count();
        if whitespace == 0 {
            continue;
        }
        let remainder = &remainder[whitespace..];
        let end = remainder
            .bytes()
            .take_while(|byte| byte.is_ascii_digit() || matches!(byte, b'h' | b'm' | b's'))
            .count();
        if end > 0 {
            return Some(&remainder[..end]);
        }
    }
    None
}

fn parse_go_duration(raw: &str) -> Option<TimeDelta> {
    if raw == "0" {
        return Some(TimeDelta::zero());
    }
    let bytes = raw.as_bytes();
    let (negative, mut index) = match bytes.first() {
        Some(b'-') => (true, 1),
        Some(b'+') => (false, 1),
        _ => (false, 0),
    };
    let mut total = 0_i128;
    let mut components = 0;
    while index < bytes.len() {
        let whole_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        let whole = if index == whole_start {
            0_i128
        } else {
            raw[whole_start..index].parse::<i128>().ok()?
        };
        let mut fraction = 0_i128;
        let mut scale = 1_i128;
        if bytes.get(index) == Some(&b'.') {
            index += 1;
            let fraction_start = index;
            while bytes.get(index).is_some_and(u8::is_ascii_digit) {
                fraction = fraction
                    .checked_mul(10)?
                    .checked_add(i128::from(bytes[index] - b'0'))?;
                scale = scale.checked_mul(10)?;
                index += 1;
            }
            if whole_start == index - 1 && fraction_start == index {
                return None;
            }
        } else if index == whole_start {
            return None;
        }
        let (unit_nanos, unit_len) = duration_unit(&raw[index..])?;
        index += unit_len;
        let nanos = whole
            .checked_mul(unit_nanos)?
            .checked_add(fraction.checked_mul(unit_nanos)?.checked_div(scale)?)?;
        total = total.checked_add(nanos)?;
        components += 1;
    }
    if components == 0 {
        return None;
    }
    if negative {
        total = total.checked_neg()?;
    }
    let nanos = i64::try_from(total).ok()?;
    Some(TimeDelta::nanoseconds(nanos))
}

fn duration_unit(raw: &str) -> Option<(i128, usize)> {
    [
        ("ns", 1_i128),
        ("us", 1_000),
        ("µs", 1_000),
        ("μs", 1_000),
        ("ms", 1_000_000),
        ("s", 1_000_000_000),
        ("m", 60 * 1_000_000_000),
        ("h", 60 * 60 * 1_000_000_000),
    ]
    .into_iter()
    .find_map(|(unit, nanos)| raw.starts_with(unit).then_some((nanos, unit.len())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delete_is_exact_noop_and_removes_nested_fields() {
        for (body, key) in [
            (br#" { "a" : 1 } "#.as_slice(), "missing"),
            (b"not-json", "a"),
            (b"{}", ""),
        ] {
            let input = body.to_vec();
            let pointer = input.as_ptr();
            let output = delete_json_field(input, key);
            assert_eq!(output.as_ptr(), pointer);
            assert_eq!(output, body);
        }
        let output = delete_json_field(br#"{"a":{"b":1,"c":2}}"#.to_vec(), "a.b");
        assert_eq!(
            serde_json::from_slice::<Value>(&output).unwrap(),
            serde_json::json!({"a":{"c":2}})
        );
    }

    #[test]
    fn retry_priority_and_duration_grammar_match_go() {
        let retry = br#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","metadata":{"quotaResetDelay":"9s"}},{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"1h2m3.5s"}]}}"#;
        assert_eq!(
            parse_retry_delay(retry).unwrap(),
            TimeDelta::milliseconds(3_723_500)
        );
        assert_eq!(
            parse_go_duration("-1.25s"),
            Some(TimeDelta::milliseconds(-1_250))
        );
        assert_eq!(
            parse_go_duration("250us"),
            Some(TimeDelta::microseconds(250))
        );
    }

    #[test]
    fn retry_fallbacks_errors_and_bound_are_deterministic() {
        assert_eq!(
            parse_retry_delay(br#"{"error":{"message":"retry after 90s."}}"#).unwrap(),
            TimeDelta::seconds(90)
        );
        assert_eq!(
            parse_retry_delay(br#"{"error":{"message":"Retry AFTER 1h2m3s."}}"#).unwrap(),
            TimeDelta::seconds(3_723)
        );
        let invalid_primary = br#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"bad"},{"@type":"type.googleapis.com/google.rpc.ErrorInfo","metadata":{"quotaResetDelay":"1s"}}]}}"#;
        assert_eq!(
            parse_retry_delay(invalid_primary),
            Err(RetryDelayError::InvalidDuration)
        );
        assert_eq!(
            parse_retry_delay(b"{}"),
            Err(RetryDelayError::MissingRetryInfo)
        );
        assert_eq!(
            parse_retry_delay(&vec![b'x'; MAX_RETRY_ERROR_BODY_BYTES + 1]),
            Err(RetryDelayError::BodyTooLarge)
        );
    }
}
