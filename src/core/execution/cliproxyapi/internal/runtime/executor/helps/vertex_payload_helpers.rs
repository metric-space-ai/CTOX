// ref: internal/runtime/executor/helps/vertex_payload_helpers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::borrow::Cow;

use crate::internal::util::get_gjson_bytes_no_copy;

/// Removes OpenAI Responses call IDs that Vertex rejects in Gemini
/// `functionCall` and `functionResponse` payloads.
///
/// The fast path borrows the original body. Allocation and parsing happen only
/// after a borrowed GJSON scan proves that a targeted ID exists.
#[must_use]
pub fn strip_vertex_openai_responses_tool_call_ids<'a>(
    payload: &'a [u8],
    source_format: &str,
) -> Cow<'a, [u8]> {
    if !source_format.trim().eq_ignore_ascii_case("openai-response") {
        return Cow::Borrowed(payload);
    }

    let contents = get_gjson_bytes_no_copy(payload, "contents");
    if contents.kind() != gjson::Kind::Array || !vertex_contents_have_tool_call_ids(&contents) {
        return Cow::Borrowed(payload);
    }

    let Some(mut edits) = vertex_tool_call_id_edits(payload, &contents) else {
        return Cow::Borrowed(payload);
    };
    edits.sort_unstable_by_key(|edit| edit.start);
    if edits.is_empty() || edits.windows(2).any(|pair| pair[0].end > pair[1].start) {
        return Cow::Borrowed(payload);
    }

    let mut output = payload.to_vec();
    for edit in edits.into_iter().rev() {
        output.splice(edit.start..edit.end, edit.replacement);
    }
    Cow::Owned(output)
}

fn vertex_contents_have_tool_call_ids(contents: &gjson::Value<'_>) -> bool {
    let mut found = false;
    contents.each(|_, content| {
        let parts = content.get("parts");
        if parts.kind() != gjson::Kind::Array {
            return true;
        }
        parts.each(|_, part| {
            found =
                part.get("functionCall.id").exists() || part.get("functionResponse.id").exists();
            !found
        });
        !found
    });
    found
}

struct RawEdit {
    start: usize,
    end: usize,
    replacement: Vec<u8>,
}

fn vertex_tool_call_id_edits(payload: &[u8], contents: &gjson::Value<'_>) -> Option<Vec<RawEdit>> {
    let payload_start = payload.as_ptr() as usize;
    let payload_end = payload_start.checked_add(payload.len())?;
    let mut edits = Vec::new();
    let mut valid = true;

    contents.each(|_, content| {
        let parts = content.get("parts");
        if parts.kind() != gjson::Kind::Array {
            return true;
        }
        parts.each(|_, part| {
            for field in ["functionCall", "functionResponse"] {
                let call = part.get(field);
                if call.kind() != gjson::Kind::Object || !call.get("id").exists() {
                    continue;
                }
                let raw = call.json().as_bytes();
                let raw_start = raw.as_ptr() as usize;
                let Some(raw_end) = raw_start.checked_add(raw.len()) else {
                    valid = false;
                    return false;
                };
                if raw_start < payload_start || raw_end > payload_end {
                    valid = false;
                    return false;
                }
                let Some(replacement) = remove_top_level_object_member(raw, "id") else {
                    valid = false;
                    return false;
                };
                edits.push(RawEdit {
                    start: raw_start - payload_start,
                    end: raw_end - payload_start,
                    replacement,
                });
            }
            valid
        });
        valid
    });

    valid.then_some(edits)
}

fn remove_top_level_object_member(raw: &[u8], target: &str) -> Option<Vec<u8>> {
    if raw.first() != Some(&b'{') {
        return None;
    }
    let mut index = skip_ascii_whitespace(raw, 1);
    let mut previous_comma = None;
    while index < raw.len() && raw[index] != b'}' {
        let member_start = index;
        let key_end = scan_json_string(raw, index)?;
        let key = serde_json::from_slice::<String>(&raw[index..key_end]).ok()?;
        index = skip_ascii_whitespace(raw, key_end);
        if raw.get(index) != Some(&b':') {
            return None;
        }
        index = skip_ascii_whitespace(raw, index + 1);
        let value_end = scan_json_value(raw, index)?;
        let delimiter = skip_ascii_whitespace(raw, value_end);
        let comma_after = (raw.get(delimiter) == Some(&b',')).then_some(delimiter);

        if key == target {
            let (remove_start, remove_end) = if let Some(comma) = comma_after {
                (member_start, comma + 1)
            } else if let Some(comma) = previous_comma {
                (comma, value_end)
            } else {
                (member_start, value_end)
            };
            let mut output = Vec::with_capacity(raw.len() - (remove_end - remove_start));
            output.extend_from_slice(&raw[..remove_start]);
            output.extend_from_slice(&raw[remove_end..]);
            return Some(output);
        }

        match comma_after {
            Some(comma) => {
                previous_comma = Some(comma);
                index = skip_ascii_whitespace(raw, comma + 1);
            }
            None if raw.get(delimiter) == Some(&b'}') => return None,
            None => return None,
        }
    }
    None
}

fn skip_ascii_whitespace(raw: &[u8], mut index: usize) -> usize {
    while raw.get(index).is_some_and(u8::is_ascii_whitespace) {
        index += 1;
    }
    index
}

fn scan_json_string(raw: &[u8], start: usize) -> Option<usize> {
    if raw.get(start) != Some(&b'"') {
        return None;
    }
    let mut index = start + 1;
    while index < raw.len() {
        match raw[index] {
            b'"' => return Some(index + 1),
            b'\\' => index = index.checked_add(2)?,
            _ => index += 1,
        }
    }
    None
}

fn scan_json_value(raw: &[u8], start: usize) -> Option<usize> {
    match *raw.get(start)? {
        b'"' => scan_json_string(raw, start),
        b'{' | b'[' => scan_json_container(raw, start),
        _ => {
            let mut index = start;
            while let Some(byte) = raw.get(index) {
                if byte.is_ascii_whitespace() || matches!(byte, b',' | b'}' | b']') {
                    break;
                }
                index += 1;
            }
            (index > start).then_some(index)
        }
    }
}

fn scan_json_container(raw: &[u8], start: usize) -> Option<usize> {
    let mut stack = vec![raw[start]];
    let mut index = start + 1;
    while index < raw.len() {
        match raw[index] {
            b'"' => index = scan_json_string(raw, index)?,
            b'{' | b'[' => {
                stack.push(raw[index]);
                index += 1;
            }
            b'}' | b']' => {
                let open = stack.pop()?;
                if !matches!((open, raw[index]), (b'{', b'}') | (b'[', b']')) {
                    return None;
                }
                index += 1;
                if stack.is_empty() {
                    return Some(index);
                }
            }
            _ => index += 1,
        }
    }
    None
}
