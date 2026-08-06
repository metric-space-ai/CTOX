// ref: internal/translator/common/bytes.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

pub fn gemini_token_count_json(count: i64) -> Vec<u8> {
    format!(
        "{{\"totalTokens\":{count},\"promptTokensDetails\":[{{\"modality\":\"TEXT\",\"tokenCount\":{count}}}]}}"
    ).into_bytes()
}

pub fn claude_input_tokens_json(count: i64) -> Vec<u8> {
    format!("{{\"input_tokens\":{count}}}").into_bytes()
}

/// Preserves Go's nil-versus-empty distinction: non-positive capacities return
/// `None`, while positive capacities return an allocated, empty item vector.
pub fn new_raw_array_items(capacity: i64) -> Option<Vec<Vec<u8>>> {
    let capacity = usize::try_from(capacity)
        .ok()
        .filter(|capacity| *capacity > 0)?;
    Some(Vec::with_capacity(capacity))
}

pub fn join_raw_array(items: &[Vec<u8>]) -> Vec<u8> {
    if items.is_empty() {
        return b"[]".to_vec();
    }
    let size = items.iter().map(Vec::len).sum::<usize>() + items.len() + 1;
    let mut output = Vec::with_capacity(size);
    output.push(b'[');
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            output.push(b',');
        }
        output.extend_from_slice(item);
    }
    output.push(b']');
    output
}

/// Replaces the JSON value at a gjson-style dotted object path with the raw
/// array. Existing values are spliced byte-for-byte so surrounding field order
/// and raw tool arguments are not re-encoded.
pub fn set_raw_array_items(data: &[u8], path: &str, items: &[Vec<u8>]) -> Vec<u8> {
    if items.is_empty() {
        return data.to_vec();
    }
    let Ok(document) = std::str::from_utf8(data) else {
        return data.to_vec();
    };
    let replacement = join_raw_array(items);
    let existing = gjson::get(document, path);
    if existing.exists() {
        let raw = existing.json();
        let document_start = document.as_ptr() as usize;
        let raw_start = raw.as_ptr() as usize;
        if raw_start >= document_start {
            let start = raw_start - document_start;
            if start <= data.len() && raw.len() <= data.len() - start {
                let mut output = Vec::with_capacity(data.len() - raw.len() + replacement.len());
                output.extend_from_slice(&data[..start]);
                output.extend_from_slice(&replacement);
                output.extend_from_slice(&data[start + raw.len()..]);
                return output;
            }
        }
    }

    // `sjson.SetRawBytes` also creates a missing dotted object path. This
    // fallback is only used for that case; ordinary hot paths above retain the
    // original bytes exactly.
    let Ok(mut root) = serde_json::from_slice::<Value>(data) else {
        return data.to_vec();
    };
    let Ok(array) = serde_json::from_slice::<Value>(&replacement) else {
        return data.to_vec();
    };
    let mut segments = path.split('.').peekable();
    let Some(first) = segments.next() else {
        return data.to_vec();
    };
    let mut current = &mut root;
    let mut segment = first;
    loop {
        let Some(object) = current.as_object_mut() else {
            return data.to_vec();
        };
        if segments.peek().is_none() {
            object.insert(segment.to_owned(), array);
            break;
        }
        current = object
            .entry(segment.to_owned())
            .or_insert_with(|| Value::Object(Default::default()));
        segment = segments.next().expect("peeked path segment");
    }
    serde_json::to_vec(&root).unwrap_or_else(|_| data.to_vec())
}

pub fn sse_event_data(event: &str, payload: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(event.len() + payload.len() + 14);
    append_sse_event(&mut output, event, payload, 0);
    output
}

pub fn append_sse_event(
    output: &mut Vec<u8>,
    event: &str,
    payload: &[u8],
    trailing_newlines: usize,
) {
    output.extend_from_slice(b"event: ");
    output.extend_from_slice(event.as_bytes());
    output.push(b'\n');
    output.extend_from_slice(b"data: ");
    output.extend_from_slice(payload);
    output.extend(std::iter::repeat_n(b'\n', trailing_newlines));
}
