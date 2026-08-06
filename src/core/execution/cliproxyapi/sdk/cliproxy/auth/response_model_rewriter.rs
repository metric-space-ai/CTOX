// ref: sdk/cliproxy/auth/response_model_rewriter.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

const MODEL_FIELD_PATHS: [&[&str]; 5] = [
    &["model"],
    &["modelVersion"],
    &["response", "model"],
    &["response", "modelVersion"],
    &["message", "model"],
];
const MAX_PENDING_BUFFER_SIZE: usize = 1 << 20;

#[must_use]
pub fn rewrite_sse_payload_lines(payload: &[u8], target_model: &str) -> Vec<u8> {
    if target_model.is_empty() || payload.is_empty() {
        return payload.to_vec();
    }
    let mut output = payload
        .split(|byte| *byte == b'\n')
        .map(|line| {
            let Some((prefix, json)) = extract_sse_data_line(line) else {
                return line.to_vec();
            };
            if json.first() != Some(&b'{') || serde_json::from_slice::<Value>(json).is_err() {
                return line.to_vec();
            }
            let mut rewritten = prefix.to_vec();
            rewritten.extend(rewrite_model_in_response(json, target_model));
            rewritten
        })
        .collect::<Vec<_>>()
        .join(&b'\n');
    if payload.last() == Some(&b'\n') && output.last() != Some(&b'\n') {
        output.push(b'\n');
    }
    output
}

#[must_use]
pub fn rewrite_model_in_response(data: &[u8], target_model: &str) -> Vec<u8> {
    if target_model.is_empty() || data.is_empty() {
        return data.to_vec();
    }
    let Ok(mut document) = serde_json::from_slice::<Value>(data) else {
        return data.to_vec();
    };
    let mut changed = false;
    for path in MODEL_FIELD_PATHS {
        if let Some(value) = value_at_path_mut(&mut document, path) {
            *value = Value::String(target_model.to_owned());
            changed = true;
        }
    }
    if changed {
        serde_json::to_vec(&document).unwrap_or_else(|_| data.to_vec())
    } else {
        data.to_vec()
    }
}

fn value_at_path_mut<'a>(mut value: &'a mut Value, path: &[&str]) -> Option<&'a mut Value> {
    for segment in path {
        value = value.as_object_mut()?.get_mut(*segment)?;
    }
    Some(value)
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StreamRewriteOptions {
    pub rewrite_model: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StreamRewriter {
    options: StreamRewriteOptions,
    pending: Vec<u8>,
}

impl StreamRewriter {
    #[must_use]
    pub fn new(options: StreamRewriteOptions) -> Self {
        Self {
            options,
            pending: Vec::new(),
        }
    }

    #[must_use]
    pub(crate) fn rewrite_model(&self) -> &str {
        &self.options.rewrite_model
    }

    pub fn rewrite_chunk(&mut self, chunk: &[u8]) -> Vec<u8> {
        if self.options.rewrite_model.is_empty() {
            return chunk.to_vec();
        }
        let mut chunk = if self.pending.is_empty() {
            chunk.to_vec()
        } else {
            let mut combined = std::mem::take(&mut self.pending);
            if combined.last() != Some(&b'\n') {
                combined.push(b'\n');
            }
            combined.extend_from_slice(chunk);
            combined
        };
        chunk = normalize_glued_sse_events(&chunk);
        if chunk.len() > MAX_PENDING_BUFFER_SIZE {
            return chunk;
        }

        let trimmed = trim_ascii(&chunk);
        if trimmed.first() == Some(&b'{') && serde_json::from_slice::<Value>(trimmed).is_ok() {
            return rewrite_model_in_response(trimmed, &self.options.rewrite_model);
        }

        let process = if let Some(last) = find_last_double_newline(&chunk) {
            let boundary = last + 2;
            let trailing = &chunk[boundary..];
            if !trailing.is_empty() && trailing != b"\n" {
                self.pending.extend_from_slice(trailing);
                chunk[..boundary].to_vec()
            } else {
                chunk
            }
        } else if extract_last_data_payload(&chunk)
            .is_some_and(|json| serde_json::from_slice::<Value>(json).is_ok())
        {
            chunk
        } else if trimmed.is_empty() {
            return chunk;
        } else if !chunk.is_empty() {
            self.pending = chunk;
            return Vec::new();
        } else {
            return chunk;
        };

        self.rewrite_complete_lines(&process)
    }

    fn rewrite_complete_lines(&mut self, chunk: &[u8]) -> Vec<u8> {
        let mut result = Vec::<Vec<u8>>::new();
        let mut pending_event: Option<Vec<u8>> = None;
        let mut skip_blanks = false;
        for line in chunk.split(|byte| *byte == b'\n') {
            if line.is_empty() && skip_blanks {
                continue;
            }
            if !line.is_empty() {
                skip_blanks = false;
            }
            if line.starts_with(b"event:") {
                pending_event = Some(line.to_vec());
                continue;
            }
            if let Some((prefix, json)) = extract_sse_data_line(line) {
                if json.first() == Some(&b'{') {
                    if serde_json::from_slice::<Value>(json).is_err() {
                        if let Some(event) = pending_event.take() {
                            self.pending.extend(event);
                            self.pending.push(b'\n');
                        }
                        self.pending.extend_from_slice(line);
                        continue;
                    }
                    if let Some(event) = pending_event.take() {
                        result.push(event);
                    }
                    let mut rewritten = prefix.to_vec();
                    rewritten.extend(rewrite_model_in_response(json, &self.options.rewrite_model));
                    result.push(rewritten);
                    continue;
                }
            }
            if let Some(event) = pending_event.take() {
                result.push(event);
            }
            result.push(line.to_vec());
        }
        if let Some(event) = pending_event {
            result.push(event);
        }
        let joined = result.join(&b'\n');
        if joined.is_empty() && !chunk.is_empty() {
            rewrite_sse_payload_lines(chunk, &self.options.rewrite_model)
        } else {
            joined
        }
    }

    pub fn finish(&mut self) -> Vec<u8> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let mut buffer = std::mem::take(&mut self.pending);
        buffer.extend_from_slice(b"\n\n");
        let buffer = normalize_glued_sse_events(&buffer);
        let mut output = self.rewrite_chunk(&buffer);
        if !self.pending.is_empty() {
            let tail = rewrite_sse_payload_lines(&self.pending, &self.options.rewrite_model);
            self.pending.clear();
            output.extend(tail);
        }
        output
    }
}

fn extract_last_data_payload(chunk: &[u8]) -> Option<&[u8]> {
    chunk.split(|byte| *byte == b'\n').rev().find_map(|line| {
        extract_sse_data_line(line)
            .map(|(_, json)| json)
            .filter(|json| !json.is_empty())
    })
}

fn extract_sse_data_line(line: &[u8]) -> Option<(&[u8], &[u8])> {
    line.strip_prefix(b"data: ")
        .map(|json| (&b"data: "[..], json))
        .or_else(|| {
            line.strip_prefix(b"data:")
                .map(|json| (&b"data:"[..], json))
        })
}

#[must_use]
pub fn normalize_glued_sse_events(chunk: &[u8]) -> Vec<u8> {
    let chunk = safe_replace_glued(chunk, b"}event:", b"}\n\nevent:");
    let chunk = safe_replace_glued(&chunk, b"}\r\nevent:", b"}\r\n\r\nevent:");
    let chunk = safe_replace_glued(&chunk, b"}data:", b"}\ndata:");
    safe_replace_glued(&chunk, b"}\r\ndata:", b"}\r\ndata:")
}

fn safe_replace_glued(chunk: &[u8], old: &[u8], new: &[u8]) -> Vec<u8> {
    if old.is_empty() || !contains(chunk, old) {
        return chunk.to_vec();
    }
    let mut result = Vec::with_capacity(chunk.len());
    let mut offset = 0;
    while let Some(relative) = find_subslice(&chunk[offset..], old) {
        let index = offset + relative;
        let line_start = chunk[..index]
            .iter()
            .rposition(|byte| *byte == b'\n')
            .map_or(0, |pos| pos + 1);
        let part = &chunk[line_start..=index];
        let valid = extract_sse_data_line(part)
            .is_some_and(|(_, json)| serde_json::from_slice::<Value>(json).is_ok());
        if valid {
            result.extend_from_slice(&chunk[offset..index]);
            result.extend_from_slice(new);
        } else {
            result.extend_from_slice(&chunk[offset..index + old.len()]);
        }
        offset = index + old.len();
    }
    result.extend_from_slice(&chunk[offset..]);
    result
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map_or(start, |index| index + 1);
    &bytes[start..end]
}

fn find_last_double_newline(bytes: &[u8]) -> Option<usize> {
    bytes.windows(2).rposition(|window| window == b"\n\n")
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    find_subslice(haystack, needle).is_some()
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_spaced_unspaced_nested_and_raw_models() {
        for chunk in [
            b"data:{\"message\":{\"model\":\"kimi-k2.5\"}}\n\n".as_slice(),
            b"data: {\"response\":{\"model\":\"gpt-5.4\"}}\n\n".as_slice(),
            b"{\"response\":{\"modelVersion\":\"gemini-3-flash\"}}".as_slice(),
        ] {
            let mut rewriter = StreamRewriter::new(StreamRewriteOptions {
                rewrite_model: "public-model".to_owned(),
            });
            let rewritten = rewriter.rewrite_chunk(chunk);
            assert!(String::from_utf8_lossy(&rewritten).contains("public-model"));
        }
    }

    #[test]
    fn joins_partial_event_and_splits_only_valid_glue() {
        let mut rewriter = StreamRewriter::new(StreamRewriteOptions {
            rewrite_model: "gpt-fast".to_owned(),
        });
        assert!(rewriter
            .rewrite_chunk(b"event: response.created\n")
            .is_empty());
        let output = rewriter.rewrite_chunk(
            b"data: {\"type\":\"response.created\",\"response\":{\"model\":\"gpt\"}}\n\n",
        );
        let output = String::from_utf8(output).unwrap();
        assert_eq!(output.matches("event: response.created").count(), 1);
        assert!(output.contains("gpt-fast"));

        let glued =
            normalize_glued_sse_events(b"data: {\"type\":\"a\"}event: b\ndata: {\"type\":\"b\"}");
        assert!(glued.windows(9).any(|window| window == b"}\n\nevent:"));
        let inside = b"data: {\"text\":\"literal }event: inside\"}";
        assert_eq!(normalize_glued_sse_events(inside), inside);
    }
}
