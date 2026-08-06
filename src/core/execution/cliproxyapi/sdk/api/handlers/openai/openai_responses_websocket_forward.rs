// ref: sdk/api/handlers/openai/openai_responses_websocket_forward.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeSet;

#[must_use]
pub fn sorted_string_set(values: &BTreeSet<String>) -> Vec<String> {
    values.iter().cloned().collect()
}

#[must_use]
pub fn websocket_json_payloads_from_chunk(chunk: &[u8]) -> Vec<Vec<u8>> {
    if serde_json::from_slice::<serde_json::Value>(chunk).is_ok() {
        return vec![chunk.to_vec()];
    }
    chunk
        .split(|byte| *byte == b'\n')
        .filter_map(|line| line.strip_prefix(b"data:"))
        .map(trim_ascii)
        .filter(|data| !data.is_empty() && *data != b"[DONE]")
        .filter(|data| serde_json::from_slice::<serde_json::Value>(data).is_ok())
        .map(<[u8]>::to_vec)
        .collect()
}

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}
