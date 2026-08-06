// ref: sdk/api/handlers/handlers_interceptors.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::pluginapi::{Headers, RequestInterceptResponse};

use super::header_filter::{filter_upstream_headers, HeaderMap};
use super::HandlerResponse;

pub const MAX_STREAM_INTERCEPTOR_HISTORY_CHUNKS: usize = 64;
pub const MAX_STREAM_INTERCEPTOR_HISTORY_BYTES: usize = 1 << 20;

#[must_use]
pub fn append_stream_interceptor_history(history: &[Vec<u8>], chunk: &[u8]) -> Vec<Vec<u8>> {
    let mut result = history.to_vec();
    if !chunk.is_empty() {
        result.push(chunk.to_vec());
    }
    while result.len() > MAX_STREAM_INTERCEPTOR_HISTORY_CHUNKS
        || result.iter().map(Vec::len).sum::<usize>() > MAX_STREAM_INTERCEPTOR_HISTORY_BYTES
    {
        if result.is_empty() {
            break;
        }
        result.remove(0);
    }
    result
}

#[must_use]
pub fn apply_interceptor_headers(
    current: &Headers,
    updates: &Headers,
    clear: &[String],
) -> Headers {
    let mut result = current.clone();
    for name in clear {
        remove_header(&mut result, name);
    }
    for (name, values) in updates {
        remove_header(&mut result, name);
        if !values.is_empty() {
            result.insert(name.clone(), values.clone());
        }
    }
    result
}

#[must_use]
pub fn termination_response(intercepted: &RequestInterceptResponse) -> Option<HandlerResponse> {
    if !intercepted.terminate {
        return None;
    }
    let status = if intercepted.status_code == 0 {
        403
    } else {
        intercepted.status_code
    };
    Some(HandlerResponse {
        status,
        headers: filter_upstream_headers(&intercepted.response_headers).unwrap_or_default(),
        body: intercepted.response_body.clone(),
    })
}

fn remove_header(headers: &mut HeaderMap, name: &str) {
    let keys = headers
        .keys()
        .filter(|key| key.eq_ignore_ascii_case(name.trim()))
        .cloned()
        .collect::<Vec<_>>();
    for key in keys {
        headers.remove(&key);
    }
}

#[cfg(test)]
#[path = "handlers_interceptors_test.rs"]
mod handlers_interceptors_test;
