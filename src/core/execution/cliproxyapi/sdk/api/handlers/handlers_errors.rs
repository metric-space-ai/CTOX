// ref: sdk/api/handlers/handlers_errors.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::config::SdkConfig;
use crate::internal::interfaces::ErrorMessage;

use super::core::{build_error_response_body, passthrough_headers_enabled};
use super::header_filter::{filter_upstream_headers, is_cpa_reserved_response_header, HeaderMap};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HandlerResponse {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

/// Materializes an error response without owning a socket or HTTP listener.
/// The host may commit this value exactly once to its downstream transport.
#[must_use]
pub fn build_error_response(
    config: Option<&SdkConfig>,
    message: Option<&ErrorMessage>,
    existing_headers: &HeaderMap,
) -> HandlerResponse {
    let status = message
        .and_then(|message| u16::try_from(message.status_code).ok())
        .filter(|status| *status > 0)
        .unwrap_or(500);
    let mut headers = existing_headers.clone();

    if let Some(message) = message {
        if message.direct_response {
            merge_filtered_headers(&mut headers, &message.headers);
            ensure_content_type(&mut headers);
            return HandlerResponse {
                status,
                headers,
                body: message.body.clone(),
            };
        }
        if passthrough_headers_enabled(config) {
            merge_unreserved_headers(&mut headers, &message.addon);
        }
    }

    ensure_content_type(&mut headers);
    let error_text = message
        .and_then(|message| message.error.as_ref())
        .map_or_else(|| status_text(status).to_owned(), ToString::to_string);
    HandlerResponse {
        status,
        headers,
        body: build_error_response_body(status, &error_text),
    }
}

fn merge_filtered_headers(destination: &mut HeaderMap, source: &HeaderMap) {
    if let Some(source) = filter_upstream_headers(source) {
        merge_unreserved_headers(destination, &source);
    }
}

fn merge_unreserved_headers(destination: &mut HeaderMap, source: &HeaderMap) {
    for (key, values) in source {
        if values.is_empty() || is_cpa_reserved_response_header(key) {
            continue;
        }
        remove_header(destination, key);
        destination.insert(key.clone(), values.clone());
    }
}

fn ensure_content_type(headers: &mut HeaderMap) {
    if !headers
        .keys()
        .any(|key| key.eq_ignore_ascii_case("Content-Type"))
    {
        headers.insert(
            "Content-Type".to_owned(),
            vec!["application/json".to_owned()],
        );
    }
}

fn remove_header(headers: &mut HeaderMap, name: &str) {
    let keys = headers
        .keys()
        .filter(|key| key.eq_ignore_ascii_case(name))
        .cloned()
        .collect::<Vec<_>>();
    for key in keys {
        headers.remove(&key);
    }
}

fn status_text(status: u16) -> &'static str {
    match status {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Internal Server Error",
    }
}

#[cfg(test)]
#[path = "handlers_error_response_test.rs"]
mod handlers_error_response_test;
