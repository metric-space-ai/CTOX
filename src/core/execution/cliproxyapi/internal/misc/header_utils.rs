// ref: internal/misc/header_utils.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use crate::internal::util::canonical_header_name;

/// Rust representation of Go's multi-value `http.Header`.
pub type Headers = BTreeMap<String, Vec<String>>;

const SCRUBBED_HEADERS: &[&str] = &[
    // Proxy tracing headers.
    "X-Forwarded-For",
    "X-Forwarded-Host",
    "X-Forwarded-Proto",
    "X-Forwarded-Port",
    "X-Real-IP",
    "Forwarded",
    "Via",
    // Client identity headers.
    "X-Title",
    "X-Stainless-Lang",
    "X-Stainless-Package-Version",
    "X-Stainless-Os",
    "X-Stainless-Arch",
    "X-Stainless-Runtime",
    "X-Stainless-Runtime-Version",
    "Http-Referer",
    "Referer",
    // Browser / Chromium fingerprint headers.
    "Sec-Ch-Ua",
    "Sec-Ch-Ua-Mobile",
    "Sec-Ch-Ua-Platform",
    "Sec-Fetch-Mode",
    "Sec-Fetch-Site",
    "Sec-Fetch-Dest",
    "Priority",
    // Encoding negotiation.
    "Accept-Encoding",
];

/// Removes headers that reveal proxy infrastructure, client identity, or
/// browser fingerprints from an outgoing request.
///
/// Go's `http.Header.Del` canonicalizes names before deleting them. Rust maps
/// do not, so this port deliberately removes every ASCII-case variant. The
/// caller supplies only the request's header map because Rust transports do
/// not share Go's concrete `*http.Request` type.
pub fn scrub_proxy_and_fingerprint_headers(headers: &mut Headers) {
    headers.retain(|name, _| {
        !SCRUBBED_HEADERS
            .iter()
            .any(|scrubbed| name.eq_ignore_ascii_case(scrubbed))
    });
}

/// Ensures that `key` exists in `target`, using the first non-blank source
/// value, an existing non-blank target value, or the non-blank default in that
/// order.
pub fn ensure_header(
    target: &mut Headers,
    source: Option<&Headers>,
    key: &str,
    default_value: &str,
) {
    if let Some(value) = source.and_then(|headers| header_value(headers, key)) {
        let value = value.trim();
        if !value.is_empty() {
            set_header(target, key, value);
            return;
        }
    }

    if header_value(target, key).is_some_and(|value| !value.trim().is_empty()) {
        return;
    }

    let default_value = default_value.trim();
    if !default_value.is_empty() {
        set_header(target, key, default_value);
    }
}

fn header_value<'a>(headers: &'a Headers, key: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(key))
        .and_then(|(_, values)| values.first())
        .map(String::as_str)
}

fn set_header(headers: &mut Headers, key: &str, value: &str) {
    headers.retain(|name, _| !name.eq_ignore_ascii_case(key));
    headers.insert(canonical_header_name(key), vec![value.to_owned()]);
}
