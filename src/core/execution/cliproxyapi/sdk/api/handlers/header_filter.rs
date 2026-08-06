// ref: sdk/api/handlers/header_filter.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};

pub type HeaderMap = BTreeMap<String, Vec<String>>;

const GATEWAY_HEADER_PREFIXES: &[&str] = &[
    "x-litellm-",
    "helicone-",
    "x-portkey-",
    "cf-aig-",
    "x-kong-",
    "x-bt-",
];

const HOP_BY_HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "set-cookie",
    "content-length",
    "content-encoding",
];

const CPA_RESERVED_RESPONSE_HEADERS: &[&str] = &[
    "access-control-allow-credentials",
    "access-control-allow-headers",
    "access-control-allow-methods",
    "access-control-allow-origin",
    "access-control-expose-headers",
    "access-control-max-age",
    "x-cpa-trace-id",
];

pub fn is_cpa_reserved_response_header(name: &str) -> bool {
    contains_ascii_case_insensitive(CPA_RESERVED_RESPONSE_HEADERS, name.trim())
}

// ref: sdk/api/handlers/header_filter.go:53-90
pub fn filter_upstream_headers(source: &HeaderMap) -> Option<HeaderMap> {
    let connection_scoped = connection_scoped_headers(source);
    let mut destination = HeaderMap::new();
    for (key, values) in source {
        let normalized = key.trim().to_ascii_lowercase();
        if contains_ascii_case_insensitive(HOP_BY_HOP_HEADERS, &normalized)
            || is_cpa_reserved_response_header(&normalized)
            || connection_scoped.contains(&normalized)
            || GATEWAY_HEADER_PREFIXES
                .iter()
                .any(|prefix| normalized.starts_with(prefix))
        {
            continue;
        }
        destination.insert(key.clone(), values.clone());
    }
    (!destination.is_empty()).then_some(destination)
}

// ref: sdk/api/handlers/header_filter.go:108-121
pub fn write_upstream_headers(destination: &mut HeaderMap, source: &HeaderMap) {
    for (key, values) in source {
        if destination
            .keys()
            .any(|existing| existing.eq_ignore_ascii_case(key))
        {
            continue;
        }
        destination.insert(key.clone(), values.clone());
    }
}

fn connection_scoped_headers(source: &HeaderMap) -> BTreeSet<String> {
    source
        .iter()
        .filter(|(key, _)| key.eq_ignore_ascii_case("connection"))
        .flat_map(|(_, values)| values)
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

fn contains_ascii_case_insensitive(haystack: &[&str], needle: &str) -> bool {
    haystack
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(needle))
}

#[cfg(test)]
#[path = "header_filter_test.rs"]
mod header_filter_test;
