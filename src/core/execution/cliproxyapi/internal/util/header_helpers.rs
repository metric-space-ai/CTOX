// ref: internal/util/header_helpers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

/// Rust representation of the two places in which Go's `http.Request` stores
/// request headers. `Host` must be mirrored into [`Self::host`] because Go's
/// HTTP writer does not read it from `Request.Header`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeaderRequest {
    pub host: String,
    pub headers: BTreeMap<String, Vec<String>>,
}

/// Applies user-defined `header:<name>` attributes to a request.
///
/// Like the pinned Go implementation, custom values replace a header's full
/// value list and take precedence over existing defaults.
pub fn apply_custom_headers_from_attrs(
    request: &mut HeaderRequest,
    attrs: &BTreeMap<String, String>,
) {
    if let Some(headers) = extract_custom_headers(attrs) {
        apply_custom_headers(request, &headers);
    }
}

#[must_use]
pub fn extract_custom_headers(
    attrs: &BTreeMap<String, String>,
) -> Option<BTreeMap<String, String>> {
    if attrs.is_empty() {
        return None;
    }

    let headers = attrs
        .iter()
        .filter_map(|(key, value)| {
            let name = key.strip_prefix("header:")?.trim();
            let value = value.trim();
            (!name.is_empty() && !value.is_empty()).then(|| (name.to_owned(), value.to_owned()))
        })
        .collect::<BTreeMap<_, _>>();

    (!headers.is_empty()).then_some(headers)
}

pub fn apply_custom_headers(request: &mut HeaderRequest, headers: &BTreeMap<String, String>) {
    for (name, value) in headers {
        if name.is_empty() || value.is_empty() {
            continue;
        }

        let canonical_name = canonical_header_name(name);
        if canonical_name == "Host" {
            request.host.clone_from(value);
        }
        request
            .headers
            .insert(canonical_name, vec![value.to_owned()]);
    }
}

/// Mirrors `http.CanonicalHeaderKey` for UTF-8 Rust strings. Valid HTTP token
/// names are ASCII, so a non-token or non-ASCII name is returned unchanged as
/// it is by Go's `textproto.CanonicalMIMEHeaderKey`.
#[must_use]
pub fn canonical_header_name(name: &str) -> String {
    if !name.bytes().all(is_header_token_byte) {
        return name.to_owned();
    }

    let mut upper = true;
    name.bytes()
        .map(|byte| {
            let canonical = if upper {
                byte.to_ascii_uppercase()
            } else {
                byte.to_ascii_lowercase()
            };
            upper = byte == b'-';
            canonical as char
        })
        .collect()
}

fn is_header_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}
