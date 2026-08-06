// ref: sdk/cliproxy/auth/custom_headers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use serde_json::Value;

use super::Auth;

#[must_use]
pub fn extract_custom_headers_from_metadata(
    metadata: &BTreeMap<String, Value>,
) -> BTreeMap<String, String> {
    metadata
        .get("headers")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(name, value)| {
            let name = name.trim();
            let value = value.as_str()?.trim();
            valid_header(name, value).then(|| (name.to_owned(), value.to_owned()))
        })
        .collect()
}

pub fn apply_custom_headers_from_metadata(auth: &mut Auth) {
    for (name, value) in extract_custom_headers_from_metadata(&auth.metadata) {
        auth.attributes.insert(format!("header:{name}"), value);
    }
}

fn valid_header(name: &str, value: &str) -> bool {
    !name.is_empty()
        && !value.is_empty()
        && name.bytes().all(|byte| {
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
        })
        && value
            .bytes()
            .all(|byte| byte == b'\t' || (byte >= 0x20 && byte != 0x7f))
}
