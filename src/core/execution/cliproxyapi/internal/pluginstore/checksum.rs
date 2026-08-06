// ref: internal/pluginstore/checksum.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use super::github::store_error;

pub fn parse_checksums(data: &[u8]) -> crate::sdk::pluginstore::Result<HashMap<String, String>> {
    let text = std::str::from_utf8(data).map_err(|_| store_error("checksums file is not UTF-8"))?;
    let mut checksums = HashMap::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let digest = fields.next().unwrap_or_default().to_ascii_lowercase();
        let name = fields.next().unwrap_or_default().trim_start_matches('*');
        if digest.len() != 64
            || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            || name.is_empty()
        {
            return Err(store_error(format!("invalid checksums line {}", index + 1)));
        }
        checksums.insert(name.to_owned(), digest);
    }
    Ok(checksums)
}

pub fn verify_checksum(
    name: &str,
    data: &[u8],
    checksums: &HashMap<String, String>,
) -> crate::sdk::pluginstore::Result<()> {
    let expected = checksums
        .get(name.trim())
        .ok_or_else(|| store_error(format!("checksum for {} not found", name.trim())))?;
    let actual = format!("{:x}", Sha256::digest(data));
    if subtle::ConstantTimeEq::ct_eq(actual.as_bytes(), expected.as_bytes()).into() {
        Ok(())
    } else {
        Err(store_error(format!(
            "checksum mismatch for {}",
            name.trim()
        )))
    }
}
