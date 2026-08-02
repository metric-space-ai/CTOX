//! Shared content hashing for Business OS.
//!
//! `hex_sha256` existed four times (store, server, importer, rxdb_peer) and
//! `short_hash` twice. Three of the four were byte-identical; the fourth
//! produced the same lowercase hex through `format!("{:x}")`. Copies that
//! agree today are the ones worth merging — the next edit is where they start
//! to differ, and a hash that differs by caller is silently wrong rather than
//! loudly broken.
//!
//! These are content stamps and identity keys, not password hashing.

use sha2::{Digest, Sha256};

/// Lowercase hex SHA-256 of `bytes`.
pub(super) fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// First 10 characters of the URL-safe base64 SHA-256 of `value`.
///
/// Used for readable ids and cache keys, never where collision resistance
/// carries a security decision — 10 base64 characters are 60 bits.
pub(super) fn short_hash(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, &digest)[..10]
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four former copies must keep agreeing. rxdb_peer's used
    /// `format!("{:x}")` over the digest rather than per-byte formatting; this
    /// pins that those two spellings are the same string.
    #[test]
    fn hex_sha256_matches_the_formatter_spelling_it_replaced() {
        for input in [b"".as_slice(), b"a", b"business-os", &[0u8, 255, 16]] {
            let mut hasher = Sha256::new();
            hasher.update(input);
            let via_formatter = format!("{:x}", hasher.finalize());
            assert_eq!(hex_sha256(input), via_formatter);
            assert_eq!(hex_sha256(input).len(), 64);
        }
    }

    #[test]
    fn short_hash_is_ten_url_safe_characters() {
        let value = short_hash("business-os");
        assert_eq!(value.len(), 10);
        assert!(!value.contains('+') && !value.contains('/') && !value.contains('='));
        assert_ne!(short_hash("a"), short_hash("b"));
    }
}
