// ref: internal/runtime/executor/helps/claude_mcp_alias.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use sha2::{Digest, Sha256};

const HMAC_BLOCK_BYTES: usize = 64;
const BASE32_ALPHABET: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";

/// Reports whether `name` follows Claude Code's MCP tool convention and uses
/// only characters accepted by Anthropic tool names.
pub fn is_claude_mcp_tool_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 64 || !name.starts_with("mcp__") {
        return false;
    }
    let rest = &name[5..];
    let Some(separator) = rest.find("__") else {
        return false;
    };
    if separator == 0 || separator + 2 >= rest.len() {
        return false;
    }
    name.bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

/// Derives a deterministic Claude Code-style MCP tool name scoped to one
/// caller secret. A higher attempt changes the stable tool ID after collision.
pub fn claude_mcp_tool_alias(secret: &str, original: &str, attempt: u32) -> String {
    let server_digest = claude_mcp_alias_digest(secret, "server", "", 0);
    let tool_digest = claude_mcp_alias_digest(secret, "tool", original, attempt);
    let server = base32_no_padding(&server_digest)[..12].to_owned();
    let tool_id = base32_no_padding(&tool_digest)[..12].to_owned();
    let semantic = claude_mcp_tool_semantic_suffix(original, 32);
    format!("mcp__{server}__{tool_id}_{semantic}")
}

fn claude_mcp_tool_semantic_suffix(original: &str, max_length: usize) -> String {
    let mut semantic = String::with_capacity(original.len().min(max_length));
    let mut pending_separator = false;
    for character in original.chars() {
        let valid = character.is_ascii_alphanumeric() || character == '_' || character == '-';
        if !valid {
            pending_separator = !semantic.is_empty();
            continue;
        }
        if pending_separator && semantic.len() + 1 < max_length {
            semantic.push('_');
        }
        pending_separator = false;
        if semantic.len() >= max_length {
            break;
        }
        semantic.push(character);
    }
    let result = semantic.trim_matches(['_', '-']);
    if result.is_empty() {
        "tool".to_owned()
    } else {
        result.to_owned()
    }
}

fn claude_mcp_alias_digest(secret: &str, purpose: &str, original: &str, attempt: u32) -> [u8; 32] {
    let mut message = Vec::with_capacity(32 + purpose.len() + original.len());
    message.extend_from_slice(b"cpa-claude-mcp-alias-v2\0");
    message.extend_from_slice(purpose.as_bytes());
    message.push(0);
    message.extend_from_slice(original.as_bytes());
    message.extend_from_slice(&attempt.to_be_bytes());
    hmac_sha256(secret.as_bytes(), &message)
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut key_block = [0_u8; HMAC_BLOCK_BYTES];
    if key.len() > HMAC_BLOCK_BYTES {
        key_block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }
    let mut inner_pad = [0x36_u8; HMAC_BLOCK_BYTES];
    let mut outer_pad = [0x5c_u8; HMAC_BLOCK_BYTES];
    for ((inner, outer), key_byte) in inner_pad
        .iter_mut()
        .zip(outer_pad.iter_mut())
        .zip(key_block)
    {
        *inner ^= key_byte;
        *outer ^= key_byte;
    }
    let inner = Sha256::new()
        .chain_update(inner_pad)
        .chain_update(message)
        .finalize();
    Sha256::new()
        .chain_update(outer_pad)
        .chain_update(inner)
        .finalize()
        .into()
}

fn base32_no_padding(bytes: &[u8]) -> String {
    let mut result = String::with_capacity((bytes.len() * 8).div_ceil(5));
    let mut accumulator = 0_u16;
    let mut bits = 0_u8;
    for byte in bytes {
        accumulator = (accumulator << 8) | u16::from(*byte);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            result.push(BASE32_ALPHABET[((accumulator >> bits) & 0x1f) as usize] as char);
        }
        accumulator &= (1_u16 << bits) - 1;
    }
    if bits > 0 {
        result.push(BASE32_ALPHABET[((accumulator << (5 - bits)) & 0x1f) as usize] as char);
    }
    result
}
