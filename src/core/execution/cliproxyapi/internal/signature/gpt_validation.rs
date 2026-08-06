// ref: internal/signature/gpt_validation.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};

pub const MAX_GPT_REASONING_SIGNATURE_LEN: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GptReasoningSignatureInfo {
    pub decoded_len: usize,
    pub ciphertext_len: usize,
}

pub fn is_valid_gpt_reasoning_signature(raw_signature: &str) -> bool {
    inspect_gpt_reasoning_signature(raw_signature).is_ok()
}

pub fn inspect_gpt_reasoning_signature(
    raw_signature: &str,
) -> Result<GptReasoningSignatureInfo, String> {
    let signature = raw_signature.trim();
    if signature.is_empty() {
        return Err("empty GPT reasoning signature".to_owned());
    }
    if signature.len() > MAX_GPT_REASONING_SIGNATURE_LEN {
        return Err(format!(
            "GPT reasoning signature exceeds maximum length ({MAX_GPT_REASONING_SIGNATURE_LEN} bytes)"
        ));
    }
    if !signature.starts_with("gAAAA") {
        return Err("invalid GPT reasoning signature: expected gAAAA prefix".to_owned());
    }
    if let Some((index, character)) = signature.char_indices().find(|(_, character)| {
        !character.is_ascii_alphanumeric() && !matches!(character, '-' | '_' | '=')
    }) {
        return Err(format!(
            "invalid GPT reasoning signature: contains non-base64url character U+{:04X} at byte {index}",
            u32::from(character)
        ));
    }
    let decoded = general_purpose::URL_SAFE_NO_PAD
        .decode(signature)
        .or_else(|_| general_purpose::URL_SAFE.decode(signature))
        .map_err(|_| "invalid GPT reasoning signature: base64url decode failed".to_owned())?;
    if decoded.len() < 73 {
        return Err("invalid GPT reasoning signature: decoded payload too short".to_owned());
    }
    if decoded[0] != 0x80 {
        return Err(format!(
            "invalid GPT reasoning signature: expected version 0x80, got 0x{:02x}",
            decoded[0]
        ));
    }
    let ciphertext_len = decoded.len() - 1 - 8 - 16 - 32;
    if ciphertext_len == 0 || !ciphertext_len.is_multiple_of(16) {
        return Err(format!(
            "invalid GPT reasoning signature: ciphertext length {ciphertext_len} is not a positive AES block multiple"
        ));
    }
    Ok(GptReasoningSignatureInfo {
        decoded_len: decoded.len(),
        ciphertext_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_fernet_shaped_gpt_reasoning_signature() {
        let mut payload = vec![0_u8; 1 + 8 + 16 + 16 + 32];
        payload[0] = 0x80;
        payload[8] = 1;
        for (index, byte) in payload.iter_mut().enumerate().skip(9) {
            *byte = index as u8;
        }
        assert!(is_valid_gpt_reasoning_signature(
            &general_purpose::URL_SAFE.encode(payload)
        ));
    }

    #[test]
    fn rejects_wrong_version_and_ciphertext_length() {
        assert!(!is_valid_gpt_reasoning_signature("gAAAA"));
        let encoded = general_purpose::URL_SAFE.encode(vec![0x80; 73]);
        assert!(!is_valid_gpt_reasoning_signature(&encoded));
    }
}
