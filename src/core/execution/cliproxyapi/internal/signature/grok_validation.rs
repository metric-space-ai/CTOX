// ref: internal/signature/grok_validation.go:12-180 @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine as _};

use super::claude_validation::{
    is_valid_claude_cais_signature, is_valid_claude_thinking_signature_with_options,
    ClaudeSignatureValidationOptions,
};
use super::gemini_validation::{
    is_valid_gemini_thought_signature, GeminiThoughtSignatureValidationOptions,
};
use super::provider_compatibility::has_signature_provider_prefix;

pub const MAX_GROK_ENCRYPTED_CONTENT_LEN: usize = 8 * 1024 * 1024;
pub const MIN_GROK_ENCRYPTED_CONTENT_DECODED_LEN: usize = 32;
pub const MIN_GROK_ENCRYPTED_CONTENT_ENTROPY_RATIO: f64 = 0.85;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrokEncryptedContentInfo {
    pub raw_len: usize,
    pub decoded_len: usize,
}

pub fn inspect_grok_encrypted_content(
    raw: &str,
) -> Result<GrokEncryptedContentInfo, GrokEncryptedContentError> {
    if raw.is_empty() {
        return Err(GrokEncryptedContentError::InvalidShape);
    }
    if raw.len() > MAX_GROK_ENCRYPTED_CONTENT_LEN {
        return Err(GrokEncryptedContentError::TooLarge);
    }
    if raw.trim() != raw
        || raw.contains('=')
        || !raw
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/'))
        || has_signature_provider_prefix(raw)
    {
        return Err(GrokEncryptedContentError::InvalidShape);
    }
    if raw.starts_with("gAAAA")
        || is_valid_claude_thinking_signature_with_options(
            raw,
            ClaudeSignatureValidationOptions {
                strict: true,
                ..ClaudeSignatureValidationOptions::default()
            },
        )
        || is_valid_claude_cais_signature(raw)
        || is_valid_gemini_thought_signature(
            raw,
            GeminiThoughtSignatureValidationOptions {
                require_known_envelope: true,
                ..GeminiThoughtSignatureValidationOptions::default()
            },
        )
    {
        return Err(GrokEncryptedContentError::ForeignEnvelope);
    }
    let decoded = STANDARD_NO_PAD
        .decode(raw)
        .map_err(|_| GrokEncryptedContentError::InvalidShape)?;
    if decoded.len() < MIN_GROK_ENCRYPTED_CONTENT_DECODED_LEN {
        return Err(GrokEncryptedContentError::TooShort);
    }
    if byte_entropy_ratio(&decoded) < MIN_GROK_ENCRYPTED_CONTENT_ENTROPY_RATIO {
        return Err(GrokEncryptedContentError::LowEntropy);
    }
    Ok(GrokEncryptedContentInfo {
        raw_len: raw.len(),
        decoded_len: decoded.len(),
    })
}

pub fn is_valid_grok_encrypted_content(raw: &str) -> bool {
    inspect_grok_encrypted_content(raw).is_ok()
}

fn byte_entropy_ratio(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }
    let mut counts = [0_usize; 256];
    for byte in bytes {
        counts[*byte as usize] += 1;
    }
    let n = bytes.len() as f64;
    let entropy = counts
        .into_iter()
        .filter(|count| *count != 0)
        .map(|count| {
            let probability = count as f64 / n;
            -probability * probability.log2()
        })
        .sum::<f64>();
    entropy / (bytes.len().min(256) as f64).log2()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrokEncryptedContentError {
    InvalidShape,
    TooLarge,
    TooShort,
    LowEntropy,
    ForeignEnvelope,
}

impl std::fmt::Display for GrokEncryptedContentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("invalid Grok encrypted_content")
    }
}

impl std::error::Error for GrokEncryptedContentError {}
