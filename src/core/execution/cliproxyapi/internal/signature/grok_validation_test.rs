// ref: internal/signature/grok_validation_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};

use super::{
    claude_validation::test_claude_signature, inspect_grok_encrypted_content,
    GrokEncryptedContentError,
};

fn high_entropy(len: usize) -> String {
    general_purpose::STANDARD_NO_PAD.encode((0..len).map(|index| index as u8).collect::<Vec<_>>())
}

fn gpt_signature() -> String {
    let mut payload = vec![0_u8; 73];
    payload[0] = 0x80;
    general_purpose::URL_SAFE_NO_PAD.encode(payload)
}

fn gemini_signature() -> String {
    let value = [0x01, 0x0c, 0x39, 0xd6, 0xc7, 0xaa];
    let mut inner = vec![0x0a, value.len() as u8];
    inner.extend(value);
    let mut outer = vec![0x12, inner.len() as u8];
    outer.extend(inner);
    general_purpose::STANDARD_NO_PAD.encode(outer)
}

fn unpadded_claude_signature() -> String {
    let mut payload = general_purpose::STANDARD
        .decode(test_claude_signature())
        .unwrap();
    // Unknown length-delimited field; the Claude structural validator permits
    // forward-compatible fields, and four bytes make this fixture unpadded.
    payload.extend([0x22, 0x02, 0x01, 0x02]);
    let signature = general_purpose::STANDARD.encode(payload);
    assert!(!signature.ends_with('='));
    signature
}

#[test]
fn accepts_unpadded_high_entropy_native_shape() {
    let raw = high_entropy(256);
    let info = inspect_grok_encrypted_content(&raw).unwrap();
    assert_eq!(info.raw_len, raw.len());
    assert_eq!(info.decoded_len, 256);
}

#[test]
fn rejects_whitespace_padding_prefix_and_bad_alphabet() {
    for raw in [
        format!(" {}", high_entropy(64)),
        format!("{}=", high_entropy(64)),
        format!("grok#{}", high_entropy(64)),
        format!("{}…", high_entropy(64)),
    ] {
        assert_eq!(
            inspect_grok_encrypted_content(&raw).unwrap_err(),
            GrokEncryptedContentError::InvalidShape
        );
    }
}

#[test]
fn rejects_foreign_self_describing_envelopes() {
    for (name, raw) in [
        ("gpt", gpt_signature()),
        ("claude", unpadded_claude_signature()),
        ("gemini", gemini_signature()),
    ] {
        assert_eq!(
            inspect_grok_encrypted_content(&raw).unwrap_err(),
            GrokEncryptedContentError::ForeignEnvelope,
            "{name}: {raw}"
        );
    }
}

#[test]
fn rejects_short_and_low_entropy_payloads() {
    assert_eq!(
        inspect_grok_encrypted_content(&high_entropy(31)).unwrap_err(),
        GrokEncryptedContentError::TooShort
    );
    let low = general_purpose::STANDARD_NO_PAD.encode(vec![0_u8; 128]);
    assert_eq!(
        inspect_grok_encrypted_content(&low).unwrap_err(),
        GrokEncryptedContentError::LowEntropy
    );
}

#[test]
fn rejects_invalid_unpadded_base64_length() {
    assert_eq!(
        inspect_grok_encrypted_content("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA").unwrap_err(),
        GrokEncryptedContentError::InvalidShape
    );
}
