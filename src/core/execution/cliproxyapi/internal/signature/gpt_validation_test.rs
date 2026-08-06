// ref: internal/signature/gpt_validation_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};

use super::{detect_signature_provider, inspect_gpt_reasoning_signature, SignatureProvider};

fn signature(ciphertext_len: usize) -> String {
    let mut payload = vec![0_u8; 1 + 8 + 16 + ciphertext_len + 32];
    payload[0] = 0x80;
    general_purpose::URL_SAFE_NO_PAD.encode(payload)
}

#[test]
fn inspects_and_classifies_fernet_shape() {
    let raw = signature(16);
    let info = inspect_gpt_reasoning_signature(&raw).unwrap();
    assert_eq!(info.decoded_len, 73);
    assert_eq!(info.ciphertext_len, 16);
    assert_eq!(detect_signature_provider(&raw), SignatureProvider::Gpt);
}

#[test]
fn rejects_unicode_and_invalid_ciphertext_length() {
    let unicode = format!("{}…", signature(16));
    let error = inspect_gpt_reasoning_signature(&unicode).unwrap_err();
    assert!(error.contains("U+2026"));
    assert!(error.contains("byte"));
    assert!(inspect_gpt_reasoning_signature(&signature(15)).is_err());
}

#[test]
fn rejects_empty_bad_prefix_and_too_short() {
    assert!(inspect_gpt_reasoning_signature("").is_err());
    assert!(inspect_gpt_reasoning_signature("EAAAA").is_err());
    assert!(inspect_gpt_reasoning_signature("gAAAA").is_err());
}
