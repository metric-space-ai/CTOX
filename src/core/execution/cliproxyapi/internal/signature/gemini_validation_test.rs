// ref: internal/signature/gemini_validation_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};

use super::{
    inspect_gemini_thought_signature, validate_gemini_function_call_pairing,
    validate_gemini_thought_signatures, GeminiThoughtSignatureEnvelope,
    GeminiThoughtSignatureValidationOptions, GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR,
};

fn field2_signature(value: &[u8]) -> String {
    let mut inner = vec![0x0a, value.len() as u8];
    inner.extend_from_slice(value);
    let mut outer = vec![0x12, inner.len() as u8];
    outer.extend_from_slice(&inner);
    general_purpose::STANDARD.encode(outer)
}

fn known_options() -> GeminiThoughtSignatureValidationOptions {
    GeminiThoughtSignatureValidationOptions {
        require_known_envelope: true,
        ..GeminiThoughtSignatureValidationOptions::default()
    }
}

#[test]
fn opaque_base64_is_valid_only_without_known_envelope_gate() {
    let raw = general_purpose::STANDARD.encode(b"opaque provider state");
    let info =
        inspect_gemini_thought_signature(&raw, GeminiThoughtSignatureValidationOptions::default())
            .unwrap();
    assert_eq!(info.envelope, GeminiThoughtSignatureEnvelope::Unknown);
    assert!(!info.known_envelope);
    assert!(inspect_gemini_thought_signature(&raw, known_options()).is_err());
}

#[test]
fn recognizes_field2_tink_and_wrapped_uuid_envelopes() {
    for value in [
        &[0x01, 0x0c, 0x39, 0xd6, 0xc7, 0xaa][..],
        b"e24830a7-5cd6-42fe-998b-ee539e72b9c3".as_slice(),
    ] {
        let info =
            inspect_gemini_thought_signature(&field2_signature(value), known_options()).unwrap();
        assert_eq!(
            info.envelope,
            GeminiThoughtSignatureEnvelope::ProtobufField2
        );
        assert_eq!(info.record_count, 1);
        assert_eq!(info.opaque_payload_len, value.len());
        assert!(info.has_observed_marker);
    }
}

#[test]
fn bare_uuid_is_classified_but_not_replay_safe() {
    let raw = general_purpose::STANDARD.encode(b"e24830a7-5cd6-42fe-998b-ee539e72b9c3");
    let info =
        inspect_gemini_thought_signature(&raw, GeminiThoughtSignatureValidationOptions::default())
            .unwrap();
    assert_eq!(info.envelope, GeminiThoughtSignatureEnvelope::AsciiUuid);
    assert!(inspect_gemini_thought_signature(&raw, known_options()).is_err());
}

#[test]
fn bypass_requires_explicit_option() {
    assert!(inspect_gemini_thought_signature(
        GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR,
        GeminiThoughtSignatureValidationOptions::default()
    )
    .is_err());
    let info = inspect_gemini_thought_signature(
        GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR,
        GeminiThoughtSignatureValidationOptions {
            allow_bypass_sentinel: true,
            ..GeminiThoughtSignatureValidationOptions::default()
        },
    )
    .unwrap();
    assert!(info.is_bypass_sentinel);
}

#[test]
fn thought_validation_enforces_first_call_and_canonical_field() {
    let missing = br#"{"contents":[{"role":"model","parts":[{"functionCall":{"name":"a"}}]}]}"#;
    assert!(validate_gemini_thought_signatures(missing, known_options())
        .unwrap_err()
        .contains("first functionCall"));

    let signature = field2_signature(&[0x01, 1, 2, 3, 4]);
    let nested = format!(
        r#"{{"contents":[{{"role":"model","parts":[{{"functionCall":{{"name":"a","thoughtSignature":"{signature}"}}}}]}}]}}"#
    );
    assert!(
        validate_gemini_thought_signatures(nested.as_bytes(), known_options())
            .unwrap_err()
            .contains("canonical top-level")
    );

    let duplicate = format!(
        r#"{{"contents":[{{"role":"model","parts":[{{"functionCall":{{"name":"a"}},"thoughtSignature":"{signature}","thoughtSignature":"{signature}"}}]}}]}}"#
    );
    assert!(
        validate_gemini_thought_signatures(duplicate.as_bytes(), known_options())
            .unwrap_err()
            .contains("canonical top-level")
    );

    let explicit_null =
        br#"{"contents":[{"role":"model","parts":[{"text":"x","thoughtSignature":null}]}]}"#;
    assert!(
        validate_gemini_thought_signatures(explicit_null, known_options())
            .unwrap_err()
            .contains("empty thoughtSignature")
    );
}

#[test]
fn thought_validation_allows_first_bypass_and_unsigned_parallel_sibling() {
    let input = format!(
        r#"{{"request":{{"contents":[{{"role":"model","parts":[{{"functionCall":{{"name":"a"}},"thoughtSignature":"{GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR}"}},{{"functionCall":{{"name":"b"}}}}]}}]}}}}"#
    );
    validate_gemini_thought_signatures(
        input.as_bytes(),
        GeminiThoughtSignatureValidationOptions {
            allow_bypass_sentinel: true,
            ..GeminiThoughtSignatureValidationOptions::default()
        },
    )
    .unwrap();
}

#[test]
fn thought_validation_rejects_response_signature_and_sibling_bypass() {
    let response = br#"{"contents":[{"role":"user","parts":[{"functionResponse":{"name":"a"},"thoughtSignature":"bad"}]}]}"#;
    assert!(validate_gemini_thought_signatures(
        response,
        GeminiThoughtSignatureValidationOptions::default()
    )
    .unwrap_err()
    .contains("functionResponse"));
    let sibling = format!(
        r#"{{"contents":[{{"role":"model","parts":[{{"functionCall":{{"name":"a"}},"thoughtSignature":"{GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR}"}},{{"functionCall":{{"name":"b"}},"thoughtSignature":"{GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR}"}}]}}]}}"#
    );
    assert!(validate_gemini_thought_signatures(
        sibling.as_bytes(),
        GeminiThoughtSignatureValidationOptions {
            allow_bypass_sentinel: true,
            ..GeminiThoughtSignatureValidationOptions::default()
        }
    )
    .is_err());
}

#[test]
fn pairing_accepts_parallel_group() {
    let input = br#"{"contents":[{"role":"model","parts":[{"functionCall":{"id":"1","name":"a"}},{"functionCall":{"id":"2","name":"b"}}]},{"role":"user","parts":[{"functionResponse":{"id":"1","name":"a"}},{"functionResponse":{"id":"2","name":"b"}}]}]}"#;
    validate_gemini_function_call_pairing(input).unwrap();
}

#[test]
fn pairing_rejects_boundaries_counts_ids_names_and_interleaving() {
    let invalid = [
        br#"{"contents":[{"parts":[{"functionCall":{"name":"a"}}]},{"parts":[{"text":"boundary"}]}]}"#.as_slice(),
        br#"{"contents":[{"parts":[{"functionCall":{"name":"a"}},{"functionCall":{"name":"b"}}]},{"parts":[{"functionResponse":{"name":"a"}}]}]}"#.as_slice(),
        br#"{"contents":[{"parts":[{"functionCall":{"id":"1","name":"a"}}]},{"parts":[{"functionResponse":{"id":"2","name":"a"}}]}]}"#.as_slice(),
        br#"{"contents":[{"parts":[{"functionCall":{"name":"a"}}]},{"parts":[{"functionResponse":{"name":"b"}}]}]}"#.as_slice(),
        br#"{"contents":[{"parts":[{"functionCall":{"name":"a"}},{"functionResponse":{"name":"a"}}]}]}"#.as_slice(),
    ];
    for input in invalid {
        assert!(validate_gemini_function_call_pairing(input).is_err());
    }
}
