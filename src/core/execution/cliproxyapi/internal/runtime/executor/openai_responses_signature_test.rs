// ref: internal/runtime/executor/openai_responses_signature_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::borrow::Cow;

use base64::{engine::general_purpose, Engine as _};
use serde_json::Value;

use super::openai_responses_signature::sanitize_openai_responses_reasoning_encrypted_content;

fn valid_encrypted_content() -> String {
    let mut payload = vec![0_u8; 1 + 8 + 16 + 16 + 32];
    payload[0] = 0x80;
    for (index, byte) in payload.iter_mut().enumerate().skip(9) {
        *byte = index as u8;
    }
    general_purpose::URL_SAFE_NO_PAD.encode(payload)
}

#[test]
fn strips_orphan_ids_when_store_is_disabled() {
    let valid = valid_encrypted_content();
    let body = format!(
        r#"{{"store":false,"input":[{{"id":"rs_bad","type":"reasoning","encrypted_content":"bad","summary":[]}},{{"id":"rs_orphan","type":"reasoning","summary":[]}},{{"id":"rs_good","type":"reasoning","encrypted_content":"{valid}","summary":[]}},{{"id":"msg_1","type":"message","role":"user","content":"hi"}}]}}"#
    );
    let output = sanitize_openai_responses_reasoning_encrypted_content("test", body.as_bytes());
    let value: Value = serde_json::from_slice(&output).unwrap();
    assert!(value.pointer("/input/0/encrypted_content").is_none());
    assert!(value.pointer("/input/0/id").is_none());
    assert!(value.pointer("/input/1/id").is_none());
    assert_eq!(value["input"][2]["id"], "rs_good");
    assert_eq!(value["input"][2]["encrypted_content"], valid);
    assert_eq!(value["input"][3]["id"], "msg_1");
}

#[test]
fn keeps_ids_when_store_is_enabled() {
    let body = br#"{"store":true,"input":[{"id":"rs_bad","type":"reasoning","encrypted_content":"bad","summary":[]},{"id":"rs_orphan","type":"reasoning","summary":[]}]}"#;
    let output = sanitize_openai_responses_reasoning_encrypted_content("test", body);
    let value: Value = serde_json::from_slice(&output).unwrap();
    assert!(value.pointer("/input/0/encrypted_content").is_none());
    assert_eq!(value["input"][0]["id"], "rs_bad");
    assert_eq!(value["input"][1]["id"], "rs_orphan");
}

#[test]
fn noop_borrows_the_original_body() {
    let valid = valid_encrypted_content();
    let body = format!(
        r#"{{"store":false,"input":[{{"id":"rs_good","type":"reasoning","encrypted_content":"{valid}","summary":[]}},{{"role":"user","content":"hi"}}]}}"#
    );
    let output = sanitize_openai_responses_reasoning_encrypted_content("test", body.as_bytes());
    assert!(matches!(output, Cow::Borrowed(_)));
    assert_eq!(output.as_ref(), body.as_bytes());
}

#[test]
fn invalid_json_and_non_array_input_are_byte_identical_noops() {
    for body in [b"not-json".as_slice(), br#"{"input":"hello"}"#] {
        let output = sanitize_openai_responses_reasoning_encrypted_content("", body);
        assert!(matches!(output, Cow::Borrowed(_)));
        assert_eq!(output.as_ref(), body);
    }
}
