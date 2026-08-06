// ref: internal/signature/claude_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::{
    claude_validation::test_claude_signature, strip_invalid_claude_thinking_blocks,
    strip_invalid_claude_thinking_blocks_and_empty_messages, ClaudeSignatureValidationOptions,
};

fn content_len(output: &[u8]) -> usize {
    serde_json::from_slice::<Value>(output).unwrap()["messages"][0]["content"]
        .as_array()
        .unwrap()
        .len()
}

#[test]
fn removes_gpt_and_malformed_claude_thinking_blocks() {
    for signature in ["gAAAAABopenai-encrypted-content", "E!!!invalid!!!"] {
        let input = format!(
            r#"{{"messages":[{{"role":"assistant","content":[{{"type":"thinking","thinking":"bad","signature":"{signature}"}},{{"type":"text","text":"Answer"}}]}}]}}"#
        );
        let output = strip_invalid_claude_thinking_blocks(
            input.as_bytes(),
            ClaudeSignatureValidationOptions::default(),
        );
        assert_eq!(content_len(&output), 1);
        assert!(!String::from_utf8(output).unwrap().contains(signature));
    }
}

#[test]
fn base64_only_keeps_decodable_e_but_rejects_invalid_base64() {
    let options = ClaudeSignatureValidationOptions {
        base64_only: true,
        ..ClaudeSignatureValidationOptions::default()
    };
    for (signature, expected) in [("Ebad", 2), ("E!!!invalid!!!", 1)] {
        let input = format!(
            r#"{{"messages":[{{"content":[{{"type":"thinking","thinking":"x","signature":"{signature}"}},{{"type":"text","text":"ok"}}]}}]}}"#
        );
        assert_eq!(
            content_len(&strip_invalid_claude_thinking_blocks(
                input.as_bytes(),
                options
            )),
            expected
        );
    }
}

#[test]
fn empty_placeholder_semantics_cover_text_and_nested_thinking() {
    let options = ClaudeSignatureValidationOptions {
        allow_empty_signature_with_empty_text: true,
        ..ClaudeSignatureValidationOptions::default()
    };
    for body in [
        r#"{"type":"thinking","text":"","signature":""}"#,
        r#"{"type":"thinking","thinking":{"text":"  "},"signature":" "}"#,
        r#"{"type":"thinking","thinking":{"thinking":""}}"#,
    ] {
        let input = format!(r#"{{"messages":[{{"content":[{body}]}}]}}"#);
        assert_eq!(
            strip_invalid_claude_thinking_blocks(input.as_bytes(), options),
            input.as_bytes()
        );
    }
}

#[test]
fn drops_messages_only_when_requested_and_preserves_noop_bytes() {
    let invalid = br#" {"messages":[{"role":"assistant","content":[{"type":"thinking","signature":"bad"}]},{"role":"user","content":[{"type":"text","text":"next"}]}]} "#;
    let kept =
        strip_invalid_claude_thinking_blocks(invalid, ClaudeSignatureValidationOptions::default());
    assert_eq!(
        serde_json::from_slice::<Value>(&kept).unwrap()["messages"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let dropped = strip_invalid_claude_thinking_blocks_and_empty_messages(
        invalid,
        ClaudeSignatureValidationOptions::default(),
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&dropped).unwrap()["messages"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let noop = br#" { "messages": [{"content":[{"type":"text","text":"x"}]}] } "#;
    assert_eq!(
        strip_invalid_claude_thinking_blocks(noop, ClaudeSignatureValidationOptions::default()),
        noop
    );
}

#[test]
fn strict_keeps_real_claude_signature() {
    let signature = test_claude_signature();
    let input = format!(
        r#"{{"messages":[{{"content":[{{"type":"thinking","thinking":"x","signature":"{signature}"}}]}}]}}"#
    );
    assert_eq!(
        strip_invalid_claude_thinking_blocks(
            input.as_bytes(),
            ClaudeSignatureValidationOptions {
                strict: true,
                ..ClaudeSignatureValidationOptions::default()
            }
        ),
        input.as_bytes()
    );
}
