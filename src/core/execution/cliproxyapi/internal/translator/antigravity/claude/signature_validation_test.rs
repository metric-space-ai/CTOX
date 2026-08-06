// ref: internal/translator/antigravity/claude/signature_validation_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};
use serde_json::Value;

use crate::internal::cache::signature_cache::{
    set_signature_bypass_strict_mode, signature_cache_test_guard,
};

use super::{
    decode_gemini_claude_carrier_signature, encode_gemini_claude_carrier_signature,
    strip_empty_signature_thinking_blocks, strip_invalid_bypass_signature_thinking_blocks,
    strip_invalid_gemini_signature_thinking_blocks, validate_claude_bypass_signatures,
};

fn valid_signature() -> String {
    general_purpose::STANDARD.encode([0x12, 0x08, 0x0a, 0x06, 0x01, 0x0c, 0x39, 0xd6, 0xc7, 0x34])
}

fn filter(input: Value) -> Value {
    serde_json::from_slice(&strip_invalid_gemini_signature_thinking_blocks(
        &serde_json::to_vec(&input).unwrap(),
    ))
    .unwrap()
}

#[test]
fn carrier_signature_round_trip() {
    let signature = valid_signature();
    for (direction, kind) in [
        ("next", "text"),
        ("previous", "function"),
        ("standalone", "any"),
    ] {
        let encoded = encode_gemini_claude_carrier_signature(&signature, direction, kind);
        let decoded = decode_gemini_claude_carrier_signature(&encoded).unwrap();
        assert!(decoded.marked);
        assert_eq!(decoded.signature, signature);
        assert_eq!(decoded.direction, direction);
        assert_eq!(decoded.target_kind, kind);
    }
}

#[test]
fn preserves_marked_non_empty_thinking_with_matching_targets() {
    let signature = valid_signature();
    let standalone = encode_gemini_claude_carrier_signature(&signature, "standalone", "text");
    let next_function = encode_gemini_claude_carrier_signature(&signature, "next", "function");
    let invalid_previous = encode_gemini_claude_carrier_signature(&signature, "previous", "text");
    let output = filter(
        serde_json::json!({"messages":[{"role":"assistant","content":[
            {"type":"thinking","thinking":"signed thought","signature":standalone},
            {"type":"thinking","thinking":"tool preface","signature":next_function},
            {"type":"tool_use","id":"tool-1","name":"run","input":{}},
            {"type":"thinking","thinking":"invalid backward","signature":invalid_previous}
        ]}]}),
    );
    let content = output["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 3);
    assert_eq!(content[2]["type"], "tool_use");
}

#[test]
fn drops_mismatched_directional_thinking() {
    let signature = valid_signature();
    let next_function = encode_gemini_claude_carrier_signature(&signature, "next", "function");
    let standalone_function =
        encode_gemini_claude_carrier_signature(&signature, "standalone", "function");
    let output = filter(
        serde_json::json!({"messages":[{"role":"assistant","content":[
            {"type":"thinking","thinking":"wrong next target","signature":next_function},
            {"type":"text","text":"visible"},
            {"type":"thinking","thinking":"wrong standalone target","signature":standalone_function}
        ]}]}),
    );
    let content = output["messages"][0]["content"].as_array().unwrap();
    assert_eq!(
        content,
        &[serde_json::json!({"type":"text","text":"visible"})]
    );
}

#[test]
fn raw_carrier_is_assistant_only() {
    let signature = valid_signature();
    let output = filter(serde_json::json!({"messages":[
        {"role":"user","content":[{"type":"thinking","thinking":"","signature":signature},{"type":"text","text":"user"}]},
        {"role":"assistant","content":[{"type":"thinking","thinking":"","signature":signature},{"type":"text","text":"assistant"}]}
    ]}));
    assert_eq!(
        output["messages"][0]["content"].as_array().unwrap().len(),
        1
    );
    assert_eq!(
        output["messages"][1]["content"].as_array().unwrap().len(),
        2
    );
}

#[test]
fn invalid_carriers_are_removed_without_touching_valid_neighbors() {
    let signature = valid_signature();
    let previous = encode_gemini_claude_carrier_signature(&signature, "previous", "text");
    let output = filter(
        serde_json::json!({"messages":[{"role":"assistant","content":[
            {"type":"text","text":"first"},
            {"type":"thinking","thinking":"","signature":signature},
            {"type":"thinking","thinking":"","signature":previous},
            {"type":"thinking","thinking":"","signature":"cpa-gemini-carrier-v1:previous:text:invalid"},
            {"type":"thinking","thinking":"","signature":"invalid"},
            {"type":"text","text":"last"}
        ]}]}),
    );
    let content = output["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 4);
    assert_eq!(content[3]["text"], "last");
}

#[test]
fn prefix_cleanup_is_deliberately_shallow() {
    let _guard = signature_cache_test_guard();
    let input = br#" {"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"legacy","signature":"E!!!!"},{"type":"thinking","thinking":"foreign","signature":"gAAAA"},{"type":"text","text":"answer"}]}]} "#;
    let output = strip_empty_signature_thinking_blocks(input);
    let output: Value = serde_json::from_slice(&output).unwrap();
    let parts = output["messages"][0]["content"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["signature"], "E!!!!");
}

#[test]
fn strict_bypass_rejects_shallow_marker_that_basic_mode_accepts() {
    let _guard = signature_cache_test_guard();
    let input = br#"{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"legacy","signature":"EgA="},{"type":"text","text":"answer"}]}]}"#;
    let previous = set_signature_bypass_strict_mode(false);
    let basic = strip_invalid_bypass_signature_thinking_blocks(input);
    assert_eq!(basic, input);
    assert!(validate_claude_bypass_signatures(input).is_ok());

    set_signature_bypass_strict_mode(true);
    let strict: Value =
        serde_json::from_slice(&strip_invalid_bypass_signature_thinking_blocks(input)).unwrap();
    assert_eq!(
        strict["messages"][0]["content"].as_array().unwrap().len(),
        1
    );
    assert!(validate_claude_bypass_signatures(input).is_err());
    set_signature_bypass_strict_mode(previous);
}

#[test]
fn bypass_validation_reports_missing_signature_location() {
    let _guard = signature_cache_test_guard();
    let input = br#"{"messages":[{"content":[{"type":"thinking","thinking":"why"}]}]}"#;
    let error = validate_claude_bypass_signatures(input).unwrap_err();
    assert!(error.contains("messages[0].content[0]"));
}
