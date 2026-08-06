// ref: internal/runtime/executor/helps/codex_input_ids_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::codex_input_ids::{
    sanitize_codex_input_item_ids, shorten_codex_input_item_id_with_attempt,
};

fn input(body: &[u8]) -> Vec<Value> {
    serde_json::from_slice::<Value>(body).unwrap()["input"]
        .as_array()
        .unwrap()
        .clone()
}

#[test]
fn boundaries_use_unicode_character_count() {
    let id64 = "a".repeat(64);
    let id65 = "b".repeat(65);
    let unicode65 = "界".repeat(65);
    let body =
        format!(r#"{{"input":[{{"id":"{id64}"}},{{"id":"{id65}"}},{{"id":"{unicode65}"}}]}}"#);
    let items = input(&sanitize_codex_input_item_ids(body.as_bytes()));
    assert_eq!(items[0]["id"], id64);
    assert_eq!(items[1]["id"].as_str().unwrap().chars().count(), 64);
    assert_eq!(items[2]["id"].as_str().unwrap().chars().count(), 64);
}

#[test]
fn normalizes_only_invalid_message_ids_deterministically() {
    let invalid = "item_74ec40c883248ebb4885ec84";
    let body = format!(
        r#"{{"input":[{{"type":"message","id":"{invalid}","role":"user"}},{{"type":"message","id":"msg-1","role":"assistant"}},{{"type":"function_call","id":"item_call"}}]}}"#
    );
    let first = sanitize_codex_input_item_ids(body.as_bytes());
    let second = sanitize_codex_input_item_ids(body.as_bytes());
    let items = input(&first);
    assert_eq!(items[0]["id"], format!("msg_{invalid}"));
    assert_eq!(items[1]["id"], "msg-1");
    assert_eq!(items[2]["id"], "item_call");
    assert_eq!(first, second);
}

#[test]
fn drops_only_overlong_nonempty_encrypted_reasoning() {
    let long_reasoning = format!("rs_{}", "a".repeat(64));
    let short_reasoning = format!("rs_{}", "b".repeat(48));
    let long_call = "call-item-".repeat(8);
    let body = format!(
        r#"{{"input":[{{"type":"message","id":"msg-1"}},{{"type":"reasoning","id":"{long_reasoning}","encrypted_content":"cipher"}},{{"type":"reasoning","id":"{short_reasoning}","encrypted_content":"cipher"}},{{"type":"function_call","id":"{long_call}"}}]}}"#
    );
    let items = input(&sanitize_codex_input_item_ids(body.as_bytes()));
    assert_eq!(items.len(), 3);
    assert_eq!(items[0]["id"], "msg-1");
    assert_eq!(items[1]["id"], short_reasoning);
    assert_ne!(items[2]["id"], long_call);
    assert_eq!(items[2]["id"].as_str().unwrap().chars().count(), 64);
}

#[test]
fn absent_empty_and_null_encrypted_content_are_shortened_not_dropped() {
    let long = format!("rs_{}", "a".repeat(64));
    for suffix in [
        "",
        r#","encrypted_content":"""#,
        r#","encrypted_content":null"#,
    ] {
        let body =
            format!(r#"{{"input":[{{"type":"reasoning","id":"{long}"{suffix},"summary":[]}}]}}"#);
        let items = input(&sanitize_codex_input_item_ids(body.as_bytes()));
        assert_eq!(items.len(), 1);
        assert_ne!(items[0]["id"], long);
        assert_eq!(items[0]["id"].as_str().unwrap().chars().count(), 64);
    }
}

#[test]
fn avoids_existing_short_id_collision_with_stable_attempt_hash() {
    let long = "grok-item-".repeat(10);
    let colliding = shorten_codex_input_item_id_with_attempt(&long, 0);
    let body = format!(r#"{{"input":[{{"id":"{long}"}},{{"id":"{colliding}"}}]}}"#);
    let first = sanitize_codex_input_item_ids(body.as_bytes());
    let second = sanitize_codex_input_item_ids(body.as_bytes());
    let items = input(&first);
    assert_ne!(items[0]["id"], colliding);
    assert_eq!(items[1]["id"], colliding);
    assert_eq!(first, second);
}

#[test]
fn unsupported_and_noop_payloads_are_byte_identical() {
    for body in [
        b"not-json".as_slice(),
        br#"{"input":{"id":"item-1"}}"#,
        br#" { "input" : [1,{"id":2},{"id":"item-1"}], "keep" : 1 } "#,
    ] {
        assert_eq!(sanitize_codex_input_item_ids(body), body);
    }
}

#[test]
fn mutation_preserves_outer_payload_bytes_and_unmodified_item_bytes() {
    let long = "z".repeat(65);
    let body = format!(
        r#" {{
  "keep" : 900719925474099312345, "input" : [ {{ "id" : "ok", "n" : 1 }},{{"id":"{long}"}} ], "tail" : true
}} "#
    );
    let output = String::from_utf8(sanitize_codex_input_item_ids(body.as_bytes())).unwrap();
    assert!(output.starts_with(" {\n  \"keep\" : 900719925474099312345, \"input\" : ["));
    assert!(output.contains("{ \"id\" : \"ok\", \"n\" : 1 }"));
    assert!(output.ends_with(", \"tail\" : true\n} "));
}
