// ref: internal/runtime/executor/antigravity_reasoning_replay_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::antigravity_reasoning_replay::apply_antigravity_reasoning_replay_items;
use serde_json::{json, Value};

#[test]
fn merges_signature_into_semantically_matching_function_call() {
    let payload = br#"{"request":{"contents":[{"role":"model","parts":[{"functionCall":{"id":"call-1","name":"weather","args":{"city":"Berlin"}}}]}]}}"#;
    let item = br#"{"type":"function_call_part","contentIndex":0,"partIndex":0,"call_id":"call-1","name":"weather","args":{"city":"Berlin"},"thoughtSignature":"native-signature-123456"}"#.to_vec();
    let (out, applied) = apply_antigravity_reasoning_replay_items(payload, &[item]).unwrap();
    let value: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(applied, 1);
    assert_eq!(
        value.pointer("/request/contents/0/parts/0/thoughtSignature"),
        Some(&json!("native-signature-123456"))
    );
}

#[test]
fn replay_rejects_tool_output_across_user_boundary_without_mutation() {
    let payload = br#"{"request":{"contents":[{"role":"model","parts":[{"text":"old","thoughtSignature":"stale-signature-123456"}]},{"role":"user","parts":[{"text":"new turn"}]},{"role":"user","parts":[{"functionResponse":{"id":"call-1","name":"weather","response":{}}}]}]}}"#;
    let item = br#"{"type":"function_call_part","contentIndex":0,"partIndex":1,"call_id":"call-1","name":"weather","args":{},"thoughtSignature":"native-signature-123456"}"#.to_vec();
    let (out, applied) = apply_antigravity_reasoning_replay_items(payload, &[item]).unwrap();
    assert_eq!(applied, 0);
    assert_eq!(
        serde_json::from_slice::<Value>(&out).unwrap(),
        serde_json::from_slice::<Value>(payload).unwrap()
    );
}

#[test]
fn malformed_and_duplicate_items_fail_closed() {
    let payload = br#"{"request":{"contents":[{"role":"model","parts":[{"functionCall":{"id":"x","name":"f","args":{}}}]}]}}"#;
    let (_, applied) = apply_antigravity_reasoning_replay_items(
        payload,
        &[b"bad".to_vec(), br#"{"type":"unknown"}"#.to_vec()],
    )
    .unwrap();
    assert_eq!(applied, 0);
}
