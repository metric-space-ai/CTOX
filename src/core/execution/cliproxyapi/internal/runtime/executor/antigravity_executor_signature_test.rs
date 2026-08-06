// ref: internal/runtime/executor/antigravity_executor_signature_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::translator::antigravity::gemini::convert_gemini_request_to_antigravity;
use serde_json::{json, Value};

#[test]
fn finalizes_parallel_calls_and_orders_function_responses_without_crossing_history() {
    let input = br#"{"contents":[{"role":"model","parts":[{"functionCall":{"id":"a","name":"one","args":{}}},{"functionCall":{"id":"b","name":"two","args":{}}}]},{"role":"user","parts":[{"functionResponse":{"id":"b","name":"two","response":{"v":2}}},{"functionResponse":{"id":"a","name":"one","response":{"v":1}}}]}]}"#;
    let out: Value = serde_json::from_slice(&convert_gemini_request_to_antigravity(
        "gemini-3", input, false,
    ))
    .unwrap();
    let contents = out
        .pointer("/request/contents")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(contents[1]["role"], "user");
    assert_eq!(
        contents[1].pointer("/parts/0/functionResponse/name"),
        Some(&json!("one"))
    );
    assert_eq!(
        contents[1].pointer("/parts/1/functionResponse/name"),
        Some(&json!("two"))
    );
}

#[test]
fn strict_gemini_signature_sanitizer_keeps_valid_carrier_and_strips_short_signature() {
    let input = br#"{"contents":[{"role":"model","parts":[{"text":"thought","thought":true,"thoughtSignature":"short"},{"text":"answer"}]}]}"#;
    let out: Value = serde_json::from_slice(&convert_gemini_request_to_antigravity(
        "gemini-3-pro",
        input,
        false,
    ))
    .unwrap();
    assert!(out
        .pointer("/request/contents/0/parts/0/thoughtSignature")
        .is_none());
    assert_eq!(
        out.pointer("/request/contents/0/parts/1/text"),
        Some(&json!("answer"))
    );
}
