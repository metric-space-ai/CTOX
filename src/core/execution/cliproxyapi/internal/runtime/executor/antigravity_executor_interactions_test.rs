// ref: internal/runtime/executor/antigravity_executor_interactions_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::translator::register_all;
use crate::sdk::translator::{Format, Registry, TranslationContext};
use serde_json::{json, Value};

#[test]
fn translates_interactions_request_to_antigravity_stream_envelope() {
    let registry = Registry::new();
    register_all(&registry);
    let body = registry.translate_request(&TranslationContext::default(), &Format::from("interactions"), &Format::from("antigravity"), "gemini-3.1-flash-lite", br#"{"model":"gemini-3.1-flash-lite","input":[{"type":"user_input","content":[{"type":"input_text","text":"hi"}]}],"stream":true}"#, true);
    let value: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        value.pointer("/request/contents/0/role"),
        Some(&json!("user"))
    );
    assert_eq!(
        value.pointer("/request/contents/0/parts/0/text"),
        Some(&json!("hi"))
    );
    assert!(value.get("input").is_none());
}
