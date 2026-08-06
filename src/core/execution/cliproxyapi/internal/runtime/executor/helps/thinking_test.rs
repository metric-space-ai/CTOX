// ref: internal/runtime/executor/helps/thinking_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use serde_json::Value;

use super::apply_thinking_with_source_payload;
use crate::internal::thinking::ThinkingEngine;
use crate::sdk::translator::{builtin, Format, PluginHooks, TranslationContext};

struct SummaryRemovingPluginHooks;

impl PluginHooks for SummaryRemovingPluginHooks {
    fn normalize_request(
        &self,
        _context: &TranslationContext,
        _from: &Format,
        _to: &Format,
        _model: &str,
        body: Vec<u8>,
        _stream: bool,
    ) -> Vec<u8> {
        let mut document = serde_json::from_slice::<Value>(&body).expect("translated request");
        assert_eq!(
            document.pointer("/generationConfig/thinkingConfig/includeThoughts"),
            Some(&Value::Bool(true)),
            "request normalizer must receive enabled summary"
        );
        document
            .pointer_mut("/generationConfig/thinkingConfig")
            .and_then(Value::as_object_mut)
            .expect("thinking config")
            .remove("includeThoughts");
        serde_json::to_vec(&document).expect("normalized request")
    }
}

#[test]
fn apply_thinking_with_source_payload_preserves_normalizer_summary_removal() {
    let registry = builtin::registry();
    registry.set_plugin_hooks(Some(Arc::new(SummaryRemovingPluginHooks)));
    let source = br#"{"model":"gemini-3.6-flash","reasoning":{"effort":"high","summary":"auto"},"input":"hi"}"#;
    let translated = registry.translate_request(
        &TranslationContext::default(),
        &Format::from("openai-response"),
        &Format::from("gemini"),
        "gemini-3.6-flash",
        source,
        false,
    );
    let translated_json = serde_json::from_slice::<Value>(&translated).unwrap();
    assert!(translated_json
        .pointer("/generationConfig/thinkingConfig/includeThoughts")
        .is_none());

    let output = apply_thinking_with_source_payload(
        &ThinkingEngine::default(),
        &registry,
        &translated,
        source,
        source,
        "gemini-3.6-flash",
        "openai-response",
        "gemini",
        "gemini",
    )
    .unwrap();
    let output = serde_json::from_slice::<Value>(&output).unwrap();
    assert!(output
        .pointer("/generationConfig/thinkingConfig/includeThoughts")
        .is_none());
}

#[test]
fn apply_thinking_with_source_payload_preserves_original_only_summary() {
    let registry = builtin::registry();
    let current_source = br#"{"model":"gemini-3.6-flash","input":"hi"}"#;
    let original_source =
        br#"{"model":"gemini-3.6-flash","reasoning":{"summary":null},"input":"hi"}"#;
    let body = br#"{"generationConfig":{"thinkingConfig":{"thinkingLevel":"high"}}}"#;

    let output = apply_thinking_with_source_payload(
        &ThinkingEngine::default(),
        &registry,
        body,
        current_source,
        original_source,
        "gemini-3.6-flash",
        "openai-response",
        "gemini",
        "gemini",
    )
    .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&output)
            .unwrap()
            .pointer("/generationConfig/thinkingConfig/includeThoughts"),
        Some(&Value::Bool(false))
    );
}

#[test]
fn missing_transform_does_not_apply_source_shaped_summary() {
    let registry = crate::sdk::translator::Registry::new();
    let source = br#"{"reasoning":{"summary":"auto"}}"#;
    let body = br#" { "keep" : 900719925474099312345 } "#;
    let output = apply_thinking_with_source_payload(
        &ThinkingEngine::default(),
        &registry,
        body,
        source,
        source,
        "unknown-model",
        "openai-response",
        "gemini",
        "gemini",
    )
    .unwrap();
    assert_eq!(output, body);
}
