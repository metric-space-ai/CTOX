// ref: sdk/translator/registry_bytes_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use super::{gemini, openai, Registry, ResponseTransform, TranslationContext};

#[test]
fn response_transforms_return_owned_byte_payloads() {
    let registry = Registry::new();
    registry.register(
        openai(),
        gemini(),
        None,
        ResponseTransform {
            stream: Some(Arc::new(|_, _, _, _, raw, _| vec![raw.to_vec()])),
            non_stream: Some(Arc::new(|_, _, _, _, raw, _| raw.to_vec())),
            token_count: Some(Arc::new(|_, _| br#"{"totalTokens":7}"#.to_vec())),
        },
    );
    let context = TranslationContext::default();
    let mut state = None;
    assert_eq!(
        registry.translate_stream(
            &context,
            &gemini(),
            &openai(),
            "model",
            &[],
            &[],
            br#"{"chunk":true}"#,
            &mut state,
        ),
        vec![br#"{"chunk":true}"#.to_vec()]
    );
    assert_eq!(
        registry.translate_non_stream(
            &context,
            &gemini(),
            &openai(),
            "model",
            &[],
            &[],
            br#"{"done":true}"#,
            &mut state,
        ),
        br#"{"done":true}"#
    );
    assert_eq!(
        registry.translate_token_count(&context, &gemini(), &openai(), 7, br#"{"fallback":true}"#,),
        br#"{"totalTokens":7}"#
    );
}
