// ref: internal/translator/openai/interactions/chat-completions/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Wire-up for the OpenAI chat-completions <-> Interactions pair.

use std::sync::Arc;

use crate::sdk::translator::{interactions, openai, Registry, ResponseTransform};

use super::{
    convert_interactions_request_to_openai, convert_interactions_response_to_openai,
    convert_interactions_response_to_openai_non_stream, convert_openai_request_to_interactions,
    convert_openai_response_to_interactions, convert_openai_response_to_interactions_non_stream,
};

/// Replaces the upstream Go `init()` registration by explicitly registering
/// the request path and the reverse response path for both directions of the
/// OpenAI chat-completions <-> Interactions pair.
pub fn register_openai_chat_interactions(registry: &Registry) {
    registry.register_pair(
        openai(),
        interactions(),
        Arc::new(convert_openai_request_to_interactions),
        ResponseTransform {
            stream: Some(Arc::new(|ctx, model, original, request, raw, state| {
                convert_openai_response_to_interactions(ctx, model, original, request, raw, state)
            })),
            non_stream: Some(Arc::new(|ctx, model, original, request, raw, state| {
                convert_openai_response_to_interactions_non_stream(
                    ctx, model, original, request, raw, state,
                )
            })),
            token_count: None,
        },
    );
    registry.register_pair(
        interactions(),
        openai(),
        Arc::new(convert_interactions_request_to_openai),
        ResponseTransform {
            stream: Some(Arc::new(|ctx, model, original, request, raw, state| {
                convert_interactions_response_to_openai(ctx, model, original, request, raw, state)
            })),
            non_stream: Some(Arc::new(|ctx, model, original, request, raw, state| {
                convert_interactions_response_to_openai_non_stream(
                    ctx, model, original, request, raw, state,
                )
            })),
            token_count: None,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdk::translator::TranslationContext;

    #[test]
    fn registration_activates_both_requests_and_response_modes() {
        let registry = Registry::new();
        register_openai_chat_interactions(&registry);

        for (from, to) in [(openai(), interactions()), (interactions(), openai())] {
            assert!(registry.has_request_transformer(&from, &to));
            assert!(registry.has_response_transformer(&from, &to));
            assert!(registry.has_stream_response_transformer(&from, &to));
            assert!(registry.has_non_stream_response_transformer(&from, &to));
        }

        let request = registry.translate_request(
            &TranslationContext::default(),
            &openai(),
            &interactions(),
            "gpt-test",
            br#"{"messages":[{"role":"user","content":"hello"}]}"#,
            false,
        );
        let value: serde_json::Value = serde_json::from_slice(&request).unwrap();
        assert_eq!(value["input"][0]["type"], "user_input");
        assert_eq!(value["input"][0]["content"][0]["text"], "hello");
    }
}
