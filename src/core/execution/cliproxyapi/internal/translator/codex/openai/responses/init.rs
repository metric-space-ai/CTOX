// ref: internal/translator/codex/openai/responses/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::translator::{codex, openai_response, Registry, ResponseTransform};

use super::{
    convert_codex_response_to_openai_responses,
    convert_codex_response_to_openai_responses_non_stream,
    convert_openai_responses_request_to_codex,
};

pub fn register_openai_responses_codex(registry: &Registry) {
    registry.register_pair(
        openai_response(),
        codex(),
        Arc::new(convert_openai_responses_request_to_codex),
        ResponseTransform {
            stream: Some(Arc::new(|_, _, _, _, raw, _| {
                convert_codex_response_to_openai_responses(raw)
            })),
            non_stream: Some(Arc::new(|_, _, _, _, raw, _| {
                convert_codex_response_to_openai_responses_non_stream(raw)
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
    fn registry_activates_request_and_both_response_directions() {
        let registry = Registry::new();
        register_openai_responses_codex(&registry);
        assert!(registry.has_request_transformer(&openai_response(), &codex()));
        assert!(registry.has_stream_response_transformer(&openai_response(), &codex()));
        assert!(registry.has_non_stream_response_transformer(&openai_response(), &codex()));

        let output = registry.translate_request(
            &TranslationContext::default(),
            &openai_response(),
            &codex(),
            "gpt-5.6",
            br#"{"input":"hello"}"#,
            false,
        );
        let output: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(output["input"][0]["content"][0]["text"], "hello");
    }
}
