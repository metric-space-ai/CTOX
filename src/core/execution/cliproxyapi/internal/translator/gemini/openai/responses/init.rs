// ref: internal/translator/gemini/openai/responses/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use super::{
    convert_gemini_response_to_openai_responses_non_stream,
    convert_gemini_response_to_openai_responses_stream, convert_openai_responses_request_to_gemini,
    GeminiToResponsesState,
};
use crate::sdk::translator::{
    gemini, openai_response, Registry, ResponseTransform, TranslationState,
};

/// Activates the verified request plus non-stream and stream response paths.
pub fn register_openai_responses_gemini_request(registry: &Registry) {
    registry.register_pair(
        openai_response(),
        gemini(),
        Arc::new(convert_openai_responses_request_to_gemini),
        ResponseTransform {
            stream: Some(Arc::new(|_, _, original_request, request, raw, state| {
                let state = gemini_state(state);
                convert_gemini_response_to_openai_responses_stream(
                    original_request,
                    request,
                    raw,
                    state,
                )
            })),
            non_stream: Some(Arc::new(|_, _, original_request, request, raw, _| {
                convert_gemini_response_to_openai_responses_non_stream(
                    original_request,
                    request,
                    raw,
                )
            })),
            token_count: None,
        },
    );
}

fn gemini_state(state: &mut TranslationState) -> &mut GeminiToResponsesState {
    let needs_state = match state.as_ref() {
        Some(value) => !value.is::<GeminiToResponsesState>(),
        None => true,
    };
    if needs_state {
        *state = Some(Box::new(GeminiToResponsesState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<GeminiToResponsesState>())
        .expect("Gemini state was initialized with the expected type")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdk::translator::TranslationContext;

    #[test]
    fn registration_activates_request_and_both_response_directions() {
        let registry = Registry::new();
        register_openai_responses_gemini_request(&registry);

        assert!(registry.has_request_transformer(&openai_response(), &gemini()));
        assert!(registry.has_response_transformer(&openai_response(), &gemini()));
        assert!(registry.has_non_stream_response_transformer(&openai_response(), &gemini()));
        assert!(registry.has_stream_response_transformer(&openai_response(), &gemini()));

        let output = registry.translate_request(
            &TranslationContext::default(),
            &openai_response(),
            &gemini(),
            "gemini-3.6-flash-high",
            br#"{"input":"hello"}"#,
            false,
        );
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["contents"][0]["parts"][0]["text"], "hello");

        let mut state = None;
        let response = registry.translate_non_stream(
            &TranslationContext::default(),
            &gemini(),
            &openai_response(),
            "gemini-3.6-flash-high",
            br#"{"model":"gemini-3.6-flash-high"}"#,
            b"",
            br#"{"responseId":"native","createTime":"2026-08-03T12:34:56Z","candidates":[{"content":{"parts":[{"text":"done"}]}}]}"#,
            &mut state,
        );
        let value: serde_json::Value = serde_json::from_slice(&response).unwrap();
        assert_eq!(value["id"], "resp_native");
        assert_eq!(value["output"][0]["content"][0]["text"], "done");

        let mut state = None;
        let mut events = registry.translate_stream(
            &TranslationContext::default(),
            &gemini(),
            &openai_response(),
            "gemini-3.6-flash-high",
            br#"{"model":"gemini-3.6-flash-high"}"#,
            b"",
            br#"{"responseId":"native-stream","createTime":"2026-08-03T12:34:56Z","candidates":[{"content":{"parts":[{"text":"do"}]}}]}"#,
            &mut state,
        );
        events.extend(registry.translate_stream(
            &TranslationContext::default(),
            &gemini(),
            &openai_response(),
            "gemini-3.6-flash-high",
            br#"{"model":"gemini-3.6-flash-high"}"#,
            b"",
            br#"{"candidates":[{"content":{"parts":[{"text":"ne"}]},"finishReason":"STOP"}]}"#,
            &mut state,
        ));
        let wire = events.concat();
        assert!(wire
            .windows(b"event: response.output_text.delta".len())
            .any(|window| window == b"event: response.output_text.delta"));
        assert!(wire
            .windows(b"event: response.completed".len())
            .any(|window| window == b"event: response.completed"));
    }
}
