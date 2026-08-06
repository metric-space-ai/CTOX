// ref: internal/translator/openai/gemini/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
use super::{
    convert_gemini_request_to_openai, convert_openai_response_to_gemini_non_stream,
    convert_openai_response_to_gemini_stream, OpenAiToGeminiState,
};
use crate::sdk::translator::{gemini, openai, Registry, ResponseTransform, TranslationState};
use std::sync::Arc;

pub fn register_openai_gemini(registry: &Registry) {
    registry.register_pair(
        gemini(),
        openai(),
        Arc::new(convert_gemini_request_to_openai),
        ResponseTransform {
            stream: Some(Arc::new(|_, _, _, _, raw, state| {
                convert_openai_response_to_gemini_stream(raw, gemini_state(state))
            })),
            non_stream: Some(Arc::new(|_, _, _, _, raw, _| {
                convert_openai_response_to_gemini_non_stream(raw)
            })),
            token_count: Some(Arc::new(|_, n| super::gemini_token_count(n))),
        },
    );
}

fn gemini_state(state: &mut TranslationState) -> &mut OpenAiToGeminiState {
    if state
        .as_ref()
        .is_none_or(|value| !value.is::<OpenAiToGeminiState>())
    {
        *state = Some(Box::new(OpenAiToGeminiState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<OpenAiToGeminiState>())
        .expect("OpenAI-to-Gemini state was initialized with the expected type")
}
