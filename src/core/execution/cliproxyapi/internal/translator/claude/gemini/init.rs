// ref: internal/translator/claude/gemini/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::translator::{claude, gemini, Registry, ResponseTransform, TranslationState};

use super::{
    convert_claude_response_to_gemini, convert_claude_response_to_gemini_non_stream,
    convert_gemini_request_to_claude, gemini_token_count, ClaudeToGeminiState,
};

pub fn register_gemini_claude(registry: &Registry) {
    registry.register_pair(
        gemini(),
        claude(),
        Arc::new(convert_gemini_request_to_claude),
        ResponseTransform {
            stream: Some(Arc::new(|_, model, original, request, raw, state| {
                convert_claude_response_to_gemini(
                    model,
                    original,
                    request,
                    raw,
                    stream_state(state),
                )
            })),
            non_stream: Some(Arc::new(|_, model, original, request, raw, _| {
                convert_claude_response_to_gemini_non_stream(model, original, request, raw)
            })),
            token_count: Some(Arc::new(|_, count| gemini_token_count(count))),
        },
    );
}

fn stream_state(state: &mut TranslationState) -> &mut ClaudeToGeminiState {
    if state
        .as_ref()
        .is_none_or(|value| !value.is::<ClaudeToGeminiState>())
    {
        *state = Some(Box::new(ClaudeToGeminiState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<ClaudeToGeminiState>())
        .expect("Claude-to-Gemini state type was initialized")
}
