// ref: internal/translator/gemini/claude/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::translator::{claude, gemini, Registry, ResponseTransform, TranslationState};

use super::{
    convert_claude_request_to_gemini, convert_gemini_response_to_claude_non_stream,
    convert_gemini_response_to_claude_stream, gemini_claude_token_count, GeminiToClaudeStreamState,
};

pub fn register_claude_gemini(registry: &Registry) {
    registry.register_pair(
        claude(),
        gemini(),
        Arc::new(convert_claude_request_to_gemini),
        ResponseTransform {
            stream: Some(Arc::new(|_, _, original, request, raw, state| {
                convert_gemini_response_to_claude_stream(
                    original,
                    request,
                    raw,
                    stream_state(state),
                )
            })),
            non_stream: Some(Arc::new(|_, _, original, request, raw, _| {
                convert_gemini_response_to_claude_non_stream(original, request, raw)
            })),
            token_count: Some(Arc::new(|_, count| gemini_claude_token_count(count))),
        },
    );
}

fn stream_state(state: &mut TranslationState) -> &mut GeminiToClaudeStreamState {
    let replace = state
        .as_ref()
        .is_none_or(|value| !value.is::<GeminiToClaudeStreamState>());
    if replace {
        *state = Some(Box::new(GeminiToClaudeStreamState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<GeminiToClaudeStreamState>())
        .expect("Gemini Claude state has the expected type")
}
