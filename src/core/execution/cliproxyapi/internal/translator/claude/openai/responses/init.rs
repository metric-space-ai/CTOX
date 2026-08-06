// ref: internal/translator/claude/openai/responses/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{
    convert_claude_response_to_openai_responses,
    convert_claude_response_to_openai_responses_non_stream,
    convert_openai_responses_request_to_claude, ClaudeToResponsesState,
};
use crate::sdk::translator::{
    claude, openai_response, Registry, ResponseTransform, TranslationState,
};
use std::sync::Arc;

/// Replaces Go package `init()` with an explicit activation point.
pub fn register_openai_responses_claude(registry: &Registry) {
    registry.register_pair(
        openai_response(),
        claude(),
        Arc::new(convert_openai_responses_request_to_claude),
        ResponseTransform {
            stream: Some(Arc::new(
                |_, model, original_request, request, raw, state| {
                    let state = claude_state(state);
                    convert_claude_response_to_openai_responses(
                        model,
                        original_request,
                        request,
                        raw,
                        state,
                    )
                },
            )),
            non_stream: Some(Arc::new(|_, _, original_request, request, raw, _| {
                convert_claude_response_to_openai_responses_non_stream(
                    original_request,
                    request,
                    raw,
                )
            })),
            token_count: None,
        },
    );
}

fn claude_state(state: &mut TranslationState) -> &mut ClaudeToResponsesState {
    let needs_state = match state.as_ref() {
        Some(value) => !value.is::<ClaudeToResponsesState>(),
        None => true,
    };
    if needs_state {
        *state = Some(Box::new(ClaudeToResponsesState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<ClaudeToResponsesState>())
        .expect("Claude state was initialized with the expected type")
}
