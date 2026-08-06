// ref: internal/translator/openai/claude/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::translator::{claude, openai, Registry, ResponseTransform, TranslationState};

use super::{
    claude_token_count, convert_claude_request_to_openai, convert_openai_response_to_claude,
    convert_openai_response_to_claude_non_stream, OpenAIToClaudeStreamState,
};

/// Replaces Go package `init()` with an explicit, dependency-injected
/// activation point. The pair registers the request transform
/// `Claude Messages -> OpenAI Chat` and the inverse streaming/non-stream
/// transforms plus the token-count transform.
pub fn register_openai_claude(registry: &Registry) {
    registry.register_pair(
        claude(),
        openai(),
        Arc::new(convert_claude_request_to_openai),
        ResponseTransform {
            stream: Some(Arc::new(|context, model, original, request, raw, state| {
                convert_openai_response_to_claude(
                    context,
                    model,
                    original,
                    request,
                    raw,
                    claude_state(state),
                )
            })),
            non_stream: Some(Arc::new(|context, model, original, request, raw, state| {
                convert_openai_response_to_claude_non_stream(
                    context,
                    model,
                    original,
                    request,
                    raw,
                    claude_state(state),
                )
            })),
            token_count: Some(Arc::new(claude_token_count)),
        },
    );
}

fn claude_state(state: &mut TranslationState) -> &mut OpenAIToClaudeStreamState {
    let needs_state = match state.as_ref() {
        Some(value) => !value.is::<OpenAIToClaudeStreamState>(),
        None => true,
    };
    if needs_state {
        *state = Some(Box::new(OpenAIToClaudeStreamState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<OpenAIToClaudeStreamState>())
        .expect("Claude Chat stream state was initialized with the expected type")
}
