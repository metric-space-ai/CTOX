// ref: internal/translator/interactions/claude/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::translator::{claude, interactions, Registry, ResponseTransform, TranslationState};

use super::{
    convert_claude_request_to_interactions, convert_interactions_response_to_claude,
    convert_interactions_response_to_claude_non_stream, InteractionsToClaudeStreamState,
};

pub fn register_claude_interactions(registry: &Registry) {
    registry.register_pair(
        claude(),
        interactions(),
        Arc::new(convert_claude_request_to_interactions),
        ResponseTransform {
            stream: Some(Arc::new(|_, model, original, request, raw, state| {
                convert_interactions_response_to_claude(
                    model,
                    original,
                    request,
                    raw,
                    stream_state(state),
                )
            })),
            non_stream: Some(Arc::new(|_, model, original, request, raw, _| {
                convert_interactions_response_to_claude_non_stream(model, original, request, raw)
            })),
            token_count: None,
        },
    );
}

fn stream_state(state: &mut TranslationState) -> &mut InteractionsToClaudeStreamState {
    if state
        .as_ref()
        .is_none_or(|value| !value.is::<InteractionsToClaudeStreamState>())
    {
        *state = Some(Box::new(InteractionsToClaudeStreamState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<InteractionsToClaudeStreamState>())
        .expect("Interactions-to-Claude state type was initialized")
}
