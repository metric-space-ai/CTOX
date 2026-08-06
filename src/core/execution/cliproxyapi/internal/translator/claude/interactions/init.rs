// ref: internal/translator/claude/interactions/init.go:1-19 @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::translator::{claude, interactions, Registry, ResponseTransform, TranslationState};

use super::{
    convert_claude_response_to_interactions, convert_claude_response_to_interactions_non_stream,
    convert_interactions_request_to_claude, ClaudeToInteractionsState,
};

pub fn register_interactions_claude(registry: &Registry) {
    registry.register_pair(
        interactions(),
        claude(),
        Arc::new(convert_interactions_request_to_claude),
        ResponseTransform {
            stream: Some(Arc::new(|_, model, original, request, raw, state| {
                convert_claude_response_to_interactions(
                    model,
                    original,
                    request,
                    raw,
                    stream_state(state),
                )
            })),
            non_stream: Some(Arc::new(|_, model, original, request, raw, _| {
                convert_claude_response_to_interactions_non_stream(model, original, request, raw)
            })),
            token_count: None,
        },
    );
}

fn stream_state(state: &mut TranslationState) -> &mut ClaudeToInteractionsState {
    if state
        .as_ref()
        .is_none_or(|value| !value.is::<ClaudeToInteractionsState>())
    {
        *state = Some(Box::new(ClaudeToInteractionsState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<ClaudeToInteractionsState>())
        .expect("Claude-to-Interactions state type was initialized")
}
