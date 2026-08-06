// ref: internal/translator/antigravity/interactions/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::translator::{
    antigravity, interactions, Registry, ResponseTransform, TranslationState,
};

use super::{
    convert_antigravity_response_to_interactions,
    convert_antigravity_response_to_interactions_non_stream,
    convert_interactions_request_to_antigravity, AntigravityToInteractionsState,
};

pub fn register_interactions_antigravity(registry: &Registry) {
    registry.register_pair(
        interactions(),
        antigravity(),
        Arc::new(convert_interactions_request_to_antigravity),
        ResponseTransform {
            stream: Some(Arc::new(|_, model, original, request, raw, state| {
                convert_antigravity_response_to_interactions(
                    model,
                    original,
                    request,
                    raw,
                    stream_state(state),
                )
            })),
            non_stream: Some(Arc::new(|_, model, original, request, raw, _| {
                convert_antigravity_response_to_interactions_non_stream(
                    model, original, request, raw,
                )
            })),
            token_count: None,
        },
    );
}

fn stream_state(state: &mut TranslationState) -> &mut AntigravityToInteractionsState {
    if state
        .as_ref()
        .is_none_or(|value| !value.is::<AntigravityToInteractionsState>())
    {
        *state = Some(Box::new(AntigravityToInteractionsState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<AntigravityToInteractionsState>())
        .expect("Antigravity-to-Interactions state type was initialized")
}
