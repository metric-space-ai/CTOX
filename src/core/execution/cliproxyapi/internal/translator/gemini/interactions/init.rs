// ref: internal/translator/gemini/interactions/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::translator::{gemini, interactions, Registry, ResponseTransform, TranslationState};

use super::*;

pub fn register_gemini_interactions(registry: &Registry) {
    registry.register_pair(
        interactions(),
        gemini(),
        Arc::new(convert_interactions_request_to_gemini),
        ResponseTransform {
            stream: Some(Arc::new(|_, model, original, request, raw, state| {
                convert_gemini_response_to_interactions_stream(
                    model,
                    original,
                    request,
                    raw,
                    gemini_state(state),
                )
            })),
            non_stream: Some(Arc::new(|_, model, original, request, raw, _| {
                convert_gemini_response_to_interactions_non_stream(model, original, request, raw)
            })),
            token_count: None,
        },
    );
    registry.register_pair(
        gemini(),
        interactions(),
        Arc::new(convert_gemini_request_to_interactions),
        ResponseTransform {
            stream: Some(Arc::new(|_, model, _, _, raw, state| {
                convert_interactions_response_to_gemini_stream(
                    model,
                    raw,
                    interactions_state(state),
                )
            })),
            non_stream: Some(Arc::new(|_, model, _, _, raw, _| {
                convert_interactions_response_to_gemini_non_stream(model, raw)
            })),
            token_count: None,
        },
    );
}

fn gemini_state(state: &mut TranslationState) -> &mut GeminiToInteractionsState {
    if state
        .as_ref()
        .is_none_or(|value| !value.is::<GeminiToInteractionsState>())
    {
        *state = Some(Box::new(GeminiToInteractionsState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<GeminiToInteractionsState>())
        .expect("Gemini-to-Interactions state type was initialized")
}

fn interactions_state(state: &mut TranslationState) -> &mut InteractionsToGeminiState {
    if state
        .as_ref()
        .is_none_or(|value| !value.is::<InteractionsToGeminiState>())
    {
        *state = Some(Box::new(InteractionsToGeminiState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<InteractionsToGeminiState>())
        .expect("Interactions-to-Gemini state type was initialized")
}
