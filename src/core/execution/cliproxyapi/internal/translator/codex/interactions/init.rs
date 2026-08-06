// ref: internal/translator/codex/interactions/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::translator::{codex, interactions, Registry, ResponseTransform, TranslationState};

use super::{
    convert_codex_response_to_interactions_non_stream,
    convert_codex_response_to_interactions_stream, convert_interactions_request_to_codex,
    CodexToInteractionsState,
};

pub fn register_interactions_codex(registry: &Registry) {
    registry.register_pair(
        interactions(),
        codex(),
        Arc::new(convert_interactions_request_to_codex),
        ResponseTransform {
            stream: Some(Arc::new(|context, model, original, request, raw, state| {
                convert_codex_response_to_interactions_stream(
                    context,
                    model,
                    original,
                    request,
                    raw,
                    stream_state(state),
                )
            })),
            non_stream: Some(Arc::new(|context, model, original, request, raw, _| {
                convert_codex_response_to_interactions_non_stream(
                    context, model, original, request, raw,
                )
            })),
            token_count: None,
        },
    );
}

fn stream_state(state: &mut TranslationState) -> &mut CodexToInteractionsState {
    if state
        .as_ref()
        .is_none_or(|value| !value.is::<CodexToInteractionsState>())
    {
        *state = Some(Box::new(CodexToInteractionsState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<CodexToInteractionsState>())
        .expect("Codex-to-Interactions state was initialized")
}
