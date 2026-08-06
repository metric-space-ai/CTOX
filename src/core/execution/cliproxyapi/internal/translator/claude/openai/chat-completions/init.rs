// ref: internal/translator/claude/openai/chat-completions/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::translator::{claude, openai, Registry, ResponseTransform, TranslationState};

use super::{
    convert_claude_response_to_openai_chat_non_stream,
    convert_claude_response_to_openai_chat_stream, convert_openai_chat_request_to_claude,
    ClaudeToChatStreamState,
};

/// Replaces Go package `init()` with explicit, independently verified activation.
pub fn register_openai_chat_claude_request(registry: &Registry) {
    registry.register_pair(
        openai(),
        claude(),
        Arc::new(convert_openai_chat_request_to_claude),
        ResponseTransform {
            stream: Some(Arc::new(|_, model, original, request, raw, state| {
                convert_claude_response_to_openai_chat_stream(
                    model,
                    original,
                    request,
                    raw,
                    claude_chat_state(state),
                )
            })),
            non_stream: Some(Arc::new(|_, _, original, request, raw, _| {
                convert_claude_response_to_openai_chat_non_stream(original, request, raw)
            })),
            token_count: None,
        },
    );
}

fn claude_chat_state(state: &mut TranslationState) -> &mut ClaudeToChatStreamState {
    let replace = state
        .as_ref()
        .is_none_or(|value| !value.is::<ClaudeToChatStreamState>());
    if replace {
        *state = Some(Box::new(ClaudeToChatStreamState::default()));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<ClaudeToChatStreamState>())
        .expect("Claude Chat stream state was initialized with the expected type")
}
