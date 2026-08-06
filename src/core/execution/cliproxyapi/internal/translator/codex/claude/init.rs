// ref: internal/translator/codex/claude/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::translator::{claude, codex, Registry, ResponseTransform, TranslationState};

use super::{
    claude_token_count, convert_claude_request_to_codex,
    convert_codex_response_to_claude_non_stream, convert_codex_response_to_claude_stream,
    deterministic_claude_message_id, CodexToClaudeStreamState,
};

pub fn register_claude_codex(registry: &Registry) {
    registry.register_pair(
        claude(),
        codex(),
        Arc::new(convert_claude_request_to_codex),
        ResponseTransform {
            stream: Some(Arc::new(|context, model, original, request, raw, state| {
                let identity = deterministic_claude_message_id(model, original, request);
                convert_codex_response_to_claude_stream(
                    context,
                    model,
                    original,
                    request,
                    raw,
                    stream_state(state, &identity),
                )
            })),
            non_stream: Some(Arc::new(|context, model, original, request, raw, _| {
                convert_codex_response_to_claude_non_stream(context, model, original, request, raw)
            })),
            token_count: Some(Arc::new(|_, count| claude_token_count(count))),
        },
    );
}

fn stream_state<'a>(
    state: &'a mut TranslationState,
    identity: &str,
) -> &'a mut CodexToClaudeStreamState {
    if state
        .as_ref()
        .is_none_or(|value| !value.is::<CodexToClaudeStreamState>())
    {
        *state = Some(Box::new(CodexToClaudeStreamState::with_identity(identity)));
    }
    state
        .as_mut()
        .and_then(|value| value.downcast_mut::<CodexToClaudeStreamState>())
        .expect("Codex-to-Claude stream state was initialized")
}
