// Origin: CTOX
// License: AGPL-3.0-only

mod gemini_handlers;
mod interactions_handlers;

pub use gemini_handlers::{normalize_gemini_models, GeminiAction, GeminiHandlerError};
pub use interactions_handlers::{
    build_interactions_execution_request, frame_interactions_sse_chunk,
    parse_interactions_request_target, prepare_interactions_execution_target,
    InteractionsExecutionRequest, InteractionsRequestTarget,
    INTERACTIONS_AGENT_AUTH_SELECTION_MODEL,
};
