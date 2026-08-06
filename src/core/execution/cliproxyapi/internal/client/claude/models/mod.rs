// Origin: CTOX
// License: AGPL-3.0-only

#[path = "models.rs"]
mod implementation;

pub use implementation::{
    build_response, ensure_claude_model_id_prefix, resolve_claude_model_id_prefix, ClaudeModel,
    ClaudeModelsResponse,
};

#[cfg(test)]
mod models_test;
