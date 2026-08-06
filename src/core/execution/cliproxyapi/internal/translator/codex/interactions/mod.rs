// Origin: CTOX
// License: AGPL-3.0-only

mod init;
mod interactions_codex_request;
mod interactions_codex_response;

#[cfg(test)]
mod interactions_codex_test;
#[cfg(test)]
mod noop_optimization_test;

pub use init::register_interactions_codex;
pub use interactions_codex_request::convert_interactions_request_to_codex;
pub use interactions_codex_response::{
    convert_codex_response_to_interactions_non_stream,
    convert_codex_response_to_interactions_stream, CodexToInteractionsState,
};
