// ref: internal/translator/claude/interactions @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

mod init;
mod interactions_claude_request;
mod interactions_claude_response;

#[cfg(test)]
mod interactions_claude_test;

pub use init::register_interactions_claude;
pub use interactions_claude_request::convert_interactions_request_to_claude;
pub use interactions_claude_response::{
    convert_claude_response_to_interactions, convert_claude_response_to_interactions_non_stream,
    ClaudeToInteractionsState,
};
