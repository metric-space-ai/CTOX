// Origin: CTOX
// License: AGPL-3.0-only

mod init;
mod interactions_antigravity_request;
mod interactions_antigravity_response;

pub use init::register_interactions_antigravity;
pub use interactions_antigravity_request::convert_interactions_request_to_antigravity;
#[cfg(test)]
pub(crate) use interactions_antigravity_request::rewrite_interactions_function_names;
pub use interactions_antigravity_response::{
    convert_antigravity_response_to_interactions,
    convert_antigravity_response_to_interactions_non_stream, AntigravityToInteractionsState,
};

#[cfg(test)]
mod interactions_antigravity_file_data_test;
#[cfg(test)]
mod interactions_antigravity_test;
#[cfg(test)]
mod noop_optimization_test;
