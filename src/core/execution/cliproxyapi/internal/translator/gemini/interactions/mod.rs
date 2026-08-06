// Origin: CTOX
// License: AGPL-3.0-only

mod init;
mod interactions_gemini_common;
mod interactions_gemini_response;

#[cfg(test)]
mod interactions_gemini_common_test;
#[cfg(test)]
mod interactions_gemini_file_data_test;

pub use init::register_gemini_interactions;
pub use interactions_gemini_common::{
    convert_gemini_request_to_interactions, convert_interactions_request_to_gemini,
};
pub use interactions_gemini_response::{
    convert_gemini_response_to_interactions_non_stream,
    convert_gemini_response_to_interactions_stream,
    convert_interactions_response_to_gemini_non_stream,
    convert_interactions_response_to_gemini_stream, GeminiToInteractionsState,
    InteractionsToGeminiState,
};
