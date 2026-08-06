// Origin: CTOX
// License: AGPL-3.0-only

mod init;
#[path = "antigravity_openai-responses_request.rs"]
mod request;
#[path = "antigravity_openai-responses_response.rs"]
mod response;

pub use init::register_openai_responses_antigravity;
pub use request::convert_openai_responses_request_to_antigravity;
pub use response::{
    convert_antigravity_response_to_openai_responses_non_stream,
    convert_antigravity_response_to_openai_responses_non_stream_with_state,
    convert_antigravity_response_to_openai_responses_stream, AntigravityToResponsesState,
};

#[cfg(test)]
#[path = "antigravity_openai-responses_request_test.rs"]
mod antigravity_openai_responses_request_test;
