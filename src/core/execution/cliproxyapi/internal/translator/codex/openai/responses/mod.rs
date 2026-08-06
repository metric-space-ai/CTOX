// Origin: CTOX
// License: AGPL-3.0-only

mod init;

#[path = "codex_openai-responses_request.rs"]
mod request;
#[cfg(test)]
#[path = "codex_openai-responses_request_test.rs"]
mod request_test;
#[path = "codex_openai-responses_response.rs"]
mod response;
#[cfg(test)]
#[path = "codex_openai-responses_response_test.rs"]
mod response_test;

pub use init::register_openai_responses_codex;
pub use request::convert_openai_responses_request_to_codex;
pub use response::{
    convert_codex_response_to_openai_responses,
    convert_codex_response_to_openai_responses_non_stream,
};
