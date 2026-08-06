// Origin: CTOX
// License: AGPL-3.0-only

mod init;
#[path = "gemini_openai-responses_request.rs"]
mod request;
#[path = "gemini_openai-responses_response.rs"]
mod response;
#[path = "signature_carrier.rs"]
mod signature_carrier;

#[cfg(test)]
mod noop_optimization_test;
#[cfg(test)]
#[path = "gemini_openai-responses_request_differential_test.rs"]
mod request_differential_test;
#[cfg(test)]
#[path = "gemini_openai-responses_request_test.rs"]
mod request_test;
#[cfg(test)]
#[path = "gemini_openai-responses_response_test.rs"]
mod response_test;
#[cfg(test)]
mod signature_carrier_test;

pub use init::register_openai_responses_gemini_request;
pub use request::convert_openai_responses_request_to_gemini;
pub use response::{
    convert_gemini_response_to_openai_responses_non_stream,
    convert_gemini_response_to_openai_responses_non_stream_with_state,
    convert_gemini_response_to_openai_responses_stream, GeminiToResponsesState,
};
