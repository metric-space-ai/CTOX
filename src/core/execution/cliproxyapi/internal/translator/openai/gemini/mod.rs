// Origin: CTOX
// License: AGPL-3.0-only
mod init;
mod openai_gemini_request;
mod openai_gemini_response;

#[cfg(test)]
mod openai_gemini_request_test;
#[cfg(test)]
mod openai_gemini_response_test;

pub use init::register_openai_gemini;
pub use openai_gemini_request::convert_gemini_request_to_openai;
pub use openai_gemini_response::{
    convert_openai_response_to_gemini_non_stream,
    convert_openai_response_to_gemini_non_stream_with_context,
    convert_openai_response_to_gemini_stream,
    convert_openai_response_to_gemini_stream_with_context, gemini_token_count, OpenAiToGeminiState,
};
