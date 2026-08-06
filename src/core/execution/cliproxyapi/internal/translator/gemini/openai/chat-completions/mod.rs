// Origin: CTOX
// License: AGPL-3.0-only

mod gemini_openai_request;
mod gemini_openai_response;
mod init;

#[cfg(test)]
#[path = "gemini_openai_file_data_test.rs"]
mod gemini_openai_file_data_test;
#[cfg(test)]
#[path = "gemini_openai_request_test.rs"]
mod gemini_openai_request_test;
#[cfg(test)]
#[path = "gemini_openai_response_test.rs"]
mod gemini_openai_response_test;
#[cfg(test)]
#[path = "gemini_openai_signature_test.rs"]
mod gemini_openai_signature_test;
#[cfg(test)]
#[path = "noop_optimization_test.rs"]
mod noop_optimization_test;

pub use gemini_openai_request::convert_openai_chat_request_to_gemini;
pub use gemini_openai_response::{
    convert_gemini_response_to_openai_chat_non_stream,
    convert_gemini_response_to_openai_chat_non_stream_with_state,
    convert_gemini_response_to_openai_chat_stream, GeminiToChatStreamState,
};
pub use init::register_openai_chat_gemini;
