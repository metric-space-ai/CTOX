// Origin: CTOX module graph for the pinned Codex/Gemini translator.
// License: AGPL-3.0-only

mod codex_gemini_request;
mod codex_gemini_response;
mod init;

pub use codex_gemini_request::convert_gemini_request_to_codex;
pub use codex_gemini_response::{
    convert_codex_response_to_gemini_non_stream, convert_codex_response_to_gemini_stream,
    gemini_token_count, CodexToGeminiStreamState,
};
pub use init::register_gemini_codex;

#[cfg(test)]
mod codex_gemini_request_test;
#[cfg(test)]
mod codex_gemini_response_test;
#[cfg(test)]
mod noop_optimization_test;
