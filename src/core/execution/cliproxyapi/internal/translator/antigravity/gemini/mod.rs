// Origin: CTOX
// License: AGPL-3.0-only

mod antigravity_gemini_request;
mod antigravity_gemini_response;
mod init;

#[cfg(test)]
mod antigravity_gemini_request_test;
#[cfg(test)]
mod antigravity_gemini_response_test;
#[cfg(test)]
mod noop_optimization_test;

pub use antigravity_gemini_request::convert_gemini_request_to_antigravity;
pub(crate) use antigravity_gemini_request::sanitize_antigravity_claude_gemini_request_signatures;
pub use antigravity_gemini_response::{
    convert_antigravity_response_to_gemini, convert_antigravity_response_to_gemini_non_stream,
    gemini_token_count,
};
pub use init::register_gemini_antigravity;
