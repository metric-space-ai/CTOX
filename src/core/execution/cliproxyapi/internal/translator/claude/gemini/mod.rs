// Origin: CTOX
// License: AGPL-3.0-only

mod claude_gemini_request;
mod claude_gemini_response;
mod init;

#[cfg(test)]
mod claude_gemini_request_test;
#[cfg(test)]
mod claude_gemini_response_test;
#[cfg(test)]
mod noop_optimization_test;

pub use claude_gemini_request::{
    convert_gemini_request_to_claude, lowercase_claude_tool_schema_types,
    normalize_claude_tool_schema,
};
pub use claude_gemini_response::{
    convert_claude_response_to_gemini, convert_claude_response_to_gemini_non_stream,
    gemini_token_count, ClaudeToGeminiState,
};
pub use init::register_gemini_claude;
