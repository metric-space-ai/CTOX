// Origin: CTOX
// License: AGPL-3.0-only

mod codex_openai_request;
mod codex_openai_response;
mod init;

#[cfg(test)]
mod codex_openai_request_test;
#[cfg(test)]
mod codex_openai_response_test;
#[cfg(test)]
mod noop_optimization_test;

pub use codex_openai_request::convert_openai_chat_request_to_codex;
pub use codex_openai_response::{
    convert_codex_response_to_openai_chat_non_stream, convert_codex_response_to_openai_chat_stream,
    CodexToChatStreamState,
};
pub use init::register_openai_chat_codex;
