// Origin: CTOX
// License: AGPL-3.0-only

mod antigravity_openai_request;
mod antigravity_openai_response;
mod init;

pub use antigravity_openai_request::convert_openai_chat_request_to_antigravity;
#[cfg(test)]
pub(crate) use antigravity_openai_request::normalize_antigravity_openai_thinking_config;
pub use antigravity_openai_response::{
    convert_antigravity_response_to_openai_chat_non_stream,
    convert_antigravity_response_to_openai_chat_stream, AntigravityToChatStreamState,
};
pub use init::register_openai_chat_antigravity;

#[cfg(test)]
mod antigravity_openai_file_data_test;
#[cfg(test)]
mod antigravity_openai_request_test;
#[cfg(test)]
mod antigravity_openai_response_test;
#[cfg(test)]
mod noop_optimization_test;
