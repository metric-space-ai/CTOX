// ref: internal/translator/openai/interactions/chat-completions @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Local module wiring for the OpenAI chat-completions <-> Interactions
//! translation pair. Mirrors the upstream Go package layout one file per
//! production source plus a separate test mirror per upstream `*_test.go`.

mod init;
mod interactions_openai_request;
mod interactions_openai_response;
mod openai_interactions_request;
mod openai_interactions_response;

#[cfg(test)]
mod interactions_openai_request_test;
#[cfg(test)]
mod interactions_openai_response_test;
#[cfg(test)]
mod openai_interactions_file_data_test;

pub use init::register_openai_chat_interactions;
pub use interactions_openai_request::convert_interactions_request_to_openai;
pub use interactions_openai_response::{
    convert_openai_response_to_interactions, convert_openai_response_to_interactions_non_stream,
    OpenAIToInteractionsStreamState,
};
pub use openai_interactions_request::convert_openai_request_to_interactions;
pub use openai_interactions_response::{
    convert_interactions_response_to_openai, convert_interactions_response_to_openai_non_stream,
    InteractionsToOpenAIChatStreamState,
};
