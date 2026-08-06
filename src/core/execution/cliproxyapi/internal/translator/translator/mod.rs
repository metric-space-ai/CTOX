// ref: internal/translator/translator @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

#[path = "translator.rs"]
mod implementation;

pub use implementation::{
    need_convert, register, request, response, response_non_stream, Translator,
};
