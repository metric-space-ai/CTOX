// Origin: CTOX
// License: AGPL-3.0-only

#[path = "htmlsanitize.rs"]
mod core;

#[cfg(test)]
mod htmlsanitize_test;

pub use core::{
    is_json_content_type, json_body, json_body_if_likely, json_value, looks_like_json, string,
    strings,
};
