// Origin: CTOX
// License: AGPL-3.0-only

mod example_api_keys;

#[cfg(test)]
mod example_api_keys_test;

pub use example_api_keys::{
    example_api_key_warning_page_html, example_api_keys, has_example_api_keys,
};
