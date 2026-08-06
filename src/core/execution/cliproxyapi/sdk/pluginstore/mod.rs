// Origin: CTOX
// License: AGPL-3.0-only

#[path = "pluginstore.rs"]
mod facade;

pub use facade::*;

#[cfg(test)]
#[path = "pluginstore_test.rs"]
mod pluginstore_test;
