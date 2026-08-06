// Origin: CTOX module graph for the upstream watcher package.
// License: AGPL-3.0-only

pub mod clients;
pub mod config_reload;
pub mod diff;
pub mod dispatcher;
pub mod events;
#[path = "watcher.rs"]
mod runtime;
pub mod synthesizer;

pub use runtime::*;

#[cfg(test)]
mod watcher_test;
