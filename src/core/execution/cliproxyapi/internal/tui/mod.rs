// Origin: CTOX module graph for the upstream TUI package.
// License: AGPL-3.0-only

pub mod auth_tab;
pub mod browser;
pub mod client;
pub mod config_tab;
pub mod dashboard;
pub mod i18n;
pub mod keys_tab;
pub mod loghook;
pub mod logs_tab;
pub mod oauth_tab;
#[path = "app.rs"]
mod runtime;
pub mod styles;

pub use runtime::*;

#[cfg(test)]
mod oauth_tab_test;
