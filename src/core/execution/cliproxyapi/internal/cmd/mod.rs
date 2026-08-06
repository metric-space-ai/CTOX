// Origin: CTOX module graph for the upstream command package.
// License: AGPL-3.0-only

pub mod anthropic_login;
pub mod antigravity_login;
pub mod auth_manager;
pub mod kimi_login;
pub mod login_prompt;
pub mod openai_device_login;
pub mod openai_login;
pub mod run;
pub mod vertex_import;
pub mod xai_login;

// The upstream utility binaries share these typed cores with the outer CTOX
// host. Keeping the source at its mirrored command path preserves upstream
// reviewability while making the capability boundary consumable as a library.
#[path = "../../cmd/fetch_antigravity_models/main.rs"]
pub mod fetch_antigravity_models;
#[path = "../../cmd/fetch_codex_models/main.rs"]
pub mod fetch_codex_models;
#[path = "../../cmd/server/main.rs"]
pub mod server;

pub use auth_manager::*;
pub use openai_login::*;

#[cfg(test)]
mod supplemental_test;
