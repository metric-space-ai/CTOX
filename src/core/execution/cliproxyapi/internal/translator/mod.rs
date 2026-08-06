// Origin: CTOX
// License: AGPL-3.0-only

pub mod antigravity;
pub mod claude;
pub mod codex;
pub mod common;
#[path = "translator/mod.rs"]
pub mod facade;
pub mod gemini;
mod init;
pub mod interactions;
pub mod openai;

#[cfg(test)]
mod request_benchmark_test;
#[cfg(test)]
mod response_benchmark_test;

pub use init::register_all;
