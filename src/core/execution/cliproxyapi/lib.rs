// Origin: CTOX
// License: AGPL-3.0-only

//! Rust port of CLIProxyAPI.
//!
//! Upstream-derived files retain a `ref` and `Port-Status` header. The public
//! API is intentionally small while the port is incomplete; unported scaffold
//! files are not part of the module graph.

// Mirrored command sources are compiled both as binaries and as library host
// ABI modules. Give those shared sources one stable crate-qualified path in
// both compilation contexts.
extern crate self as ctox_cliproxyapi;

pub mod internal;
pub mod protocol;
pub mod sdk;

// The upstream plugin examples are executable, bounded in-process examples in
// the Rust port. They are compiled with the test graph so every mirrored file
// remains checked without turning the C-ABI sample programs into production
// binaries.
#[cfg(test)]
#[path = "examples/plugin/mod.rs"]
mod plugin_examples;

#[cfg(test)]
#[path = "test/claude_code_compatibility_sentinel_test.rs"]
mod claude_code_compatibility_sentinel_test;

#[cfg(test)]
#[path = "test/builtin_tools_translation_test.rs"]
mod builtin_tools_translation_test;

#[cfg(test)]
#[path = "test/codex_claude_parallel_function_calls_test.rs"]
mod codex_claude_parallel_function_calls_test;

#[cfg(test)]
#[path = "test/summary_intent_translation_test.rs"]
mod summary_intent_translation_test;

#[cfg(test)]
#[path = "test/thinking_conversion_test.rs"]
mod thinking_conversion_test;

#[cfg(test)]
#[path = "test/usage_logging_test.rs"]
mod usage_logging_test;
