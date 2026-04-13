//! Low-level types and internals re-exported from `engine_core`.
//!
//! Most users don't need these types directly. They're available for advanced
//! use cases like custom pipelines, device mapping, or direct engine access.
//!
//! # When to use this module
//!
//! - Building custom loaders or pipelines
//! - Accessing diagnostic/doctor utilities
//! - Using auto-tuning features
//! - Direct engine manipulation
//!
//! For typical inference tasks, use the types exported at the crate root.

pub use engine_core::*;
