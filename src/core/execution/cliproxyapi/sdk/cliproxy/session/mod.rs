// Origin: CTOX
// License: AGPL-3.0-only

mod identity;

pub use identity::{
    caller_scope, claude_metadata_session_id, derive_id, derived_id, enrich, normalize_explicit_id,
};

#[cfg(test)]
mod identity_test;
