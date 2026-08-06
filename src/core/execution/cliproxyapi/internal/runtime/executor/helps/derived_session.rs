// ref: internal/runtime/executor/helps/derived_session.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::sdk::cliproxy::executor::ExecutionMetadata;
use crate::sdk::cliproxy::session::derived_id;

/// Returns the first context-derived session identity in metadata order.
#[must_use]
pub fn derived_session_id(metadata_sets: &[&ExecutionMetadata]) -> String {
    metadata_sets
        .iter()
        .map(|metadata| derived_id(metadata))
        .find(|identity| !identity.is_empty())
        .unwrap_or_default()
}

/// Maps a derived session identity to a provider-scoped stable UUID.
#[must_use]
pub fn derived_session_uuid(provider: &str, metadata_sets: &[&ExecutionMetadata]) -> String {
    stable_provider_session_uuid(
        provider,
        "derived-session",
        &derived_session_id(metadata_sets),
    )
}

/// Prefers a long-lived execution session and falls back to the derived identity.
#[must_use]
pub fn provider_session_uuid(provider: &str, metadata_sets: &[&ExecutionMetadata]) -> String {
    if let Some(execution_id) = metadata_sets.iter().find_map(|metadata| {
        metadata
            .execution_session_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }) {
        return stable_provider_session_uuid(provider, "execution-session", execution_id);
    }
    derived_session_uuid(provider, metadata_sets)
}

fn stable_provider_session_uuid(provider: &str, kind: &str, identity_value: &str) -> String {
    let provider = provider.trim().to_ascii_lowercase();
    let identity_value = identity_value.trim();
    if provider.is_empty() || identity_value.is_empty() {
        return String::new();
    }
    let identity = ["cli-proxy-api", provider.as_str(), kind, identity_value].join("\0");
    Uuid::new_v5(&Uuid::NAMESPACE_OID, identity.as_bytes()).to_string()
}

/// Maps a derived session identity to Antigravity's negative decimal format.
#[must_use]
pub fn derived_antigravity_session_id(metadata_sets: &[&ExecutionMetadata]) -> String {
    let derived_id = derived_session_id(metadata_sets);
    if derived_id.is_empty() {
        return String::new();
    }
    let digest = Sha256::digest(
        [
            b"cli-proxy-api:antigravity:derived-session\0".as_slice(),
            derived_id.as_bytes(),
        ]
        .concat(),
    );
    let value =
        u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 prefix")) & 0x7fff_ffff_ffff_ffff;
    format!("-{value}")
}
