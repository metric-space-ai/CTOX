// ref: sdk/cliproxy/auth/store.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

use super::Auth;

/// Authoritative persistence boundary for provider-neutral Auth records.
///
/// Implementations are supplied by the CTOX host. This crate never infers a
/// filesystem backend from `Auth.file_name`; subscription credentials remain
/// under the existing typed secret-store owners.
pub trait AuthStore: Send + Sync {
    fn list(&self) -> Result<Vec<Auth>, AuthStoreError>;
    fn save(&self, auth: &Auth) -> Result<String, AuthStoreError>;
    fn delete(&self, id: &str) -> Result<(), AuthStoreError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthStoreError {
    Read,
    Write,
    Delete,
    InvalidRecord,
}

impl fmt::Display for AuthStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Read => "auth store read failed",
            Self::Write => "auth store write failed",
            Self::Delete => "auth store delete failed",
            Self::InvalidRecord => "invalid auth store record",
        })
    }
}

impl std::error::Error for AuthStoreError {}
