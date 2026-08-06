// ref: sdk/cliproxy/auth/weight.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::credentialweight::{normalize, CredentialWeightError};

/// Validates the already typed weight of an auth candidate.
///
/// Upstream must inspect both string attributes and dynamically typed metadata.
/// CTOX rejects those fields at the closed runtime-config boundary and carries
/// one `i64` weight into `AccountCandidate`, so validation has one authoritative
/// source and cannot be overridden after credential selection.
pub fn validate_auth_weight(weight: i64) -> Result<(), CredentialWeightError> {
    normalize(weight).map(drop)
}
