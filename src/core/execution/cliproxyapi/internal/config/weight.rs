// ref: internal/config/weight.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::credentialweight::{normalize, CredentialWeightError};

pub use crate::internal::credentialweight::MAX_CREDENTIAL_WEIGHT;

/// Validates an explicitly configured credential weight.
///
/// The upstream Go config accepts a nil pointer as an omitted weight. CTOX's
/// typed account structs materialize that omission as the serde default `1`,
/// so `None` is retained here only for signature-level compatibility.
pub fn validate_credential_weight(weight: Option<i64>) -> Result<(), CredentialWeightError> {
    if let Some(weight) = weight {
        normalize(weight)?;
    }
    Ok(())
}
