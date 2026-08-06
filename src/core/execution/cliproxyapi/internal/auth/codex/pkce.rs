// ref: internal/auth/codex/pkce.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use sha2::{Digest, Sha256};

use super::openai::PkceCodes;
use super::token::{CodexTokenError, SecretString};

/// Generates the upstream-compatible 96-byte/128-character PKCE verifier.
/// ref: internal/auth/codex/pkce.go:17-57
pub fn generate_pkce_codes() -> Result<PkceCodes, CodexTokenError> {
    let mut random = [0_u8; 96];
    getrandom::fill(&mut random).map_err(|_| CodexTokenError::Randomness)?;
    let verifier = URL_SAFE_NO_PAD.encode(random);
    let challenge = generate_code_challenge(&verifier);
    PkceCodes::new(SecretString::new(verifier)?, challenge)
}

pub fn generate_code_challenge(code_verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(code_verifier.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_matches_rfc_length_and_s256_challenge() {
        let codes = generate_pkce_codes().unwrap();
        assert_eq!(codes.code_verifier().expose_secret().len(), 128);
        assert_eq!(
            codes.code_challenge(),
            generate_code_challenge(codes.code_verifier().expose_secret())
        );
        assert_eq!(
            generate_code_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }
}
