// ref: sdk/cliproxy/auth/token_fingerprint.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: explicit typed observer replaces context.WithValue secret authority
// License: MIT (upstream); modifications AGPL-3.0-only

use sha2::{Digest, Sha256};

use super::Auth;

/// A request-local observation of the credential snapshot actually used by an
/// executor. The token itself stays inside the owned auth snapshot; callers
/// receive the one-way fingerprint separately and must explicitly consume the
/// observation before they can use the snapshot for a refresh retry.
pub struct AccessTokenFingerprintObservation {
    auth: Auth,
    fingerprint: String,
}

impl AccessTokenFingerprintObservation {
    #[must_use]
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    #[must_use]
    pub fn into_auth(self) -> Auth {
        self.auth
    }
}

impl std::fmt::Debug for AccessTokenFingerprintObservation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AccessTokenFingerprintObservation")
            .field("auth_id", &self.auth.id)
            .field("provider", &self.auth.provider)
            .field("has_fingerprint", &!self.fingerprint.is_empty())
            .finish()
    }
}

pub type AccessTokenFingerprintObserver =
    dyn Fn(AccessTokenFingerprintObservation) + Send + Sync + 'static;

/// Returns the normalized OAuth access-token fingerprint used to fence
/// asynchronous Home results without exposing the credential itself.
#[must_use]
pub fn access_token_sha256(auth: &Auth) -> String {
    let Some(access_token) = access_token_for_fingerprint(auth) else {
        return String::new();
    };
    let digest = Sha256::digest(access_token.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Notifies an explicitly supplied request-local observer. Unlike upstream's
/// context value, the observer cannot be recovered from ambient process state.
pub fn notify_access_token_fingerprint(
    observer: Option<&AccessTokenFingerprintObserver>,
    auth: &Auth,
) -> bool {
    let Some(observer) = observer else {
        return false;
    };
    let fingerprint = access_token_sha256(auth);
    if fingerprint.is_empty() {
        return false;
    }
    observer(AccessTokenFingerprintObservation {
        auth: auth.clone(),
        fingerprint,
    });
    true
}

fn access_token_for_fingerprint(auth: &Auth) -> Option<&str> {
    for key in ["access_token", "accessToken"] {
        if let Some(value) = auth
            .metadata
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Some(value);
        }
    }
    for key in ["token", "Token"] {
        let Some(token) = auth
            .metadata
            .get(key)
            .and_then(serde_json::Value::as_object)
        else {
            continue;
        };
        for token_key in ["access_token", "accessToken"] {
            if let Some(value) = token
                .get(token_key)
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                return Some(value);
            }
        }
    }
    None
}
