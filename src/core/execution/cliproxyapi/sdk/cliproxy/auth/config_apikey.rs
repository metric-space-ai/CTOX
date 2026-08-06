// ref: sdk/cliproxy/auth/config_apikey.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{Auth, AuthKind, AuthSourceKind};

/// Reports whether an auth is an API-key credential synthesized from typed
/// configuration. The key stays owned by the injected auth/secret authority.
#[must_use]
pub fn is_config_api_key_auth(auth: Option<&Auth>) -> bool {
    auth.is_some_and(|auth| {
        auth.auth_kind() == Some(AuthKind::ApiKey)
            && auth.auth_source_kind() == Some(AuthSourceKind::Config)
            && auth
                .attributes
                .get("api_key")
                .is_some_and(|value| !value.trim().is_empty())
    })
}
