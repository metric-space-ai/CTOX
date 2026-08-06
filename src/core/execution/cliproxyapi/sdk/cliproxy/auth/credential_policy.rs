// ref: sdk/cliproxy/auth/credential_policy.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{Auth, AuthKind};

pub const CREDENTIAL_POLICY_CODEX_ALPHA_SEARCH_V1: &str = "codex_alpha_search_v1";

#[must_use]
pub fn normalize_credential_policy(policy: &str) -> Option<&'static str> {
    policy
        .trim()
        .eq_ignore_ascii_case(CREDENTIAL_POLICY_CODEX_ALPHA_SEARCH_V1)
        .then_some(CREDENTIAL_POLICY_CODEX_ALPHA_SEARCH_V1)
}

#[must_use]
pub fn credential_policy_allows(policy: &str, auth: Option<&Auth>) -> bool {
    let Some(auth) = auth else { return false };
    if normalize_credential_policy(policy).is_none()
        || !auth.provider.trim().eq_ignore_ascii_case("codex")
    {
        return false;
    }
    match auth.auth_kind() {
        Some(AuthKind::OAuth) => true,
        Some(AuthKind::ApiKey) => auth
            .attributes
            .get("codex_alpha_search")
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("true")),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_is_closed_and_respects_auth_kind() {
        assert_eq!(
            normalize_credential_policy(" CODEX_ALPHA_SEARCH_V1 "),
            Some(CREDENTIAL_POLICY_CODEX_ALPHA_SEARCH_V1)
        );
        assert_eq!(normalize_credential_policy("unknown"), None);
        assert!(!credential_policy_allows(
            CREDENTIAL_POLICY_CODEX_ALPHA_SEARCH_V1,
            None
        ));
        let mut oauth = Auth::default();
        oauth.provider = "codex".into();
        oauth.attributes.insert("auth_kind".into(), "oauth".into());
        assert!(credential_policy_allows(
            CREDENTIAL_POLICY_CODEX_ALPHA_SEARCH_V1,
            Some(&oauth)
        ));
        let mut key = Auth::default();
        key.provider = "codex".into();
        key.attributes.extend([
            ("api_key".into(), "secret-ref".into()),
            ("codex_alpha_search".into(), "true".into()),
        ]);
        assert!(credential_policy_allows(
            CREDENTIAL_POLICY_CODEX_ALPHA_SEARCH_V1,
            Some(&key)
        ));
    }
}
