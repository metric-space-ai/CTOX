// ref: internal/access/config_access/provider.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

use crate::sdk::access::{
    new_invalid_credential_error, new_no_credentials_error, new_not_handled_error,
    register_provider, unregister_provider, AuthenticationFuture, AuthenticationOutcome, Provider,
    Request, Result as AccessResult, SharedProvider, ACCESS_PROVIDER_TYPE_CONFIG_API_KEY,
    DEFAULT_ACCESS_PROVIDER_NAME,
};

/// Registers the inline config API-key provider from an explicit typed key
/// snapshot. CTOX configuration ownership remains outside the proxy crate.
pub fn register(api_keys: &[String]) {
    let keys = normalize_keys(api_keys);
    if keys.is_empty() {
        unregister_provider(ACCESS_PROVIDER_TYPE_CONFIG_API_KEY);
        return;
    }
    register_provider(
        ACCESS_PROVIDER_TYPE_CONFIG_API_KEY,
        Some(ConfigApiKeyProvider::shared(
            DEFAULT_ACCESS_PROVIDER_NAME,
            &keys,
        )),
    );
}

struct ConfigApiKeyProvider {
    name: String,
    keys: Vec<Zeroizing<String>>,
}

impl ConfigApiKeyProvider {
    fn shared(name: &str, keys: &[String]) -> SharedProvider {
        let name = name.trim();
        let keys = normalize_keys(keys);
        Arc::new(Self {
            name: if name.is_empty() {
                DEFAULT_ACCESS_PROVIDER_NAME.to_owned()
            } else {
                name.to_owned()
            },
            keys: keys.iter().map(|key| Zeroizing::new(key.clone())).collect(),
        })
    }

    fn matches(&self, candidate: &str) -> bool {
        let candidate = candidate.as_bytes();
        self.keys.iter().fold(0_u8, |matched, key| {
            let equal = if key.len() == candidate.len() {
                key.as_bytes().ct_eq(candidate).unwrap_u8()
            } else {
                0
            };
            matched | equal
        }) == 1
    }
}

impl Provider for ConfigApiKeyProvider {
    fn identifier(&self) -> &str {
        &self.name
    }

    fn authenticate<'a>(&'a self, request: &'a mut Request) -> AuthenticationFuture<'a> {
        Box::pin(async move {
            if self.keys.is_empty() {
                return AuthenticationOutcome::failure(new_not_handled_error());
            }

            let authorization = header_value(request, "authorization");
            let google = header_value(request, "x-goog-api-key");
            let anthropic = header_value(request, "x-api-key");
            let (query_key, query_auth_token) = query_credentials(request.url.as_deref());
            if authorization.is_empty()
                && google.is_empty()
                && anthropic.is_empty()
                && query_key.is_empty()
                && query_auth_token.is_empty()
            {
                return AuthenticationOutcome::failure(new_no_credentials_error());
            }

            let bearer = extract_bearer_token(&authorization);
            for (candidate, source) in [
                (bearer.as_str(), "authorization"),
                (google.as_str(), "x-goog-api-key"),
                (anthropic.as_str(), "x-api-key"),
                (query_key.as_str(), "query-key"),
                (query_auth_token.as_str(), "query-auth-token"),
            ] {
                if candidate.is_empty() || !self.matches(candidate) {
                    continue;
                }
                return AuthenticationOutcome::success(Some(AccessResult {
                    provider: self.identifier().to_owned(),
                    // Upstream returns the secret itself. CTOX uses a stable
                    // opaque identity so downstream logs cannot disclose it.
                    principal: opaque_principal(candidate),
                    metadata: Some(BTreeMap::from([("source".to_owned(), source.to_owned())])),
                }));
            }

            AuthenticationOutcome::failure(new_invalid_credential_error())
        })
    }
}

fn header_value(request: &Request, wanted: &str) -> String {
    request
        .headers
        .as_ref()
        .and_then(|headers| {
            headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(wanted))
        })
        .and_then(|(_, values)| values.first())
        .cloned()
        .unwrap_or_default()
}

fn query_credentials(raw_url: Option<&str>) -> (String, String) {
    let Some(url) = raw_url.and_then(|raw| url::Url::parse(raw).ok()) else {
        return (String::new(), String::new());
    };
    let mut key = String::new();
    let mut auth_token = String::new();
    for (name, value) in url.query_pairs() {
        if name == "key" && key.is_empty() {
            key = value.into_owned();
        } else if name == "auth_token" && auth_token.is_empty() {
            auth_token = value.into_owned();
        }
    }
    (key, auth_token)
}

fn extract_bearer_token(header: &str) -> String {
    let Some((scheme, token)) = header.split_once(' ') else {
        return header.to_owned();
    };
    if !scheme.eq_ignore_ascii_case("bearer") {
        return header.to_owned();
    }
    token.trim().to_owned()
}

fn normalize_keys(keys: &[String]) -> Vec<String> {
    let mut normalized = Vec::new();
    for key in keys {
        let key = key.trim();
        if key.is_empty() || normalized.iter().any(|known| known == key) {
            continue;
        }
        normalized.push(key.to_owned());
    }
    normalized
}

fn opaque_principal(api_key: &str) -> String {
    let digest = Sha256::digest(api_key.as_bytes());
    let mut encoded = String::with_capacity(16 + 7);
    encoded.push_str("api-key:");
    for byte in &digest[..8] {
        use std::fmt::Write as _;
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdk::access::{Headers, AUTH_ERROR_CODE_INVALID_CREDENTIAL};

    #[tokio::test]
    async fn authenticates_all_upstream_carriers_without_exposing_secret() {
        let provider = ConfigApiKeyProvider::shared("", &[" secret ".to_owned()]);
        for (url, header, source) in [
            (
                "https://example.test/v1",
                Some(("Authorization", "Bearer secret")),
                "authorization",
            ),
            (
                "https://example.test/v1",
                Some(("X-Goog-Api-Key", "secret")),
                "x-goog-api-key",
            ),
            (
                "https://example.test/v1",
                Some(("x-api-key", "secret")),
                "x-api-key",
            ),
            ("https://example.test/v1?key=secret", None, "query-key"),
            (
                "https://example.test/v1?auth_token=secret",
                None,
                "query-auth-token",
            ),
        ] {
            let headers = header
                .map(|(name, value)| Headers::from([(name.to_owned(), vec![value.to_owned()])]));
            let mut request = Request {
                url: Some(url.to_owned()),
                headers,
                ..Request::default()
            };
            let result = provider
                .authenticate(&mut request)
                .await
                .result
                .expect("successful auth");
            assert_eq!(result.provider, DEFAULT_ACCESS_PROVIDER_NAME);
            assert_eq!(result.metadata.unwrap()["source"], source);
            assert!(result.principal.starts_with("api-key:"));
            assert!(!result.principal.contains("secret"));
        }
    }

    #[tokio::test]
    async fn distinguishes_missing_and_invalid_credentials() {
        let provider = ConfigApiKeyProvider::shared("inline", &["good".to_owned()]);
        let missing = provider.authenticate(&mut Request::default()).await;
        assert!(missing.error.is_some());

        let mut invalid = Request {
            headers: Some(Headers::from([(
                "Authorization".to_owned(),
                vec!["Bearer bad".to_owned()],
            )])),
            ..Request::default()
        };
        assert_eq!(
            provider
                .authenticate(&mut invalid)
                .await
                .error
                .unwrap()
                .code,
            AUTH_ERROR_CODE_INVALID_CREDENTIAL
        );
    }

    #[test]
    fn normalization_and_bearer_parsing_match_upstream() {
        assert_eq!(
            normalize_keys(&[
                " a ".to_owned(),
                "".to_owned(),
                "a".to_owned(),
                "b".to_owned()
            ]),
            ["a", "b"]
        );
        assert_eq!(extract_bearer_token("Bearer token "), "token");
        assert_eq!(extract_bearer_token("Basic token"), "Basic token");
        assert_eq!(extract_bearer_token("raw"), "raw");
    }
}
