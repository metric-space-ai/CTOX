// ref: internal/pluginstore/auth.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use base64::Engine;
use chrono::{DateTime, Utc};
use url::Url;

use crate::sdk::pluginstore::{
    ResolvedAuthConfig, AUTH_TYPE_BASIC, AUTH_TYPE_BEARER, AUTH_TYPE_GITHUB_TOKEN,
    AUTH_TYPE_HEADER, AUTH_TYPE_NONE,
};

use super::github::store_error;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UrlPolicy {
    /// Exact HTTP origins that may be contacted. HTTPS is always allowed.
    pub allow_http_origins: Vec<String>,
}

impl UrlPolicy {
    pub fn allow_http_origin(mut self, origin: impl Into<String>) -> Self {
        self.allow_http_origins.push(origin.into());
        self
    }
}

pub fn request_url_allowed(
    policy: &UrlPolicy,
    request_url: &str,
) -> crate::sdk::pluginstore::Result<Url> {
    let parsed =
        Url::parse(request_url.trim()).map_err(|_| store_error("invalid plugin store url"))?;
    if parsed.host_str().is_none() {
        return Err(store_error("invalid plugin store url"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(store_error("plugin store url must not contain credentials"));
    }
    if sensitive_query(&parsed) {
        return Err(store_error(
            "plugin store url contains sensitive query parameter",
        ));
    }
    match parsed.scheme() {
        "https" => Ok(parsed),
        "http" => {
            let origin = parsed.origin().ascii_serialization();
            if policy
                .allow_http_origins
                .iter()
                .any(|item| item.trim() == origin)
            {
                Ok(parsed)
            } else {
                Err(store_error(
                    "insecure plugin store url is not allowed by policy",
                ))
            }
        }
        _ => Err(store_error("plugin store url must use http or https")),
    }
}

pub fn apply_resolved_auth(
    headers: &mut BTreeMap<String, String>,
    auth: &[ResolvedAuthConfig],
    expires_at: Option<DateTime<Utc>>,
    request_url: &str,
    kind: &str,
    now: DateTime<Utc>,
) -> crate::sdk::pluginstore::Result<bool> {
    let Some(item) = auth
        .iter()
        .find(|item| auth_matches(item, request_url, kind))
    else {
        return Ok(false);
    };
    if expires_at.is_some_and(|expiry| now >= expiry) {
        return Err(store_error("plugin store resolved auth expired"));
    }
    match item.auth_type.trim().to_ascii_lowercase().as_str() {
        "" | AUTH_TYPE_NONE => Ok(false),
        AUTH_TYPE_BEARER | AUTH_TYPE_GITHUB_TOKEN => {
            let token = secret_text(item.token.expose(), "token")?;
            headers.insert("Authorization".into(), format!("Bearer {token}"));
            Ok(true)
        }
        AUTH_TYPE_BASIC => {
            let username = secret_text(item.username.expose(), "username")?;
            let password = secret_text(item.password.expose(), "password")?;
            let mut credential = Vec::with_capacity(username.len() + password.len() + 1);
            credential.extend_from_slice(username.as_bytes());
            credential.push(b':');
            credential.extend_from_slice(password.as_bytes());
            let encoded = base64::engine::general_purpose::STANDARD.encode(&credential);
            zeroize::Zeroize::zeroize(&mut credential);
            headers.insert("Authorization".into(), format!("Basic {encoded}"));
            Ok(true)
        }
        AUTH_TYPE_HEADER => {
            let name = item.header_name.trim();
            if name.is_empty() || name.contains(['\r', '\n', ':']) {
                return Err(store_error(
                    "plugin store resolved auth header name is invalid",
                ));
            }
            let value = secret_text(item.header_value.expose(), "header value")?;
            if value.contains(['\r', '\n']) {
                return Err(store_error(
                    "plugin store resolved auth header value is invalid",
                ));
            }
            headers.insert(name.to_owned(), value.to_owned());
            Ok(true)
        }
        other => Err(store_error(format!(
            "unsupported plugin store resolved auth type {other:?}"
        ))),
    }
}

fn auth_matches(item: &ResolvedAuthConfig, request_url: &str, kind: &str) -> bool {
    let (Ok(request), Ok(rule)) = (
        Url::parse(request_url.trim()),
        Url::parse(item.match_url.trim()),
    ) else {
        return false;
    };
    if request.scheme() != rule.scheme()
        || request.host_str().map(str::to_ascii_lowercase)
            != rule.host_str().map(str::to_ascii_lowercase)
        || request.port() != rule.port()
    {
        return false;
    }
    let rule_path = rule.path();
    let path_matches = rule_path.is_empty()
        || rule_path == "/"
        || request.path() == rule_path
        || (rule_path.ends_with('/') && request.path().starts_with(rule_path))
        || (!rule_path.ends_with('/') && request.path().starts_with(&(rule_path.to_owned() + "/")));
    path_matches
        && (item.apply_to.is_empty()
            || item
                .apply_to
                .iter()
                .any(|value| value.trim().eq_ignore_ascii_case(kind.trim())))
}

fn secret_text<'a>(bytes: &'a [u8], field: &str) -> crate::sdk::pluginstore::Result<&'a str> {
    if bytes.is_empty() {
        return Err(store_error(format!(
            "plugin store resolved auth {field} is empty"
        )));
    }
    std::str::from_utf8(bytes)
        .map_err(|_| store_error(format!("plugin store resolved auth {field} is invalid")))
}

fn sensitive_query(url: &Url) -> bool {
    url.query_pairs().any(|(key, _)| {
        matches!(
            key.trim().to_ascii_lowercase().as_str(),
            "token" | "access_token" | "access_key" | "secret" | "secret_key" | "api_key"
        )
    })
}
