// ref: sdk/cliproxy/auth/classification.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::types::{attribute, metadata_string};
use super::Auth;

const ATTRIBUTE_API_KEY: &str = "api_key";
const ATTRIBUTE_AUTH_KIND: &str = "auth_kind";
const ATTRIBUTE_PATH: &str = "path";
const ATTRIBUTE_RUNTIME_ONLY: &str = "runtime_only";
const ATTRIBUTE_SOURCE: &str = "source";
const ATTRIBUTE_SOURCE_BACKEND: &str = "source_backend";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthKind {
    ApiKey,
    OAuth,
}

impl AuthKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ApiKey => "apikey",
            Self::OAuth => "oauth",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthSourceKind {
    Config,
    File,
    Git,
    Memory,
    ObjectStore,
    Postgres,
}

impl AuthSourceKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Config => "config",
            Self::File => "file",
            Self::Git => "git",
            Self::Memory => "memory",
            Self::ObjectStore => "objectstore",
            Self::Postgres => "postgres",
        }
    }
}

impl Auth {
    #[must_use]
    pub fn auth_kind(&self) -> Option<AuthKind> {
        attribute(self, ATTRIBUTE_AUTH_KIND)
            .and_then(normalize_auth_kind)
            .or_else(|| metadata_string(self, ATTRIBUTE_AUTH_KIND).and_then(normalize_auth_kind))
            .or_else(|| attribute(self, ATTRIBUTE_API_KEY).map(|_| AuthKind::ApiKey))
            .or_else(|| self.has_oauth_metadata().then_some(AuthKind::OAuth))
    }

    #[must_use]
    pub fn auth_source_kind(&self) -> Option<AuthSourceKind> {
        if attribute(self, ATTRIBUTE_RUNTIME_ONLY)
            .is_some_and(|value| value.eq_ignore_ascii_case("true"))
        {
            return Some(AuthSourceKind::Memory);
        }
        if let Some(source) =
            attribute(self, ATTRIBUTE_SOURCE_BACKEND).and_then(normalize_auth_source_kind)
        {
            return Some(source);
        }
        if let Some(source) = attribute(self, ATTRIBUTE_SOURCE) {
            if source.to_lowercase().starts_with("config:") {
                return Some(AuthSourceKind::Config);
            }
            return normalize_auth_source_kind(source).or(Some(AuthSourceKind::File));
        }
        if attribute(self, ATTRIBUTE_PATH).is_some() || !self.file_name.trim().is_empty() {
            return Some(AuthSourceKind::File);
        }
        None
    }

    fn has_oauth_metadata(&self) -> bool {
        const KEYS: &[&str] = &[
            "access_token",
            "refresh_token",
            "id_token",
            "email",
            "token_type",
            "expires_at",
            "expired",
        ];
        KEYS.iter().any(|key| metadata_string(self, key).is_some())
            || self
                .metadata
                .get("token")
                .and_then(Value::as_object)
                .is_some_and(|token| !token.is_empty())
    }
}

fn normalize_auth_kind(value: &str) -> Option<AuthKind> {
    match value.trim().to_lowercase().as_str() {
        "apikey" | "api_key" | "api-key" => Some(AuthKind::ApiKey),
        "oauth" | "oauth2" => Some(AuthKind::OAuth),
        _ => None,
    }
}

fn normalize_auth_source_kind(value: &str) -> Option<AuthSourceKind> {
    match value.trim().to_lowercase().as_str() {
        "config" => Some(AuthSourceKind::Config),
        "file" | "filesystem" => Some(AuthSourceKind::File),
        "git" => Some(AuthSourceKind::Git),
        "memory" | "runtime" | "runtime_only" => Some(AuthSourceKind::Memory),
        "objectstore" | "object-store" => Some(AuthSourceKind::ObjectStore),
        "postgres" | "postgresql" | "database" | "db" => Some(AuthSourceKind::Postgres),
        _ => None,
    }
}
