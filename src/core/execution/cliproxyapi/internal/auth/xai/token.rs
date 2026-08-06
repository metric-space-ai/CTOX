// ref: internal/auth/xai/token.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: replaced_by_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::time::SystemTime;

use serde_json::Value;
use zeroize::Zeroizing;

use super::types::{AuthBundle, REFRESH_LEAD};

#[derive(Clone, Eq, PartialEq)]
pub struct SecretString(Zeroizing<String>);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Result<Self, XaiTokenError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(XaiTokenError::EmptySecret);
        }
        Ok(Self(Zeroizing::new(value.trim().to_owned())))
    }
    pub fn expose_secret(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretString([REDACTED])")
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum XaiSecretKind {
    Access,
    Refresh,
    Identity,
}

#[derive(Clone, Eq, PartialEq)]
pub struct XaiSecretHandle {
    id: String,
    kind: XaiSecretKind,
}

impl XaiSecretHandle {
    pub fn new(id: impl Into<String>, kind: XaiSecretKind) -> Result<Self, XaiTokenError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(XaiTokenError::EmptyHandle);
        }
        Ok(Self { id, kind })
    }
    pub fn kind(&self) -> XaiSecretKind {
        self.kind
    }
}

impl fmt::Debug for XaiSecretHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XaiSecretHandle")
            .field("id", &"[REDACTED]")
            .field("kind", &self.kind)
            .finish()
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct XaiCredentialHandles {
    pub access: XaiSecretHandle,
    pub refresh: XaiSecretHandle,
    pub identity: XaiSecretHandle,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct XaiStoredCredentials {
    access_token: SecretString,
    refresh_token: Option<SecretString>,
    id_token: Option<SecretString>,
}

impl XaiStoredCredentials {
    pub fn access_token(&self) -> &SecretString {
        &self.access_token
    }
    pub fn refresh_token(&self) -> Option<&SecretString> {
        self.refresh_token.as_ref()
    }
    pub fn id_token(&self) -> Option<&SecretString> {
        self.id_token.as_ref()
    }
}

pub trait XaiSecretStore: Send + Sync {
    fn load_credentials(
        &self,
        handles: &XaiCredentialHandles,
    ) -> Result<XaiStoredCredentials, XaiSecretStoreError>;
    fn store_credentials(
        &self,
        handles: &XaiCredentialHandles,
        credentials: &XaiStoredCredentials,
    ) -> Result<(), XaiSecretStoreError>;
}

/// Metadata remains inspectable, while all three credential values can only
/// cross the injected CTOX secret-store boundary. This intentionally replaces
/// upstream's plaintext JSON `SaveTokenToFile` implementation.
#[derive(Clone, PartialEq)]
pub struct TokenStorage {
    credentials: XaiStoredCredentials,
    token_type: String,
    expires_in: i64,
    expires_at: Option<SystemTime>,
    last_refresh: SystemTime,
    email: String,
    subject: String,
    base_url: String,
    redirect_uri: String,
    token_endpoint: String,
    metadata: BTreeMap<String, Value>,
}

impl TokenStorage {
    pub fn from_bundle(bundle: &AuthBundle) -> Self {
        let token = &bundle.token_data;
        Self {
            credentials: XaiStoredCredentials {
                access_token: token.access_token().clone(),
                refresh_token: token.refresh_token().cloned(),
                id_token: token.id_token().cloned(),
            },
            token_type: token.token_type().to_owned(),
            expires_in: token.expires_in(),
            expires_at: token.expires_at(),
            last_refresh: bundle.last_refresh,
            email: token.email().to_owned(),
            subject: token.subject().to_owned(),
            base_url: bundle.base_url.clone(),
            redirect_uri: bundle.redirect_uri.clone(),
            token_endpoint: bundle.token_endpoint.clone(),
            metadata: BTreeMap::new(),
        }
    }
    pub fn set_metadata(&mut self, metadata: BTreeMap<String, Value>) {
        self.metadata = metadata;
    }
    pub fn credentials(&self) -> &XaiStoredCredentials {
        &self.credentials
    }
    pub fn token_type(&self) -> &str {
        &self.token_type
    }
    pub fn expires_in(&self) -> i64 {
        self.expires_in
    }
    pub fn expires_at(&self) -> Option<SystemTime> {
        self.expires_at
    }
    pub fn last_refresh(&self) -> SystemTime {
        self.last_refresh
    }
    pub fn email(&self) -> &str {
        &self.email
    }
    pub fn subject(&self) -> &str {
        &self.subject
    }
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
    pub fn redirect_uri(&self) -> &str {
        &self.redirect_uri
    }
    pub fn token_endpoint(&self) -> &str {
        &self.token_endpoint
    }
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }
    pub fn needs_refresh_at(&self, now: SystemTime) -> bool {
        self.credentials.refresh_token.is_some()
            && self.expires_at.is_some_and(|expiry| {
                now.checked_add(REFRESH_LEAD)
                    .is_none_or(|threshold| threshold > expiry)
            })
    }
    pub fn persist_credentials(
        &self,
        store: &dyn XaiSecretStore,
        handles: &XaiCredentialHandles,
    ) -> Result<(), XaiSecretStoreError> {
        store.store_credentials(handles, &self.credentials)
    }
}

impl fmt::Debug for TokenStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenStorage")
            .field("credentials", &self.credentials)
            .field("token_type", &self.token_type)
            .field("email", &self.email)
            .field("subject", &self.subject)
            .field("metadata_keys", &self.metadata.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

pub fn credential_file_name(email: &str, subject: &str, now_millis: u128) -> String {
    for value in [email, subject] {
        let segment = sanitize_file_segment(value);
        if !segment.is_empty() {
            return format!("xai-{segment}.json");
        }
    }
    format!("xai-{now_millis}.json")
}

pub fn sanitize_file_segment(value: &str) -> String {
    value
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '@' | '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_owned()
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum XaiTokenError {
    EmptySecret,
    EmptyHandle,
}
impl fmt::Display for XaiTokenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::EmptySecret => "xAI credential is empty",
            Self::EmptyHandle => "xAI credential handle is empty",
        })
    }
}
impl std::error::Error for XaiTokenError {}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum XaiSecretStoreError {
    Missing,
    InvalidValue,
    Read,
    Write,
}
impl fmt::Display for XaiSecretStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Missing => "xAI credential is missing",
            Self::InvalidValue => "xAI credential value is invalid",
            Self::Read => "xAI credential could not be read",
            Self::Write => "xAI credential could not be written",
        })
    }
}
impl std::error::Error for XaiSecretStoreError {}
