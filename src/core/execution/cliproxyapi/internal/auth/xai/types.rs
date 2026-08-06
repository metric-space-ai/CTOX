// ref: internal/auth/xai/types.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use super::token::SecretString;

pub const DEFAULT_API_BASE_URL: &str = "https://api.x.ai/v1";
pub const CLI_CHAT_PROXY_BASE_URL: &str = "https://cli-chat-proxy.grok.com/v1";
pub const ISSUER: &str = "https://auth.x.ai";
pub const DISCOVERY_URL: &str = "https://auth.x.ai/.well-known/openid-configuration";
pub const CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
pub const SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
pub const DEVICE_CODE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);
pub const HTTP_CLIENT_TIMEOUT: Duration = Duration::from_secs(30);
pub const MAX_POLL_DURATION: Duration = Duration::from_secs(30 * 60);
pub const REFRESH_LEAD: Duration = Duration::from_secs(5 * 60);

#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Discovery {
    pub device_authorization_endpoint: String,
    pub token_endpoint: String,
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceCodeResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub verification_uri_complete: String,
    pub expires_in: i64,
    pub interval: i64,
    #[serde(skip)]
    pub token_endpoint: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TokenData {
    access_token: SecretString,
    refresh_token: Option<SecretString>,
    id_token: Option<SecretString>,
    token_type: String,
    expires_in: i64,
    expires_at: Option<SystemTime>,
    email: String,
    subject: String,
}

impl TokenData {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        access_token: SecretString,
        refresh_token: Option<SecretString>,
        id_token: Option<SecretString>,
        token_type: impl Into<String>,
        expires_in: i64,
        expires_at: Option<SystemTime>,
        email: impl Into<String>,
        subject: impl Into<String>,
    ) -> Self {
        Self {
            access_token,
            refresh_token,
            id_token,
            token_type: token_type.into().trim().to_owned(),
            expires_in,
            expires_at,
            email: email.into().trim().to_owned(),
            subject: subject.into().trim().to_owned(),
        }
    }

    pub fn access_token(&self) -> &SecretString {
        &self.access_token
    }
    pub fn refresh_token(&self) -> Option<&SecretString> {
        self.refresh_token.as_ref()
    }
    pub fn id_token(&self) -> Option<&SecretString> {
        self.id_token.as_ref()
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
    pub fn email(&self) -> &str {
        &self.email
    }
    pub fn subject(&self) -> &str {
        &self.subject
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AuthBundle {
    pub token_data: TokenData,
    pub last_refresh: SystemTime,
    pub base_url: String,
    pub redirect_uri: String,
    pub token_endpoint: String,
}
