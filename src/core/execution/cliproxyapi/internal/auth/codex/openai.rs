// ref: internal/auth/codex/openai.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::SystemTime;

use super::token::SecretString;

/// PKCE verifier/challenge pair used by the Codex OAuth authorization flow.
/// ref: internal/auth/codex/openai.go:3-11
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PkceCodes {
    code_verifier: SecretString,
    code_challenge: String,
}

impl PkceCodes {
    pub fn new(
        code_verifier: SecretString,
        code_challenge: impl Into<String>,
    ) -> Result<Self, super::token::CodexTokenError> {
        let code_challenge = code_challenge.into();
        if code_challenge.trim().is_empty() {
            return Err(super::token::CodexTokenError::EmptyPkceChallenge);
        }
        Ok(Self {
            code_verifier,
            code_challenge,
        })
    }

    pub fn code_verifier(&self) -> &SecretString {
        &self.code_verifier
    }

    pub fn code_challenge(&self) -> &str {
        &self.code_challenge
    }
}

/// OAuth token information returned by OpenAI.
/// ref: internal/auth/codex/openai.go:13-29
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexTokenData {
    id_token: SecretString,
    access_token: SecretString,
    refresh_token: SecretString,
    account_id: String,
    email: String,
    expires_at: SystemTime,
}

impl CodexTokenData {
    pub fn new(
        id_token: SecretString,
        access_token: SecretString,
        refresh_token: SecretString,
        account_id: impl Into<String>,
        email: impl Into<String>,
        expires_at: SystemTime,
    ) -> Self {
        Self {
            id_token,
            access_token,
            refresh_token,
            account_id: account_id.into(),
            email: email.into(),
            expires_at,
        }
    }

    pub fn id_token(&self) -> &SecretString {
        &self.id_token
    }

    pub fn access_token(&self) -> &SecretString {
        &self.access_token
    }

    pub fn refresh_token(&self) -> &SecretString {
        &self.refresh_token
    }

    pub fn account_id(&self) -> &str {
        &self.account_id
    }

    pub fn email(&self) -> &str {
        &self.email
    }

    pub fn expires_at(&self) -> SystemTime {
        self.expires_at
    }

    pub fn is_expired_at(&self, now: SystemTime) -> bool {
        now >= self.expires_at
    }
}

/// Result of a complete authorization-code exchange.
/// ref: internal/auth/codex/openai.go:31-40
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexAuthBundle {
    api_key: Option<SecretString>,
    token_data: CodexTokenData,
    last_refresh: SystemTime,
}

impl CodexAuthBundle {
    pub fn new(token_data: CodexTokenData, last_refresh: SystemTime) -> Self {
        Self {
            api_key: None,
            token_data,
            last_refresh,
        }
    }

    pub fn with_api_key(mut self, api_key: Option<SecretString>) -> Self {
        self.api_key = api_key;
        self
    }

    pub fn api_key(&self) -> Option<&SecretString> {
        self.api_key.as_ref()
    }

    pub fn token_data(&self) -> &CodexTokenData {
        &self.token_data
    }

    pub fn last_refresh(&self) -> SystemTime {
        self.last_refresh
    }
}
