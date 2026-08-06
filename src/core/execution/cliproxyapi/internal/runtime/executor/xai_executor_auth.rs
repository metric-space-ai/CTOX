// ref: internal/runtime/executor/xai_executor_auth.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::SystemTime;

use serde_json::Value;

use crate::sdk::cliproxy::auth::Auth;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct XaiRefreshTokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub id_token: Option<String>,
    pub token_type: Option<String>,
    pub expires_in: Option<u64>,
    pub expires_at: Option<String>,
    pub email: Option<String>,
    pub subject: Option<String>,
}

pub trait XaiRefreshTransport: Send + Sync {
    fn refresh<'a>(
        &'a self,
        refresh_token: &'a str,
        token_endpoint: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = Result<XaiRefreshTokens, XaiAuthError>> + Send + 'a>>;
}

pub trait XaiAuthClock: Send + Sync {
    fn now(&self) -> SystemTime;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum XaiAuthError {
    MissingAuth,
    Transport(String),
    Clock,
}

pub struct XaiSubscriptionAuth {
    transport: Arc<dyn XaiRefreshTransport>,
    clock: Arc<dyn XaiAuthClock>,
    default_base_url: String,
}

impl XaiSubscriptionAuth {
    #[must_use]
    pub fn new(
        transport: Arc<dyn XaiRefreshTransport>,
        clock: Arc<dyn XaiAuthClock>,
        default_base_url: impl Into<String>,
    ) -> Self {
        Self {
            transport,
            clock,
            default_base_url: default_base_url.into(),
        }
    }

    pub async fn refresh(&self, auth: Option<&Auth>) -> Result<Auth, XaiAuthError> {
        let mut auth = auth.cloned().ok_or(XaiAuthError::MissingAuth)?;
        let refresh_token = auth
            .metadata
            .get("refresh_token")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default();
        if refresh_token.is_empty() {
            return Ok(auth);
        }
        let endpoint = auth
            .metadata
            .get("token_endpoint")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_owned);
        let token = self
            .transport
            .refresh(refresh_token, endpoint.as_deref())
            .await?;
        auth.metadata
            .insert("type".into(), Value::String("xai".into()));
        auth.metadata
            .insert("auth_kind".into(), Value::String("oauth".into()));
        auth.metadata
            .insert("access_token".into(), Value::String(token.access_token));
        insert_optional(&mut auth, "refresh_token", token.refresh_token);
        insert_optional(&mut auth, "id_token", token.id_token);
        insert_optional(&mut auth, "token_type", token.token_type);
        if let Some(value) = token.expires_in {
            auth.metadata.insert("expires_in".into(), value.into());
        }
        insert_optional(&mut auth, "expired", token.expires_at);
        insert_optional(&mut auth, "email", token.email);
        insert_optional(&mut auth, "sub", token.subject);
        if let Some(endpoint) = endpoint {
            auth.metadata
                .insert("token_endpoint".into(), Value::String(endpoint));
        }
        let base_url = auth
            .metadata
            .get("base_url")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or(&self.default_base_url)
            .to_owned();
        auth.metadata
            .insert("base_url".into(), Value::String(base_url.clone()));
        auth.attributes.insert("auth_kind".into(), "oauth".into());
        auth.attributes.entry("base_url".into()).or_insert(base_url);
        auth.updated_at = chrono::DateTime::<chrono::Utc>::from(self.clock.now());
        Ok(auth)
    }
}

fn insert_optional(auth: &mut Auth, key: &str, value: Option<String>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        auth.metadata.insert(key.to_owned(), Value::String(value));
    }
}
