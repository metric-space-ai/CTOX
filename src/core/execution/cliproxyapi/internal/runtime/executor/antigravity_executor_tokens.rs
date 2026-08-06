// ref: internal/runtime/executor/antigravity_executor_tokens.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

//! Token counting through the injected Antigravity transport and translator.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use sha2::{Digest, Sha256};

use serde_json::Value;

use crate::sdk::translator::{Format, Registry, TranslationContext};

use super::antigravity_executor::{
    AntigravityGenerateRequest, AntigravityGenerateTransport, AntigravityGenerateTransportFailure,
    AntigravityUpstreamTarget,
};
use super::antigravity_executor_auth::{
    AntigravitySubscriptionAuth, AntigravitySubscriptionAuthError,
};
use super::antigravity_executor_request::{
    prepare_antigravity_generate_body, AntigravityRequestError,
};

pub struct AntigravityTokenCounter {
    auth: Arc<AntigravitySubscriptionAuth>,
    transport: Arc<dyn AntigravityGenerateTransport>,
    registry: Arc<Registry>,
    cancellation: TranslationContext,
    timeout: Duration,
    fingerprint_sink:
        Option<Arc<dyn super::antigravity_executor_execute::AntigravityAccessTokenFingerprintSink>>,
}

impl AntigravityTokenCounter {
    pub fn new(
        auth: Arc<AntigravitySubscriptionAuth>,
        transport: Arc<dyn AntigravityGenerateTransport>,
        registry: Arc<Registry>,
        cancellation: TranslationContext,
        timeout: Duration,
    ) -> Result<Self, AntigravityTokenCountError> {
        if timeout.is_zero() {
            return Err(AntigravityTokenCountError::InvalidTimeout);
        }
        Ok(Self {
            auth,
            transport,
            registry,
            cancellation,
            timeout,
            fingerprint_sink: None,
        })
    }

    pub fn with_access_token_fingerprint_sink(
        mut self,
        sink: Arc<dyn super::antigravity_executor_execute::AntigravityAccessTokenFingerprintSink>,
    ) -> Self {
        self.fingerprint_sink = Some(sink);
        self
    }

    fn publish_access_token_fingerprint(
        &self,
        credentials: &crate::internal::auth::antigravity::AntigravityStoredCredentials,
    ) {
        publish_access_token_fingerprint(self.fingerprint_sink.as_deref(), credentials);
    }

    pub async fn count_tokens(
        &self,
        target: &AntigravityUpstreamTarget,
        model: &str,
        source_format: &Format,
        response_format: &Format,
        payload: &[u8],
    ) -> Result<Vec<u8>, AntigravityTokenCountError> {
        if self.cancellation.is_cancelled() {
            return Err(AntigravityTokenCountError::Cancelled);
        }
        let translated = self.registry.translate_request(
            &self.cancellation,
            source_format,
            &Format::from("antigravity"),
            model,
            payload,
            false,
        );
        let mut credentials = self
            .auth
            .load()
            .await
            .map_err(AntigravityTokenCountError::Auth)?;
        self.publish_access_token_fingerprint(&credentials);
        for attempt in 1..=2 {
            let mut body =
                prepare_antigravity_generate_body(&translated, model, credentials.project_id())
                    .map_err(AntigravityTokenCountError::Request)?;
            strip_count_token_fields(&mut body)?;
            let request = AntigravityGenerateRequest::new_count_tokens(
                target,
                credentials.access_token().clone(),
                body,
            );
            let response = self
                .transport
                .execute(&request, self.timeout)
                .await
                .map_err(AntigravityTokenCountError::Transport)?;
            if response.status() == 401 && attempt == 1 {
                credentials = self
                    .auth
                    .refresh_after_status(401)
                    .await
                    .map_err(AntigravityTokenCountError::Auth)?
                    .credentials()
                    .clone();
                self.publish_access_token_fingerprint(&credentials);
                continue;
            }
            if !(200..300).contains(&response.status()) {
                return Err(AntigravityTokenCountError::Http(response.status()));
            }
            let count = serde_json::from_slice::<Value>(response.body())
                .ok()
                .and_then(|root| root.get("totalTokens").and_then(Value::as_i64))
                .ok_or(AntigravityTokenCountError::MissingTotalTokens)?;
            return Ok(self.registry.translate_token_count(
                &self.cancellation,
                &Format::from("antigravity"),
                response_format,
                count,
                response.body(),
            ));
        }
        Err(AntigravityTokenCountError::RefreshExhausted)
    }
}

fn publish_access_token_fingerprint(
    sink: Option<&dyn super::antigravity_executor_execute::AntigravityAccessTokenFingerprintSink>,
    credentials: &crate::internal::auth::antigravity::AntigravityStoredCredentials,
) {
    let Some(sink) = sink else {
        return;
    };
    let digest = Sha256::digest(credentials.access_token().expose_secret().as_bytes());
    sink.update_access_token_fingerprint(&format!("{digest:x}"));
}

fn strip_count_token_fields(body: &mut Vec<u8>) -> Result<(), AntigravityTokenCountError> {
    let mut root = serde_json::from_slice::<Value>(body)
        .map_err(|_| AntigravityTokenCountError::InvalidJson)?;
    if let Some(request) = root.get_mut("request").and_then(Value::as_object_mut) {
        request.remove("tools");
        request.remove("generationConfig");
        request.remove("generation_config");
        request.remove("safetySettings");
    }
    *body = serde_json::to_vec(&root).map_err(|_| AntigravityTokenCountError::InvalidJson)?;
    Ok(())
}

#[derive(Debug)]
pub enum AntigravityTokenCountError {
    InvalidTimeout,
    Cancelled,
    InvalidJson,
    MissingTotalTokens,
    RefreshExhausted,
    Auth(AntigravitySubscriptionAuthError),
    Request(AntigravityRequestError),
    Transport(AntigravityGenerateTransportFailure),
    Http(u16),
}
impl fmt::Display for AntigravityTokenCountError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimeout => {
                formatter.write_str("Antigravity token timeout must be non-zero")
            }
            Self::Cancelled => formatter.write_str("Antigravity token count cancelled"),
            Self::InvalidJson => formatter.write_str("Antigravity token payload is invalid"),
            Self::MissingTotalTokens => formatter.write_str("Antigravity totalTokens missing"),
            Self::RefreshExhausted => formatter.write_str("Antigravity token refresh exhausted"),
            Self::Auth(error) => write!(formatter, "Antigravity auth failed: {error}"),
            Self::Request(_) => formatter.write_str("Antigravity token request invalid"),
            Self::Transport(error) => {
                write!(formatter, "Antigravity token transport failed: {error:?}")
            }
            Self::Http(status) => write!(formatter, "Antigravity token upstream returned {status}"),
        }
    }
}
impl std::error::Error for AntigravityTokenCountError {}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::SystemTime;

    use super::*;
    use crate::internal::auth::antigravity::{AntigravityStoredCredentials, SecretString};

    #[derive(Default)]
    struct RecordingSink(Mutex<Vec<String>>);

    impl super::super::antigravity_executor_execute::AntigravityAccessTokenFingerprintSink
        for RecordingSink
    {
        fn update_access_token_fingerprint(&self, sha256: &str) {
            self.0.lock().unwrap().push(sha256.to_owned());
        }
    }

    #[test]
    fn candidate_count_tokens_publishes_only_the_access_token_sha256() {
        let credentials = AntigravityStoredCredentials::new(
            SecretString::new("candidate-access-secret").unwrap(),
            SecretString::new("candidate-refresh-secret").unwrap(),
            SystemTime::now() + Duration::from_secs(3600),
            "candidate-project",
        )
        .unwrap();
        let sink = RecordingSink::default();

        publish_access_token_fingerprint(Some(&sink), &credentials);

        let values = sink.0.lock().unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].len(), 64);
        assert!(values[0].bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_eq!(
            values[0],
            format!("{:x}", Sha256::digest(b"candidate-access-secret"))
        );
        assert!(!values[0].contains("candidate-access-secret"));
    }

    #[test]
    fn candidate_count_tokens_without_usage_sink_is_a_safe_noop() {
        let credentials = AntigravityStoredCredentials::new(
            SecretString::new("candidate-access-secret").unwrap(),
            SecretString::new("candidate-refresh-secret").unwrap(),
            SystemTime::now() + Duration::from_secs(3600),
            "candidate-project",
        )
        .unwrap();
        publish_access_token_fingerprint(None, &credentials);
    }
}
