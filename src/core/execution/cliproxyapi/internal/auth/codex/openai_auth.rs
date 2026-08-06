// ref: internal/auth/codex/openai_auth.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::watch;
use zeroize::Zeroizing;

use super::jwt_parser::parse_jwt_token;
use super::openai::{CodexAuthBundle, CodexTokenData, PkceCodes};
use super::token::{CodexStoredCredentials, CodexTokenError, CodexTokenStorage, SecretString};

// ref: internal/auth/codex/openai_auth.go:24-31
pub const AUTH_URL: &str = "https://auth.openai.com/oauth/authorize";
pub const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
pub const EXCHANGE_TIMEOUT: Duration = Duration::from_secs(30);
pub const REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

pub fn generate_auth_url(state: &str, pkce: &PkceCodes) -> String {
    let query = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("client_id", CLIENT_ID)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", "openid email profile offline_access")
        .append_pair("state", state)
        .append_pair("code_challenge", pkce.code_challenge())
        .append_pair("code_challenge_method", "S256")
        .append_pair("prompt", "login")
        .append_pair("id_token_add_organizations", "true")
        .append_pair("codex_cli_simplified_flow", "true")
        .finish();
    format!("{AUTH_URL}?{query}")
}

#[derive(Clone)]
pub struct CodexExchangeRequest {
    code: SecretString,
    redirect_uri: String,
    code_verifier: SecretString,
}

impl CodexExchangeRequest {
    pub fn new(
        code: &SecretString,
        redirect_uri: &str,
        pkce: &PkceCodes,
    ) -> Result<Self, CodexExchangeError> {
        let redirect_uri = redirect_uri.trim();
        if redirect_uri.is_empty() {
            return Err(CodexExchangeError::MissingRedirectUri);
        }
        Ok(Self {
            code: code.clone(),
            redirect_uri: redirect_uri.to_owned(),
            code_verifier: pkce.code_verifier().clone(),
        })
    }

    pub fn form_body(&self) -> Zeroizing<Vec<u8>> {
        Zeroizing::new(
            url::form_urlencoded::Serializer::new(String::new())
                .append_pair("grant_type", "authorization_code")
                .append_pair("client_id", CLIENT_ID)
                .append_pair("code", self.code.expose_secret())
                .append_pair("redirect_uri", &self.redirect_uri)
                .append_pair("code_verifier", self.code_verifier.expose_secret())
                .finish()
                .into_bytes(),
        )
    }

    pub fn redirect_uri(&self) -> &str {
        &self.redirect_uri
    }
}

impl fmt::Debug for CodexExchangeRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodexExchangeRequest")
            .field("grant_type", &"authorization_code")
            .field("client_id", &CLIENT_ID)
            .field("code", &"[REDACTED]")
            .field("redirect_uri", &self.redirect_uri)
            .field("code_verifier", &"[REDACTED]")
            .finish()
    }
}

pub struct CodexExchangeHttpResponse {
    status: u16,
    body: Zeroizing<Vec<u8>>,
}

impl CodexExchangeHttpResponse {
    pub fn new(status: u16, body: Vec<u8>) -> Self {
        Self {
            status,
            body: Zeroizing::new(body),
        }
    }
}

impl fmt::Debug for CodexExchangeHttpResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodexExchangeHttpResponse")
            .field("status", &self.status)
            .field("body", &"[REDACTED]")
            .finish()
    }
}

pub trait CodexCodeExchangeTransport: Send + Sync {
    fn exchange<'a>(
        &'a self,
        request: &'a CodexExchangeRequest,
        timeout: Duration,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<CodexExchangeHttpResponse, CodexRefreshTransportFailure>>
                + Send
                + 'a,
        >,
    >;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodexExchangeError {
    MissingRedirectUri,
    Transport(CodexRefreshTransportFailure),
    Http { status: u16 },
    InvalidResponse,
    Token(CodexTokenError),
}

impl fmt::Display for CodexExchangeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRedirectUri => formatter.write_str("Codex redirect URI is required"),
            Self::Transport(kind) => {
                write!(formatter, "Codex token exchange transport failed: {kind:?}")
            }
            Self::Http { status } => write!(
                formatter,
                "Codex token exchange failed with status {status}"
            ),
            Self::InvalidResponse => {
                formatter.write_str("Codex token exchange response is invalid")
            }
            Self::Token(error) => write!(
                formatter,
                "Codex token exchange returned invalid credentials: {error}"
            ),
        }
    }
}

impl std::error::Error for CodexExchangeError {}

impl From<CodexTokenError> for CodexExchangeError {
    fn from(value: CodexTokenError) -> Self {
        Self::Token(value)
    }
}

pub struct CodexAuth<T, C = SystemRefreshClock> {
    transport: T,
    clock: C,
    refresh: CodexRefreshCoordinator,
}

impl<T> CodexAuth<T, SystemRefreshClock> {
    pub fn new(transport: T) -> Self {
        Self::with_clock(transport, SystemRefreshClock)
    }
}

impl<T, C> CodexAuth<T, C> {
    pub fn with_clock(transport: T, clock: C) -> Self {
        Self {
            transport,
            clock,
            refresh: CodexRefreshCoordinator::default(),
        }
    }

    pub fn transport(&self) -> &T {
        &self.transport
    }

    pub fn generate_auth_url(&self, state: &str, pkce: &PkceCodes) -> String {
        generate_auth_url(state, pkce)
    }

    pub async fn exchange_code_for_tokens(
        &self,
        code: &SecretString,
        pkce: &PkceCodes,
    ) -> Result<CodexAuthBundle, CodexExchangeError>
    where
        T: CodexCodeExchangeTransport,
        C: RefreshClock,
    {
        self.exchange_code_for_tokens_with_redirect(code, REDIRECT_URI, pkce)
            .await
    }

    pub async fn exchange_code_for_tokens_with_redirect(
        &self,
        code: &SecretString,
        redirect_uri: &str,
        pkce: &PkceCodes,
    ) -> Result<CodexAuthBundle, CodexExchangeError>
    where
        T: CodexCodeExchangeTransport,
        C: RefreshClock,
    {
        let request = CodexExchangeRequest::new(code, redirect_uri, pkce)?;
        let response = self
            .transport
            .exchange(&request, EXCHANGE_TIMEOUT)
            .await
            .map_err(CodexExchangeError::Transport)?;
        accept_exchange_response(response, self.clock.now())
    }

    pub async fn refresh_tokens(
        &self,
        current: CodexStoredCredentials,
    ) -> Result<CodexTokenData, CodexRefreshError>
    where
        T: CodexRefreshTransport,
        C: RefreshClock,
    {
        self.refresh
            .refresh(&self.transport, &self.clock, current, 1)
            .await
    }

    pub async fn refresh_tokens_with_retry(
        &self,
        current: CodexStoredCredentials,
        max_attempts: usize,
    ) -> Result<CodexTokenData, CodexRefreshError>
    where
        T: CodexRefreshTransport,
        C: RefreshClock,
    {
        self.refresh
            .refresh(&self.transport, &self.clock, current, max_attempts)
            .await
    }

    pub fn create_token_storage(&self, bundle: &CodexAuthBundle) -> CodexTokenStorage {
        CodexTokenStorage::from_token_data(bundle.token_data(), bundle.last_refresh())
    }

    pub fn update_token_storage(&self, storage: &mut CodexTokenStorage, token: &CodexTokenData)
    where
        C: RefreshClock,
    {
        storage.update_from_token_data(token, self.clock.now());
    }
}

fn accept_exchange_response(
    response: CodexExchangeHttpResponse,
    issued_at: SystemTime,
) -> Result<CodexAuthBundle, CodexExchangeError> {
    if response.status != 200 {
        return Err(CodexExchangeError::Http {
            status: response.status,
        });
    }
    #[derive(Deserialize)]
    struct WireResponse {
        access_token: String,
        refresh_token: String,
        id_token: String,
        #[serde(default)]
        api_key: String,
        expires_in: i64,
    }
    let wire: WireResponse =
        serde_json::from_slice(&response.body).map_err(|_| CodexExchangeError::InvalidResponse)?;
    let id_token = SecretString::new(wire.id_token)?;
    let access_token = SecretString::new(wire.access_token)?;
    let refresh_token = SecretString::new(wire.refresh_token)?;
    let expires_at = checked_expiry(issued_at, wire.expires_in)?;
    let claims = parse_jwt_token(id_token.expose_secret()).ok();
    let token = CodexTokenData::new(
        id_token,
        access_token,
        refresh_token,
        claims
            .as_ref()
            .map(|value| value.account_id())
            .unwrap_or_default(),
        claims
            .as_ref()
            .map(|value| value.user_email())
            .unwrap_or_default(),
        expires_at,
    );
    let api_key = if wire.api_key.trim().is_empty() {
        None
    } else {
        Some(SecretString::new(wire.api_key)?)
    };
    Ok(CodexAuthBundle::new(token, issued_at).with_api_key(api_key))
}

fn checked_expiry(issued_at: SystemTime, expires_in: i64) -> Result<SystemTime, CodexTokenError> {
    let value = if expires_in >= 0 {
        issued_at.checked_add(Duration::from_secs(expires_in as u64))
    } else {
        issued_at.checked_sub(Duration::from_secs(expires_in.unsigned_abs()))
    };
    value.ok_or(CodexTokenError::ExpiryOverflow)
}

#[derive(Clone)]
pub struct CodexRefreshRequest {
    refresh_token: SecretString,
}

impl CodexRefreshRequest {
    pub fn new(refresh_token: SecretString) -> Self {
        Self { refresh_token }
    }

    pub fn refresh_token(&self) -> &SecretString {
        &self.refresh_token
    }

    pub fn form_body(&self) -> Zeroizing<Vec<u8>> {
        let encoded = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("client_id", CLIENT_ID)
            .append_pair("grant_type", "refresh_token")
            .append_pair("refresh_token", self.refresh_token.expose_secret())
            .append_pair("scope", "openid profile email")
            .finish();
        Zeroizing::new(encoded.into_bytes())
    }
}

impl fmt::Debug for CodexRefreshRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodexRefreshRequest")
            .field("client_id", &CLIENT_ID)
            .field("grant_type", &"refresh_token")
            .field("refresh_token", &"[REDACTED]")
            .finish()
    }
}

pub struct CodexRefreshHttpResponse {
    status: u16,
    body: Zeroizing<Vec<u8>>,
}

impl CodexRefreshHttpResponse {
    pub fn new(status: u16, body: Vec<u8>) -> Self {
        Self {
            status,
            body: Zeroizing::new(body),
        }
    }
}

impl fmt::Debug for CodexRefreshHttpResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodexRefreshHttpResponse")
            .field("status", &self.status)
            .field("body", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexRefreshTransportFailure {
    Timeout,
    Connect,
    Protocol,
    Cancelled,
}

pub trait CodexRefreshTransport: Send + Sync {
    fn execute<'a>(
        &'a self,
        request: &'a CodexRefreshRequest,
        timeout: Duration,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<CodexRefreshHttpResponse, CodexRefreshTransportFailure>>
                + Send
                + 'a,
        >,
    >;
}

pub trait RefreshClock: Send + Sync {
    fn now(&self) -> SystemTime;
    fn sleep(
        &self,
        duration: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), CodexRefreshTransportFailure>> + Send + '_>>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SystemRefreshClock;

impl RefreshClock for SystemRefreshClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }

    fn sleep(
        &self,
        duration: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), CodexRefreshTransportFailure>> + Send + '_>> {
        Box::pin(async move {
            tokio::time::sleep(duration).await;
            Ok(())
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodexRefreshError {
    Transport(CodexRefreshTransportFailure),
    Http { status: u16, retryable: bool },
    InvalidResponse,
    Token(CodexTokenError),
    InvalidSingleFlightResult,
}

impl CodexRefreshError {
    pub fn retryable(&self) -> bool {
        match self {
            Self::Transport(kind) => *kind != CodexRefreshTransportFailure::Cancelled,
            Self::Http { retryable, .. } => *retryable,
            Self::InvalidResponse | Self::Token(_) | Self::InvalidSingleFlightResult => false,
        }
    }
}

impl fmt::Display for CodexRefreshError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(kind) => write!(formatter, "Codex refresh transport failed: {kind:?}"),
            Self::Http { status, .. } => {
                write!(formatter, "Codex token refresh failed with status {status}")
            }
            Self::InvalidResponse => formatter.write_str("Codex token response is invalid"),
            Self::Token(error) => write!(formatter, "Codex token is invalid: {error}"),
            Self::InvalidSingleFlightResult => {
                formatter.write_str("Codex singleflight result is invalid")
            }
        }
    }
}

impl std::error::Error for CodexRefreshError {}

impl From<CodexTokenError> for CodexRefreshError {
    fn from(value: CodexTokenError) -> Self {
        Self::Token(value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RefreshFingerprint([u8; 32]);

impl RefreshFingerprint {
    fn for_token(token: &SecretString) -> Self {
        Self(Sha256::digest(token.expose_secret().as_bytes()).into())
    }
}

impl fmt::Debug for RefreshFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RefreshFingerprint([REDACTED])")
    }
}

type RefreshResult = Result<CodexTokenData, CodexRefreshError>;
type FlightReceiver = watch::Receiver<Option<RefreshResult>>;

/// Cross-instance refresh singleflight keyed by a one-way token fingerprint.
/// ref: internal/auth/codex/openai_auth.go:168-252
#[derive(Default)]
pub struct CodexRefreshCoordinator {
    flights: Mutex<HashMap<RefreshFingerprint, FlightReceiver>>,
}

impl CodexRefreshCoordinator {
    pub async fn refresh<T, C>(
        &self,
        transport: &T,
        clock: &C,
        current: CodexStoredCredentials,
        max_attempts: usize,
    ) -> RefreshResult
    where
        T: CodexRefreshTransport + ?Sized,
        C: RefreshClock + ?Sized,
    {
        let fingerprint = RefreshFingerprint::for_token(current.refresh_token());
        let (mut receiver, leader) = {
            let mut flights = lock_recover(&self.flights);
            if let Some(flight) = flights.get(&fingerprint) {
                (flight.clone(), None)
            } else {
                let (sender, receiver) = watch::channel(None);
                flights.insert(fingerprint, receiver.clone());
                (receiver, Some(sender))
            }
        };

        let Some(sender) = leader else {
            return wait_for_flight(&mut receiver).await;
        };
        let result = self
            .refresh_with_retry(transport, clock, &current, max_attempts)
            .await;
        if sender.send(Some(result.clone())).is_err() {
            lock_recover(&self.flights).remove(&fingerprint);
            return Err(CodexRefreshError::InvalidSingleFlightResult);
        }
        lock_recover(&self.flights).remove(&fingerprint);
        result
    }

    async fn refresh_with_retry<T, C>(
        &self,
        transport: &T,
        clock: &C,
        current: &CodexStoredCredentials,
        max_attempts: usize,
    ) -> RefreshResult
    where
        T: CodexRefreshTransport + ?Sized,
        C: RefreshClock + ?Sized,
    {
        let request = CodexRefreshRequest::new(current.refresh_token().clone());
        let mut last_error = None;
        for attempt in 0..max_attempts.max(1) {
            if attempt > 0 {
                clock
                    .sleep(Duration::from_secs(attempt as u64))
                    .await
                    .map_err(CodexRefreshError::Transport)?;
            }
            match transport.execute(&request, REFRESH_TIMEOUT).await {
                Ok(response) => match accept_response(response, current, clock.now()) {
                    Ok(token) => return Ok(token),
                    Err(error) if error.retryable() => last_error = Some(error),
                    Err(error) => return Err(error),
                },
                Err(kind) => {
                    let error = CodexRefreshError::Transport(kind);
                    if !error.retryable() {
                        return Err(error);
                    }
                    last_error = Some(error);
                }
            }
        }
        Err(last_error.unwrap_or(CodexRefreshError::InvalidSingleFlightResult))
    }
}

async fn wait_for_flight(receiver: &mut FlightReceiver) -> RefreshResult {
    loop {
        if let Some(result) = receiver.borrow().clone() {
            return result;
        }
        receiver
            .changed()
            .await
            .map_err(|_| CodexRefreshError::InvalidSingleFlightResult)?;
    }
}

fn accept_response(
    response: CodexRefreshHttpResponse,
    current: &CodexStoredCredentials,
    issued_at: SystemTime,
) -> RefreshResult {
    if response.status != 200 {
        return Err(CodexRefreshError::Http {
            status: response.status,
            retryable: !is_reused_refresh_token(&response.body),
        });
    }

    #[derive(Deserialize)]
    struct WireResponse {
        access_token: String,
        #[serde(default)]
        refresh_token: String,
        #[serde(default)]
        id_token: String,
        expires_in: i64,
    }

    let wire: WireResponse =
        serde_json::from_slice(&response.body).map_err(|_| CodexRefreshError::InvalidResponse)?;
    let access_token = SecretString::new(wire.access_token)?;
    let refresh_token = if wire.refresh_token.is_empty() {
        current.refresh_token().clone()
    } else {
        SecretString::new(wire.refresh_token)?
    };
    let id_token = if wire.id_token.is_empty() {
        current.id_token().clone()
    } else {
        SecretString::new(wire.id_token)?
    };
    let expires_at = if wire.expires_in >= 0 {
        issued_at.checked_add(Duration::from_secs(wire.expires_in as u64))
    } else {
        issued_at.checked_sub(Duration::from_secs(wire.expires_in.unsigned_abs()))
    }
    .ok_or(CodexTokenError::ExpiryOverflow)?;

    let claims = parse_jwt_token(id_token.expose_secret()).ok();
    Ok(CodexTokenData::new(
        id_token,
        access_token,
        refresh_token,
        claims
            .as_ref()
            .map(|claims| claims.account_id())
            .unwrap_or_default(),
        claims
            .as_ref()
            .map(|claims| claims.user_email())
            .unwrap_or_default(),
        expires_at,
    ))
}

fn is_reused_refresh_token(body: &[u8]) -> bool {
    #[derive(Deserialize)]
    struct OAuthError {
        #[serde(default)]
        code: String,
    }
    serde_json::from_slice::<OAuthError>(body)
        .map(|error| error.code.eq_ignore_ascii_case("refresh_token_reused"))
        .unwrap_or(false)
}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;

    use super::*;

    struct TestClock {
        now: SystemTime,
        sleeps: Mutex<Vec<Duration>>,
    }

    impl RefreshClock for TestClock {
        fn now(&self) -> SystemTime {
            self.now
        }

        fn sleep(
            &self,
            duration: Duration,
        ) -> Pin<Box<dyn Future<Output = Result<(), CodexRefreshTransportFailure>> + Send + '_>>
        {
            Box::pin(async move {
                lock_recover(&self.sleeps).push(duration);
                Ok(())
            })
        }
    }

    struct SequenceTransport {
        responses: Mutex<VecDeque<Result<CodexRefreshHttpResponse, CodexRefreshTransportFailure>>>,
        calls: AtomicUsize,
        timeouts: Mutex<Vec<Duration>>,
    }

    struct ExchangeTransport {
        response: Mutex<Option<Result<CodexExchangeHttpResponse, CodexRefreshTransportFailure>>>,
        form: Mutex<Option<Vec<u8>>>,
        timeout: Mutex<Option<Duration>>,
    }

    impl CodexCodeExchangeTransport for ExchangeTransport {
        fn exchange<'a>(
            &'a self,
            request: &'a CodexExchangeRequest,
            timeout: Duration,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<CodexExchangeHttpResponse, CodexRefreshTransportFailure>>
                    + Send
                    + 'a,
            >,
        > {
            Box::pin(async move {
                *lock_recover(&self.form) = Some(request.form_body().to_vec());
                *lock_recover(&self.timeout) = Some(timeout);
                lock_recover(&self.response).take().expect("one response")
            })
        }
    }

    impl CodexRefreshTransport for SequenceTransport {
        fn execute<'a>(
            &'a self,
            _request: &'a CodexRefreshRequest,
            timeout: Duration,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<CodexRefreshHttpResponse, CodexRefreshTransportFailure>>
                    + Send
                    + 'a,
            >,
        > {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                lock_recover(&self.timeouts).push(timeout);
                lock_recover(&self.responses)
                    .pop_front()
                    .expect("test response")
            })
        }
    }

    fn credentials() -> CodexStoredCredentials {
        CodexStoredCredentials::new(
            SecretString::new("old-id").unwrap(),
            SecretString::new("old-access").unwrap(),
            SecretString::new("shared-refresh").unwrap(),
        )
    }

    fn id_token() -> String {
        let payload = URL_SAFE_NO_PAD.encode(
            br#"{"email":"operator@example.com","https://api.openai.com/auth":{"chatgpt_account_id":"acct-7"}}"#,
        );
        format!("header.{payload}.signature")
    }

    #[test]
    fn refresh_form_and_debug_have_the_expected_secret_boundary() {
        let request = CodexRefreshRequest::new(SecretString::new("refresh-secret").unwrap());
        let pairs: HashMap<_, _> = url::form_urlencoded::parse(&request.form_body())
            .into_owned()
            .collect();
        assert_eq!(pairs.get("grant_type").unwrap(), "refresh_token");
        assert_eq!(pairs.get("scope").unwrap(), "openid profile email");
        assert_eq!(pairs.get("refresh_token").unwrap(), "refresh-secret");
        assert!(!format!("{request:?}").contains("refresh-secret"));
    }

    #[tokio::test]
    async fn retry_rotates_tokens_parses_claims_and_uses_bounded_timeout() {
        let body = format!(
            r#"{{"access_token":"new-access","refresh_token":"new-refresh","id_token":"{}","expires_in":3600}}"#,
            id_token()
        );
        let transport = SequenceTransport {
            responses: Mutex::new(
                vec![
                    Ok(CodexRefreshHttpResponse::new(
                        503,
                        b"upstream detail".to_vec(),
                    )),
                    Ok(CodexRefreshHttpResponse::new(200, body.into_bytes())),
                ]
                .into(),
            ),
            calls: AtomicUsize::new(0),
            timeouts: Mutex::new(Vec::new()),
        };
        let clock = TestClock {
            now: SystemTime::UNIX_EPOCH + Duration::from_secs(500),
            sleeps: Mutex::new(Vec::new()),
        };
        let token = CodexRefreshCoordinator::default()
            .refresh(&transport, &clock, credentials(), 3)
            .await
            .unwrap();
        assert_eq!(token.access_token().expose_secret(), "new-access");
        assert_eq!(token.refresh_token().expose_secret(), "new-refresh");
        assert_eq!(token.account_id(), "acct-7");
        assert_eq!(token.email(), "operator@example.com");
        assert_eq!(transport.calls.load(Ordering::SeqCst), 2);
        assert_eq!(*lock_recover(&transport.timeouts), vec![REFRESH_TIMEOUT; 2]);
        assert_eq!(*lock_recover(&clock.sleeps), vec![Duration::from_secs(1)]);
    }

    #[tokio::test]
    async fn exchange_supports_browser_and_device_redirects_and_redacts_secrets() {
        let issued_at = SystemTime::UNIX_EPOCH + Duration::from_secs(500);
        let body = format!(
            r#"{{"access_token":"access-secret","refresh_token":"refresh-secret","id_token":"{}","api_key":"api-secret","expires_in":3600}}"#,
            id_token()
        );
        let transport = ExchangeTransport {
            response: Mutex::new(Some(Ok(CodexExchangeHttpResponse::new(
                200,
                body.into_bytes(),
            )))),
            form: Mutex::new(None),
            timeout: Mutex::new(None),
        };
        let clock = TestClock {
            now: issued_at,
            sleeps: Mutex::new(Vec::new()),
        };
        let auth = CodexAuth::with_clock(transport, clock);
        let pkce =
            PkceCodes::new(SecretString::new("verifier-secret").unwrap(), "challenge").unwrap();
        let code = SecretString::new("code-secret").unwrap();
        let bundle = auth
            .exchange_code_for_tokens_with_redirect(
                &code,
                "https://auth.openai.com/deviceauth/callback",
                &pkce,
            )
            .await
            .unwrap();
        assert_eq!(bundle.token_data().email(), "operator@example.com");
        assert_eq!(bundle.token_data().account_id(), "acct-7");
        assert_eq!(bundle.api_key().unwrap().expose_secret(), "api-secret");
        assert_eq!(bundle.last_refresh(), issued_at);
        let form: HashMap<_, _> =
            url::form_urlencoded::parse(lock_recover(&auth.transport().form).as_deref().unwrap())
                .into_owned()
                .collect();
        assert_eq!(form.get("code").unwrap(), "code-secret");
        assert_eq!(form.get("code_verifier").unwrap(), "verifier-secret");
        assert_eq!(
            form.get("redirect_uri").unwrap(),
            "https://auth.openai.com/deviceauth/callback"
        );
        assert_eq!(
            *lock_recover(&auth.transport().timeout),
            Some(EXCHANGE_TIMEOUT)
        );
        let request = CodexExchangeRequest::new(&code, REDIRECT_URI, &pkce).unwrap();
        assert!(!format!("{request:?}").contains("code-secret"));
        assert!(!format!("{request:?}").contains("verifier-secret"));
    }

    #[tokio::test]
    async fn exchange_http_error_never_renders_provider_body() {
        let transport = ExchangeTransport {
            response: Mutex::new(Some(Ok(CodexExchangeHttpResponse::new(
                401,
                b"credential-detail-do-not-leak".to_vec(),
            )))),
            form: Mutex::new(None),
            timeout: Mutex::new(None),
        };
        let auth = CodexAuth::with_clock(
            transport,
            TestClock {
                now: SystemTime::UNIX_EPOCH,
                sleeps: Mutex::new(Vec::new()),
            },
        );
        let error = auth
            .exchange_code_for_tokens(
                &SecretString::new("code").unwrap(),
                &PkceCodes::new(SecretString::new("verifier").unwrap(), "challenge").unwrap(),
            )
            .await
            .unwrap_err();
        assert_eq!(error, CodexExchangeError::Http { status: 401 });
        assert!(!format!("{error:?} {error}").contains("do-not-leak"));
    }
}
