// ref: internal/auth/xai/xai.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use base64::Engine as _;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::watch;
use url::Url;
use zeroize::Zeroizing;

use crate::sdk::auth::LoginCancellation;

use super::token::{SecretString, TokenStorage, XaiTokenError};
use super::types::{
    AuthBundle, DeviceCodeResponse, Discovery, TokenData, CLIENT_ID, DEFAULT_API_BASE_URL,
    DEFAULT_POLL_INTERVAL, DEVICE_CODE_GRANT_TYPE, DISCOVERY_URL, HTTP_CLIENT_TIMEOUT,
    MAX_POLL_DURATION, SCOPE,
};

const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XaiHttpMethod {
    Get,
    Post,
}

pub struct XaiHttpRequest {
    pub method: XaiHttpMethod,
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Zeroizing<Vec<u8>>,
    pub proxy_url: Option<String>,
}

impl fmt::Debug for XaiHttpRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XaiHttpRequest")
            .field("method", &self.method)
            .field("url", &self.url)
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("body", &"[REDACTED]")
            .field("has_proxy", &self.proxy_url.is_some())
            .finish()
    }
}

pub struct XaiHttpResponse {
    pub status: u16,
    pub body: Zeroizing<Vec<u8>>,
}
impl XaiHttpResponse {
    pub fn new(status: u16, body: Vec<u8>) -> Self {
        Self {
            status,
            body: Zeroizing::new(body),
        }
    }
}
impl fmt::Debug for XaiHttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XaiHttpResponse")
            .field("status", &self.status)
            .field("body", &"[REDACTED]")
            .finish()
    }
}

pub type XaiHttpFuture<'a> =
    Pin<Box<dyn Future<Output = Result<XaiHttpResponse, XaiTransportFailure>> + Send + 'a>>;
pub trait XaiHttpTransport: Send + Sync {
    fn execute<'a>(
        &'a self,
        request: &'a XaiHttpRequest,
        timeout: Duration,
        cancellation: &'a LoginCancellation,
    ) -> XaiHttpFuture<'a>;
}

pub type XaiSleepFuture<'a> =
    Pin<Box<dyn Future<Output = Result<(), XaiTransportFailure>> + Send + 'a>>;
pub trait XaiClock: Send + Sync {
    fn now(&self) -> SystemTime;
    fn sleep<'a>(
        &'a self,
        duration: Duration,
        cancellation: &'a LoginCancellation,
    ) -> XaiSleepFuture<'a>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemXaiClock;
impl XaiClock for SystemXaiClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
    fn sleep<'a>(
        &'a self,
        duration: Duration,
        cancellation: &'a LoginCancellation,
    ) -> XaiSleepFuture<'a> {
        Box::pin(async move {
            tokio::select! {
                () = tokio::time::sleep(duration) => Ok(()),
                () = cancellation.cancelled() => Err(XaiTransportFailure::Cancelled),
            }
        })
    }
}

pub struct XaiAuth {
    transport: Arc<dyn XaiHttpTransport>,
    clock: Arc<dyn XaiClock>,
    refresh: Arc<XaiRefreshCoordinator>,
    proxy_url: Option<String>,
}

impl XaiAuth {
    pub fn new(
        transport: Arc<dyn XaiHttpTransport>,
        clock: Arc<dyn XaiClock>,
        refresh: Arc<XaiRefreshCoordinator>,
    ) -> Self {
        Self::with_proxy_url(transport, clock, refresh, None)
    }

    pub fn with_proxy_url(
        transport: Arc<dyn XaiHttpTransport>,
        clock: Arc<dyn XaiClock>,
        refresh: Arc<XaiRefreshCoordinator>,
        proxy_url: Option<String>,
    ) -> Self {
        Self {
            transport,
            clock,
            refresh,
            proxy_url: proxy_url.and_then(non_empty),
        }
    }

    pub async fn discover(
        &self,
        cancellation: &LoginCancellation,
    ) -> Result<Discovery, XaiAuthError> {
        let response = self
            .execute(
                self.request(XaiHttpMethod::Get, DISCOVERY_URL, None),
                cancellation,
            )
            .await?;
        ensure_status(&response, XaiAuthErrorKind::DiscoveryFailed)?;
        let discovery: Discovery = decode(&response)?;
        Ok(Discovery {
            device_authorization_endpoint: validate_oauth_endpoint(
                &discovery.device_authorization_endpoint,
                "device_authorization_endpoint",
            )?,
            token_endpoint: validate_oauth_endpoint(&discovery.token_endpoint, "token_endpoint")?,
        })
    }

    pub async fn start_device_flow(
        &self,
        cancellation: &LoginCancellation,
    ) -> Result<DeviceCodeResponse, XaiAuthError> {
        let discovery = self.discover(cancellation).await?;
        self.request_device_code(
            cancellation,
            &discovery.device_authorization_endpoint,
            &discovery.token_endpoint,
        )
        .await
    }

    pub async fn request_device_code(
        &self,
        cancellation: &LoginCancellation,
        endpoint: &str,
        token_endpoint: &str,
    ) -> Result<DeviceCodeResponse, XaiAuthError> {
        if endpoint.trim().is_empty() {
            return Err(XaiAuthError::new(XaiAuthErrorKind::MissingEndpoint));
        }
        let body = form(&[("client_id", CLIENT_ID), ("scope", SCOPE)]);
        let response = self
            .execute(
                self.request(XaiHttpMethod::Post, endpoint.trim(), Some(body)),
                cancellation,
            )
            .await?;
        ensure_status(&response, XaiAuthErrorKind::DeviceCodeRejected)?;
        let mut code: DeviceCodeResponse = decode(&response)?;
        if code.device_code.trim().is_empty() {
            return Err(XaiAuthError::new(XaiAuthErrorKind::MissingDeviceCode));
        }
        if code.user_code.trim().is_empty() {
            return Err(XaiAuthError::new(XaiAuthErrorKind::MissingUserCode));
        }
        if code.verification_uri.trim().is_empty()
            && code.verification_uri_complete.trim().is_empty()
        {
            return Err(XaiAuthError::new(XaiAuthErrorKind::MissingVerificationUri));
        }
        code.token_endpoint = token_endpoint.trim().to_owned();
        Ok(code)
    }

    pub async fn wait_for_authorization(
        &self,
        cancellation: &LoginCancellation,
        code: &DeviceCodeResponse,
    ) -> Result<AuthBundle, XaiAuthError> {
        let token_data = self.poll_for_token(cancellation, Some(code)).await?;
        Ok(AuthBundle {
            token_data,
            last_refresh: self.clock.now(),
            base_url: DEFAULT_API_BASE_URL.to_owned(),
            redirect_uri: String::new(),
            token_endpoint: code.token_endpoint.trim().to_owned(),
        })
    }

    pub async fn poll_for_token(
        &self,
        cancellation: &LoginCancellation,
        code: Option<&DeviceCodeResponse>,
    ) -> Result<TokenData, XaiAuthError> {
        let code = code.ok_or_else(|| XaiAuthError::new(XaiAuthErrorKind::NilDeviceCode))?;
        let endpoint = if code.token_endpoint.trim().is_empty() {
            self.discover(cancellation).await?.token_endpoint
        } else {
            code.token_endpoint.trim().to_owned()
        };
        let mut interval =
            Duration::from_secs(code.interval.max(0) as u64).max(DEFAULT_POLL_INTERVAL);
        let now = self.clock.now();
        let max_deadline = now.checked_add(MAX_POLL_DURATION).unwrap_or(now);
        let deadline = if code.expires_in > 0 {
            now.checked_add(Duration::from_secs(code.expires_in as u64))
                .map_or(max_deadline, |value| value.min(max_deadline))
        } else {
            max_deadline
        };
        let mut first = true;
        loop {
            if cancellation.is_cancelled() {
                return Err(XaiAuthError::new(XaiAuthErrorKind::Cancelled));
            }
            if !first {
                self.clock
                    .sleep(interval, cancellation)
                    .await
                    .map_err(XaiAuthError::transport)?;
                if self.clock.now() > deadline {
                    return Err(XaiAuthError::new(XaiAuthErrorKind::DeviceCodeExpired));
                }
            }
            first = false;
            match self
                .exchange_device_code(cancellation, &endpoint, &code.device_code)
                .await?
            {
                ExchangeOutcome::Token(token) => return Ok(token),
                ExchangeOutcome::Pending => {}
                ExchangeOutcome::SlowDown => interval += DEFAULT_POLL_INTERVAL,
            }
        }
    }

    async fn exchange_device_code(
        &self,
        cancellation: &LoginCancellation,
        endpoint: &str,
        device_code: &str,
    ) -> Result<ExchangeOutcome, XaiAuthError> {
        let body = form(&[
            ("grant_type", DEVICE_CODE_GRANT_TYPE),
            ("device_code", device_code.trim()),
            ("client_id", CLIENT_ID),
        ]);
        let response = self
            .execute(
                self.request(XaiHttpMethod::Post, endpoint, Some(body)),
                cancellation,
            )
            .await?;
        let payload: TokenResponse = decode(&response)?;
        match payload.error.as_str() {
            "authorization_pending" => return Ok(ExchangeOutcome::Pending),
            "slow_down" => return Ok(ExchangeOutcome::SlowDown),
            "expired_token" => return Err(XaiAuthError::new(XaiAuthErrorKind::DeviceCodeExpired)),
            "access_denied" => return Err(XaiAuthError::new(XaiAuthErrorKind::AccessDenied)),
            "" => {}
            _ => return Err(XaiAuthError::new(XaiAuthErrorKind::OAuth)),
        }
        if response.status != 200 {
            return Err(XaiAuthError::status(
                XaiAuthErrorKind::TokenExchangeFailed,
                response.status,
            ));
        }
        self.build_token_data(payload).map(ExchangeOutcome::Token)
    }

    pub async fn refresh_tokens(
        &self,
        refresh_token: SecretString,
        token_endpoint: Option<&str>,
    ) -> Result<TokenData, XaiAuthError> {
        let endpoint = if let Some(endpoint) = token_endpoint.and_then(non_empty_ref) {
            endpoint.to_owned()
        } else {
            self.discover(&LoginCancellation::default())
                .await?
                .token_endpoint
        };
        self.refresh.refresh(self, refresh_token, endpoint).await
    }

    async fn refresh_once(
        &self,
        refresh_token: &SecretString,
        endpoint: &str,
    ) -> Result<TokenData, XaiAuthError> {
        let body = form(&[
            ("grant_type", "refresh_token"),
            ("client_id", CLIENT_ID),
            ("refresh_token", refresh_token.expose_secret()),
        ]);
        // Matches context.WithoutCancel: a shared refresh has coordinator-owned lifetime.
        let response = self
            .execute(
                self.request(XaiHttpMethod::Post, endpoint, Some(body)),
                &LoginCancellation::default(),
            )
            .await?;
        ensure_status(&response, XaiAuthErrorKind::RefreshFailed)?;
        let payload: TokenResponse = decode(&response)?;
        self.build_token_data(payload)
    }

    pub fn create_token_storage(&self, bundle: Option<&AuthBundle>) -> Option<TokenStorage> {
        bundle.map(TokenStorage::from_bundle)
    }

    fn build_token_data(&self, payload: TokenResponse) -> Result<TokenData, XaiAuthError> {
        let access = SecretString::new(payload.access_token).map_err(XaiAuthError::token)?;
        let refresh = optional_secret(payload.refresh_token)?;
        let identity = optional_secret(payload.id_token)?;
        let (email, subject) = identity.as_ref().map_or_else(
            || (String::new(), String::new()),
            |value| parse_jwt_identity(value.expose_secret()),
        );
        let expires_at = (payload.expires_in > 0)
            .then(|| Duration::from_secs(payload.expires_in as u64))
            .and_then(|duration| self.clock.now().checked_add(duration));
        Ok(TokenData::new(
            access,
            refresh,
            identity,
            payload.token_type,
            payload.expires_in,
            expires_at,
            email,
            subject,
        ))
    }

    fn request(&self, method: XaiHttpMethod, url: &str, body: Option<String>) -> XaiHttpRequest {
        let mut headers = BTreeMap::from([("Accept".to_owned(), "application/json".to_owned())]);
        if body.is_some() {
            headers.insert(
                "Content-Type".to_owned(),
                "application/x-www-form-urlencoded".to_owned(),
            );
        }
        XaiHttpRequest {
            method,
            url: url.to_owned(),
            headers,
            body: Zeroizing::new(body.unwrap_or_default().into_bytes()),
            proxy_url: self.proxy_url.clone(),
        }
    }

    async fn execute(
        &self,
        request: XaiHttpRequest,
        cancellation: &LoginCancellation,
    ) -> Result<XaiHttpResponse, XaiAuthError> {
        let response = self
            .transport
            .execute(&request, HTTP_CLIENT_TIMEOUT, cancellation)
            .await
            .map_err(XaiAuthError::transport)?;
        if response.body.len() > MAX_RESPONSE_BYTES {
            return Err(XaiAuthError::new(XaiAuthErrorKind::ResponseTooLarge));
        }
        Ok(response)
    }
}

pub fn validate_oauth_endpoint(raw: &str, _field: &str) -> Result<String, XaiAuthError> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(XaiAuthError::new(XaiAuthErrorKind::MissingEndpoint));
    }
    let parsed = Url::parse(raw)
        .map_err(|error| XaiAuthError::source(XaiAuthErrorKind::InvalidEndpoint, error))?;
    if parsed.scheme() != "https" {
        return Err(XaiAuthError::new(XaiAuthErrorKind::InsecureEndpoint));
    }
    let host = parsed
        .host_str()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if host != "x.ai" && !host.ends_with(".x.ai") {
        return Err(XaiAuthError::new(XaiAuthErrorKind::ForeignEndpoint));
    }
    Ok(raw.to_owned())
}

fn form(fields: &[(&str, &str)]) -> String {
    fields
        .iter()
        .fold(
            url::form_urlencoded::Serializer::new(String::new()),
            |mut form, (key, value)| {
                form.append_pair(key, value);
                form
            },
        )
        .finish()
}

fn decode<'a, T: Deserialize<'a>>(response: &'a XaiHttpResponse) -> Result<T, XaiAuthError> {
    serde_json::from_slice(&response.body)
        .map_err(|error| XaiAuthError::source(XaiAuthErrorKind::Decode, error))
}

fn ensure_status(response: &XaiHttpResponse, kind: XaiAuthErrorKind) -> Result<(), XaiAuthError> {
    if response.status == 200 {
        Ok(())
    } else {
        Err(XaiAuthError::status(kind, response.status))
    }
}

fn optional_secret(value: String) -> Result<Option<SecretString>, XaiAuthError> {
    if value.trim().is_empty() {
        Ok(None)
    } else {
        SecretString::new(value)
            .map(Some)
            .map_err(XaiAuthError::token)
    }
}
fn non_empty(value: String) -> Option<String> {
    (!value.trim().is_empty()).then(|| value.trim().to_owned())
}
fn non_empty_ref(value: &str) -> Option<&str> {
    (!value.trim().is_empty()).then(|| value.trim())
}

pub fn parse_jwt_identity(token: &str) -> (String, String) {
    let Some(payload) = token.split('.').nth(1) else {
        return (String::new(), String::new());
    };
    let Ok(bytes) = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(payload))
    else {
        return (String::new(), String::new());
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return (String::new(), String::new());
    };
    (
        value
            .get("email")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        value
            .get("sub")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned(),
    )
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct TokenResponse {
    error: String,
    access_token: String,
    refresh_token: String,
    id_token: String,
    token_type: String,
    expires_in: i64,
}
enum ExchangeOutcome {
    Token(TokenData),
    Pending,
    SlowDown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XaiTransportFailure {
    Timeout,
    Connect,
    Protocol,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XaiAuthErrorKind {
    Cancelled,
    Transport,
    MissingEndpoint,
    InvalidEndpoint,
    InsecureEndpoint,
    ForeignEndpoint,
    DiscoveryFailed,
    DeviceCodeRejected,
    MissingDeviceCode,
    MissingUserCode,
    MissingVerificationUri,
    NilDeviceCode,
    DeviceCodeExpired,
    AccessDenied,
    OAuth,
    TokenExchangeFailed,
    RefreshFailed,
    Decode,
    ResponseTooLarge,
    Token,
    InvalidSingleFlightResult,
}

#[derive(Clone)]
pub struct XaiAuthError {
    pub kind: XaiAuthErrorKind,
    pub status: Option<u16>,
    pub transport: Option<XaiTransportFailure>,
    source: Option<Arc<dyn std::error::Error + Send + Sync>>,
}
impl XaiAuthError {
    pub fn new(kind: XaiAuthErrorKind) -> Self {
        Self {
            kind,
            status: None,
            transport: None,
            source: None,
        }
    }
    fn status(kind: XaiAuthErrorKind, status: u16) -> Self {
        Self {
            kind,
            status: Some(status),
            transport: None,
            source: None,
        }
    }
    fn transport(value: XaiTransportFailure) -> Self {
        Self {
            kind: if value == XaiTransportFailure::Cancelled {
                XaiAuthErrorKind::Cancelled
            } else {
                XaiAuthErrorKind::Transport
            },
            status: None,
            transport: Some(value),
            source: None,
        }
    }
    fn token(error: XaiTokenError) -> Self {
        Self::source(XaiAuthErrorKind::Token, error)
    }
    fn source(
        kind: XaiAuthErrorKind,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind,
            status: None,
            transport: None,
            source: Some(Arc::new(source)),
        }
    }
}
impl fmt::Debug for XaiAuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XaiAuthError")
            .field("kind", &self.kind)
            .field("status", &self.status)
            .field("transport", &self.transport)
            .field("has_source", &self.source.is_some())
            .finish()
    }
}
impl fmt::Display for XaiAuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "xAI authentication failed: {:?}", self.kind)
    }
}
impl std::error::Error for XaiAuthError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_deref()
            .map(|value| value as &(dyn std::error::Error + 'static))
    }
}

type RefreshResult = Result<TokenData, XaiAuthError>;
type RefreshReceiver = watch::Receiver<Option<RefreshResult>>;
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct RefreshFingerprint([u8; 32]);
impl RefreshFingerprint {
    fn new(token: &SecretString) -> Self {
        Self(Sha256::digest(token.expose_secret().as_bytes()).into())
    }
}

#[derive(Default)]
pub struct XaiRefreshCoordinator {
    flights: Mutex<HashMap<RefreshFingerprint, RefreshReceiver>>,
}
impl XaiRefreshCoordinator {
    async fn refresh(
        &self,
        client: &XaiAuth,
        token: SecretString,
        endpoint: String,
    ) -> RefreshResult {
        let fingerprint = RefreshFingerprint::new(&token);
        let (mut receiver, sender) = {
            let mut flights = self
                .flights
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(receiver) = flights.get(&fingerprint) {
                (receiver.clone(), None)
            } else {
                let (sender, receiver) = watch::channel(None);
                flights.insert(fingerprint, receiver.clone());
                (receiver, Some(sender))
            }
        };
        let Some(sender) = sender else {
            return wait_for_refresh(&mut receiver).await;
        };
        let result = client.refresh_once(&token, &endpoint).await;
        let sent = sender.send(Some(result.clone())).is_ok();
        self.flights
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&fingerprint);
        if sent {
            result
        } else {
            Err(XaiAuthError::new(
                XaiAuthErrorKind::InvalidSingleFlightResult,
            ))
        }
    }
}

async fn wait_for_refresh(receiver: &mut RefreshReceiver) -> RefreshResult {
    loop {
        if let Some(result) = receiver.borrow().clone() {
            return result;
        }
        if receiver.changed().await.is_err() {
            return Err(XaiAuthError::new(
                XaiAuthErrorKind::InvalidSingleFlightResult,
            ));
        }
    }
}
