// ref: internal/auth/kimi/kimi.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::watch;
use zeroize::Zeroizing;

use crate::sdk::auth::LoginCancellation;

use super::token::{
    DeviceCodeResponse, KimiAuthBundle, KimiTokenData, KimiTokenError, KimiTokenStorage,
    SecretString,
};

pub const KIMI_CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
pub const KIMI_OAUTH_HOST: &str = "https://auth.kimi.com";
pub const KIMI_DEVICE_CODE_URL: &str = "https://auth.kimi.com/api/oauth/device_authorization";
pub const KIMI_TOKEN_URL: &str = "https://auth.kimi.com/api/oauth/token";
pub const KIMI_API_BASE_URL: &str = "https://api.kimi.com/coding";
pub const KIMI_DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);
pub const KIMI_MAX_POLL_DURATION: Duration = Duration::from_secs(15 * 60);
pub const KIMI_HTTP_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_KIMI_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct KimiDeviceIdentity {
    pub id: String,
    pub name: String,
    pub model: String,
    pub version: String,
}

impl KimiDeviceIdentity {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        model: impl Into<String>,
        version: impl Into<String>,
    ) -> Result<Self, KimiAuthError> {
        let identity = Self {
            id: id.into(),
            name: name.into(),
            model: model.into(),
            version: version.into(),
        };
        if identity.id.trim().is_empty() {
            return Err(KimiAuthError::new(KimiAuthErrorKind::InvalidDevice));
        }
        Ok(identity)
    }
}

pub struct KimiHttpRequest {
    pub url: &'static str,
    pub headers: BTreeMap<String, String>,
    pub body: Zeroizing<Vec<u8>>,
}

impl fmt::Debug for KimiHttpRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KimiHttpRequest")
            .field("url", &self.url)
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("body", &"[REDACTED]")
            .finish()
    }
}

pub struct KimiHttpResponse {
    pub status: u16,
    pub body: Zeroizing<Vec<u8>>,
}

impl KimiHttpResponse {
    #[must_use]
    pub fn new(status: u16, body: Vec<u8>) -> Self {
        Self {
            status,
            body: Zeroizing::new(body),
        }
    }
}

impl fmt::Debug for KimiHttpResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KimiHttpResponse")
            .field("status", &self.status)
            .field("body", &"[REDACTED]")
            .finish()
    }
}

pub type KimiHttpFuture<'a> =
    Pin<Box<dyn Future<Output = Result<KimiHttpResponse, KimiTransportFailure>> + Send + 'a>>;

pub trait KimiHttpTransport: Send + Sync {
    fn execute<'a>(
        &'a self,
        request: &'a KimiHttpRequest,
        timeout: Duration,
        cancellation: &'a LoginCancellation,
    ) -> KimiHttpFuture<'a>;
}

pub type KimiSleepFuture<'a> =
    Pin<Box<dyn Future<Output = Result<(), KimiTransportFailure>> + Send + 'a>>;

pub trait KimiClock: Send + Sync {
    fn now(&self) -> SystemTime;
    fn sleep<'a>(
        &'a self,
        duration: Duration,
        cancellation: &'a LoginCancellation,
    ) -> KimiSleepFuture<'a>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SystemKimiClock;

impl KimiClock for SystemKimiClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }

    fn sleep<'a>(
        &'a self,
        duration: Duration,
        cancellation: &'a LoginCancellation,
    ) -> KimiSleepFuture<'a> {
        Box::pin(async move {
            tokio::select! {
                () = tokio::time::sleep(duration) => Ok(()),
                () = cancellation.cancelled() => Err(KimiTransportFailure::Cancelled),
            }
        })
    }
}

#[derive(Clone)]
pub struct KimiAuth {
    client: Arc<DeviceFlowClient>,
}

impl KimiAuth {
    #[must_use]
    pub fn new(client: Arc<DeviceFlowClient>) -> Self {
        Self { client }
    }

    pub async fn start_device_flow(
        &self,
        cancellation: &LoginCancellation,
    ) -> Result<DeviceCodeResponse, KimiAuthError> {
        self.client.request_device_code(cancellation).await
    }

    pub async fn wait_for_authorization(
        &self,
        cancellation: &LoginCancellation,
        device_code: &DeviceCodeResponse,
    ) -> Result<KimiAuthBundle, KimiAuthError> {
        let token_data = self
            .client
            .poll_for_token(cancellation, device_code)
            .await?;
        Ok(KimiAuthBundle::new(token_data, &self.client.identity.id))
    }

    #[must_use]
    pub fn create_token_storage(&self, bundle: &KimiAuthBundle) -> KimiTokenStorage {
        KimiTokenStorage::from_bundle(bundle)
    }
}

pub struct DeviceFlowClient {
    transport: Arc<dyn KimiHttpTransport>,
    clock: Arc<dyn KimiClock>,
    identity: KimiDeviceIdentity,
    refresh: Arc<KimiRefreshCoordinator>,
}

impl fmt::Debug for DeviceFlowClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceFlowClient")
            .field("identity", &self.identity)
            .field("transport", &"[INJECTED]")
            .field("clock", &"[INJECTED]")
            .finish()
    }
}

impl DeviceFlowClient {
    #[must_use]
    pub fn new(
        transport: Arc<dyn KimiHttpTransport>,
        clock: Arc<dyn KimiClock>,
        identity: KimiDeviceIdentity,
        refresh: Arc<KimiRefreshCoordinator>,
    ) -> Self {
        Self {
            transport,
            clock,
            identity,
            refresh,
        }
    }

    pub async fn request_device_code(
        &self,
        cancellation: &LoginCancellation,
    ) -> Result<DeviceCodeResponse, KimiAuthError> {
        let request = self.form_request(KIMI_DEVICE_CODE_URL, [("client_id", KIMI_CLIENT_ID)]);
        let response = self
            .transport
            .execute(&request, KIMI_HTTP_TIMEOUT, cancellation)
            .await
            .map_err(KimiAuthError::transport)?;
        validate_body_size(&response)?;
        if response.status != 200 {
            return Err(KimiAuthError::status(
                KimiAuthErrorKind::DeviceCodeRejected,
                response.status,
            ));
        }
        let response: DeviceCodeResponse = serde_json::from_slice(&response.body)
            .map_err(|error| KimiAuthError::source(KimiAuthErrorKind::Decode, error))?;
        if response.device_code.trim().is_empty() || response.user_code.trim().is_empty() {
            return Err(KimiAuthError::new(KimiAuthErrorKind::MissingFields));
        }
        Ok(response)
    }

    pub async fn poll_for_token(
        &self,
        cancellation: &LoginCancellation,
        device_code: &DeviceCodeResponse,
    ) -> Result<KimiTokenData, KimiAuthError> {
        if device_code.device_code.trim().is_empty() {
            return Err(KimiAuthError::new(KimiAuthErrorKind::MissingDeviceCode));
        }
        let interval =
            Duration::from_secs(device_code.interval.max(0) as u64).max(KIMI_DEFAULT_POLL_INTERVAL);
        let now = self.clock.now();
        let mut deadline = now + KIMI_MAX_POLL_DURATION;
        if device_code.expires_in > 0 {
            let code_deadline = now + Duration::from_secs(device_code.expires_in as u64);
            deadline = deadline.min(code_deadline);
        }
        loop {
            self.clock
                .sleep(interval, cancellation)
                .await
                .map_err(KimiAuthError::transport)?;
            if self.clock.now() > deadline {
                return Err(KimiAuthError::new(KimiAuthErrorKind::DeviceCodeExpired));
            }
            match self
                .exchange_device_code(cancellation, &device_code.device_code)
                .await?
            {
                ExchangeOutcome::Token(token) => return Ok(token),
                ExchangeOutcome::Pending => {}
            }
        }
    }

    async fn exchange_device_code(
        &self,
        cancellation: &LoginCancellation,
        device_code: &str,
    ) -> Result<ExchangeOutcome, KimiAuthError> {
        let request = self.form_request(
            KIMI_TOKEN_URL,
            [
                ("client_id", KIMI_CLIENT_ID),
                ("device_code", device_code),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ],
        );
        let response = self
            .transport
            .execute(&request, KIMI_HTTP_TIMEOUT, cancellation)
            .await
            .map_err(KimiAuthError::transport)?;
        self.parse_token_response(response)
    }

    pub async fn refresh_token(
        &self,
        refresh_token: SecretString,
    ) -> Result<KimiTokenData, KimiAuthError> {
        self.refresh.refresh(self, refresh_token).await
    }

    async fn refresh_token_once(
        &self,
        refresh_token: &SecretString,
    ) -> Result<KimiTokenData, KimiAuthError> {
        let request = self.form_request(
            KIMI_TOKEN_URL,
            [
                ("client_id", KIMI_CLIENT_ID),
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token.expose_secret()),
            ],
        );
        // Upstream detaches the shared refresh from the initiating caller.
        let detached = LoginCancellation::default();
        let response = self
            .transport
            .execute(&request, KIMI_HTTP_TIMEOUT, &detached)
            .await
            .map_err(KimiAuthError::transport)?;
        validate_body_size(&response)?;
        if matches!(response.status, 401 | 403) {
            return Err(KimiAuthError::status(
                KimiAuthErrorKind::RefreshRejected,
                response.status,
            ));
        }
        if response.status != 200 {
            return Err(KimiAuthError::status(
                KimiAuthErrorKind::RefreshFailed,
                response.status,
            ));
        }
        match self.parse_token_response(response)? {
            ExchangeOutcome::Token(token) => Ok(token),
            ExchangeOutcome::Pending => Err(KimiAuthError::new(KimiAuthErrorKind::InvalidResponse)),
        }
    }

    fn parse_token_response(
        &self,
        response: KimiHttpResponse,
    ) -> Result<ExchangeOutcome, KimiAuthError> {
        validate_body_size(&response)?;
        let parsed: OAuthTokenResponse = serde_json::from_slice(&response.body)
            .map_err(|error| KimiAuthError::source(KimiAuthErrorKind::Decode, error))?;
        match parsed.error.as_str() {
            "authorization_pending" | "slow_down" => return Ok(ExchangeOutcome::Pending),
            "expired_token" => {
                return Err(KimiAuthError::new(KimiAuthErrorKind::DeviceCodeExpired))
            }
            "access_denied" => return Err(KimiAuthError::new(KimiAuthErrorKind::AccessDenied)),
            "" => {}
            _ => return Err(KimiAuthError::new(KimiAuthErrorKind::OAuth)),
        }
        let access_token = SecretString::new(parsed.access_token).map_err(KimiAuthError::token)?;
        let refresh_token = optional_secret(parsed.refresh_token)?;
        let expires_at = finite_positive_seconds(parsed.expires_in)
            .and_then(|expires_in| self.clock.now().checked_add(expires_in));
        Ok(ExchangeOutcome::Token(KimiTokenData::new(
            access_token,
            refresh_token,
            parsed.token_type,
            expires_at,
            parsed.scope,
        )))
    }

    fn form_request<'a, const N: usize>(
        &self,
        url: &'static str,
        fields: [(&'a str, &'a str); N],
    ) -> KimiHttpRequest {
        let body = fields
            .into_iter()
            .fold(
                url::form_urlencoded::Serializer::new(String::new()),
                |mut form, (key, value)| {
                    form.append_pair(key, value);
                    form
                },
            )
            .finish();
        KimiHttpRequest {
            url,
            headers: self.common_headers(),
            body: Zeroizing::new(body.into_bytes()),
        }
    }

    fn common_headers(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("Accept".to_owned(), "application/json".to_owned()),
            (
                "Content-Type".to_owned(),
                "application/x-www-form-urlencoded".to_owned(),
            ),
            ("X-Msh-Platform".to_owned(), "CLIProxyAPI".to_owned()),
            ("X-Msh-Version".to_owned(), self.identity.version.clone()),
            ("X-Msh-Device-Name".to_owned(), self.identity.name.clone()),
            ("X-Msh-Device-Model".to_owned(), self.identity.model.clone()),
            ("X-Msh-Device-Id".to_owned(), self.identity.id.clone()),
        ])
    }
}

#[derive(Debug)]
enum ExchangeOutcome {
    Token(KimiTokenData),
    Pending,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct OAuthTokenResponse {
    error: String,
    access_token: String,
    refresh_token: String,
    token_type: String,
    expires_in: f64,
    scope: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KimiTransportFailure {
    Timeout,
    Connect,
    Protocol,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KimiAuthErrorKind {
    Cancelled,
    Transport,
    InvalidDevice,
    MissingDeviceCode,
    DeviceCodeRejected,
    DeviceCodeExpired,
    MissingFields,
    Decode,
    ResponseTooLarge,
    AccessDenied,
    OAuth,
    RefreshRejected,
    RefreshFailed,
    InvalidResponse,
    Token,
    InvalidSingleFlightResult,
}

#[derive(Clone)]
pub struct KimiAuthError {
    pub kind: KimiAuthErrorKind,
    pub status: Option<u16>,
    pub transport: Option<KimiTransportFailure>,
    source: Option<Arc<dyn std::error::Error + Send + Sync + 'static>>,
}

impl KimiAuthError {
    #[must_use]
    pub fn new(kind: KimiAuthErrorKind) -> Self {
        Self {
            kind,
            status: None,
            transport: None,
            source: None,
        }
    }

    fn status(kind: KimiAuthErrorKind, status: u16) -> Self {
        Self {
            kind,
            status: Some(status),
            transport: None,
            source: None,
        }
    }

    fn transport(failure: KimiTransportFailure) -> Self {
        Self {
            kind: if failure == KimiTransportFailure::Cancelled {
                KimiAuthErrorKind::Cancelled
            } else {
                KimiAuthErrorKind::Transport
            },
            status: None,
            transport: Some(failure),
            source: None,
        }
    }

    fn token(error: KimiTokenError) -> Self {
        Self::source(KimiAuthErrorKind::Token, error)
    }

    fn source(
        kind: KimiAuthErrorKind,
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

impl fmt::Debug for KimiAuthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KimiAuthError")
            .field("kind", &self.kind)
            .field("status", &self.status)
            .field("transport", &self.transport)
            .field("has_source", &self.source.is_some())
            .finish()
    }
}

impl fmt::Display for KimiAuthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            KimiAuthErrorKind::Cancelled => "Kimi authentication cancelled",
            KimiAuthErrorKind::Transport => "Kimi authentication transport failed",
            KimiAuthErrorKind::InvalidDevice => "Kimi device identity is invalid",
            KimiAuthErrorKind::MissingDeviceCode => "Kimi device code is missing",
            KimiAuthErrorKind::DeviceCodeRejected => "Kimi device-code request was rejected",
            KimiAuthErrorKind::DeviceCodeExpired => "Kimi device code expired",
            KimiAuthErrorKind::MissingFields => "Kimi response is missing required fields",
            KimiAuthErrorKind::Decode => "Kimi response could not be decoded",
            KimiAuthErrorKind::ResponseTooLarge => "Kimi response exceeded the size limit",
            KimiAuthErrorKind::AccessDenied => "Kimi access was denied by the user",
            KimiAuthErrorKind::OAuth => "Kimi OAuth request failed",
            KimiAuthErrorKind::RefreshRejected => "Kimi refresh token was rejected",
            KimiAuthErrorKind::RefreshFailed => "Kimi token refresh failed",
            KimiAuthErrorKind::InvalidResponse => "Kimi token response is invalid",
            KimiAuthErrorKind::Token => "Kimi token is invalid",
            KimiAuthErrorKind::InvalidSingleFlightResult => {
                "Kimi refresh singleflight result is invalid"
            }
        })
    }
}

impl std::error::Error for KimiAuthError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

type RefreshResult = Result<KimiTokenData, KimiAuthError>;
type RefreshReceiver = watch::Receiver<Option<RefreshResult>>;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct RefreshFingerprint([u8; 32]);

impl RefreshFingerprint {
    fn new(token: &SecretString) -> Self {
        Self(Sha256::digest(token.expose_secret().as_bytes()).into())
    }
}

impl fmt::Debug for RefreshFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RefreshFingerprint([REDACTED])")
    }
}

#[derive(Default)]
pub struct KimiRefreshCoordinator {
    flights: Mutex<HashMap<RefreshFingerprint, RefreshReceiver>>,
}

impl KimiRefreshCoordinator {
    async fn refresh(
        &self,
        client: &DeviceFlowClient,
        refresh_token: SecretString,
    ) -> RefreshResult {
        let fingerprint = RefreshFingerprint::new(&refresh_token);
        let (mut receiver, leader) = {
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
        let Some(sender) = leader else {
            return wait_for_refresh(&mut receiver).await;
        };
        let result = client.refresh_token_once(&refresh_token).await;
        if sender.send(Some(result.clone())).is_err() {
            self.remove(fingerprint);
            return Err(KimiAuthError::new(
                KimiAuthErrorKind::InvalidSingleFlightResult,
            ));
        }
        self.remove(fingerprint);
        result
    }

    fn remove(&self, fingerprint: RefreshFingerprint) {
        self.flights
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&fingerprint);
    }
}

async fn wait_for_refresh(receiver: &mut RefreshReceiver) -> RefreshResult {
    loop {
        if let Some(result) = receiver.borrow().clone() {
            return result;
        }
        if receiver.changed().await.is_err() {
            return Err(KimiAuthError::new(
                KimiAuthErrorKind::InvalidSingleFlightResult,
            ));
        }
    }
}

fn validate_body_size(response: &KimiHttpResponse) -> Result<(), KimiAuthError> {
    if response.body.len() > MAX_KIMI_RESPONSE_BYTES {
        Err(KimiAuthError::new(KimiAuthErrorKind::ResponseTooLarge))
    } else {
        Ok(())
    }
}

fn optional_secret(value: String) -> Result<Option<SecretString>, KimiAuthError> {
    if value.trim().is_empty() {
        Ok(None)
    } else {
        SecretString::new(value)
            .map(Some)
            .map_err(KimiAuthError::token)
    }
}

fn finite_positive_seconds(value: f64) -> Option<Duration> {
    (value.is_finite() && value > 0.0 && value < u64::MAX as f64)
        .then(|| Duration::from_secs(value as u64))
}
