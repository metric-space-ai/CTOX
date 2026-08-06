// ref: sdk/auth/codex_device.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::error::Error;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{LoginCancellation, LoginOptions};

pub const CODEX_LOGIN_MODE_METADATA_KEY: &str = "codex_login_mode";
pub const CODEX_LOGIN_MODE_DEVICE: &str = "device";
pub const CODEX_DEVICE_USER_CODE_URL: &str =
    "https://auth.openai.com/api/accounts/deviceauth/usercode";
pub const CODEX_DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
pub const CODEX_DEVICE_VERIFICATION_URL: &str = "https://auth.openai.com/codex/device";
pub const CODEX_DEVICE_TOKEN_EXCHANGE_REDIRECT_URI: &str =
    "https://auth.openai.com/deviceauth/callback";
pub const CODEX_DEVICE_TIMEOUT: Duration = Duration::from_secs(15 * 60);
pub const CODEX_DEVICE_DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);
const MAX_DEVICE_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct CodexDeviceUserCodeRequest<'a> {
    pub client_id: &'a str,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct CodexDeviceUserCodeResponse {
    pub device_auth_id: String,
    pub user_code: String,
    #[serde(default)]
    pub usercode: String,
    #[serde(default)]
    pub interval: Value,
}

impl CodexDeviceUserCodeResponse {
    #[must_use]
    pub fn effective_user_code(&self) -> &str {
        let user_code = self.user_code.trim();
        if user_code.is_empty() {
            self.usercode.trim()
        } else {
            user_code
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct CodexDeviceTokenRequest<'a> {
    pub device_auth_id: &'a str,
    pub user_code: &'a str,
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Deserialize)]
#[serde(default)]
pub struct CodexDeviceTokenResponse {
    pub authorization_code: String,
    pub code_verifier: String,
    pub code_challenge: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DeviceHttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

pub type DeviceHttpFuture<'a> =
    Pin<Box<dyn Future<Output = Result<DeviceHttpResponse, DeviceFlowError>> + Send + 'a>>;

pub trait CodexDeviceTransport: Send + Sync {
    fn post_json<'a>(
        &'a self,
        url: &'a str,
        body: &'a [u8],
        cancellation: &'a LoginCancellation,
    ) -> DeviceHttpFuture<'a>;
}

pub type DeviceSleepFuture<'a> =
    Pin<Box<dyn Future<Output = Result<(), DeviceFlowError>> + Send + 'a>>;

pub trait DevicePollRuntime: Send + Sync {
    fn now(&self) -> Instant;
    fn sleep<'a>(
        &'a self,
        duration: Duration,
        cancellation: &'a LoginCancellation,
    ) -> DeviceSleepFuture<'a>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TokioDevicePollRuntime;

impl DevicePollRuntime for TokioDevicePollRuntime {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn sleep<'a>(
        &'a self,
        duration: Duration,
        cancellation: &'a LoginCancellation,
    ) -> DeviceSleepFuture<'a> {
        Box::pin(async move {
            tokio::select! {
                () = tokio::time::sleep(duration) => Ok(()),
                () = cancellation.cancelled() => Err(DeviceFlowError::new(DeviceFlowErrorKind::Cancelled)),
            }
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceFlowErrorKind {
    Cancelled,
    Encode,
    Transport,
    ResponseTooLarge,
    EndpointUnavailable,
    Rejected,
    Decode,
    MissingFields,
    Timeout,
}

#[derive(Clone)]
pub struct DeviceFlowError {
    pub kind: DeviceFlowErrorKind,
    pub status: Option<u16>,
    pub source: Option<std::sync::Arc<dyn Error + Send + Sync + 'static>>,
}

impl DeviceFlowError {
    #[must_use]
    pub fn new(kind: DeviceFlowErrorKind) -> Self {
        Self {
            kind,
            status: None,
            source: None,
        }
    }

    #[must_use]
    fn status(kind: DeviceFlowErrorKind, status: u16) -> Self {
        Self {
            kind,
            status: Some(status),
            source: None,
        }
    }

    #[must_use]
    fn source(kind: DeviceFlowErrorKind, source: impl Error + Send + Sync + 'static) -> Self {
        Self {
            kind,
            status: None,
            source: Some(std::sync::Arc::new(source)),
        }
    }
}

impl fmt::Debug for DeviceFlowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceFlowError")
            .field("kind", &self.kind)
            .field("status", &self.status)
            .field("has_source", &self.source.is_some())
            .finish()
    }
}

impl fmt::Display for DeviceFlowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            DeviceFlowErrorKind::Cancelled => "codex device authentication cancelled",
            DeviceFlowErrorKind::Encode => "failed to encode codex device request",
            DeviceFlowErrorKind::Transport => "codex device transport failed",
            DeviceFlowErrorKind::ResponseTooLarge => "codex device response exceeded limit",
            DeviceFlowErrorKind::EndpointUnavailable => "codex device endpoint is unavailable",
            DeviceFlowErrorKind::Rejected => "codex device endpoint rejected the request",
            DeviceFlowErrorKind::Decode => "failed to decode codex device response",
            DeviceFlowErrorKind::MissingFields => {
                "codex device flow did not return required fields"
            }
            DeviceFlowErrorKind::Timeout => {
                "codex device authentication timed out after 15 minutes"
            }
        })
    }
}

impl Error for DeviceFlowError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn Error + 'static))
    }
}

#[must_use]
pub fn should_use_codex_device_flow(options: &LoginOptions) -> bool {
    options
        .metadata
        .get(CODEX_LOGIN_MODE_METADATA_KEY)
        .is_some_and(|value| value.trim().eq_ignore_ascii_case(CODEX_LOGIN_MODE_DEVICE))
}

#[must_use]
pub fn parse_codex_device_poll_interval(raw: &Value) -> Duration {
    let seconds = raw
        .as_str()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .or_else(|| raw.as_u64())
        .filter(|seconds| *seconds > 0);
    seconds
        .map(Duration::from_secs)
        .unwrap_or(CODEX_DEVICE_DEFAULT_POLL_INTERVAL)
}

#[must_use]
pub const fn codex_device_is_success_status(status: u16) -> bool {
    status >= 200 && status < 300
}

pub async fn request_codex_device_user_code(
    transport: &dyn CodexDeviceTransport,
    cancellation: &LoginCancellation,
    client_id: &str,
) -> Result<CodexDeviceUserCodeResponse, DeviceFlowError> {
    let body = serde_json::to_vec(&CodexDeviceUserCodeRequest { client_id })
        .map_err(|error| DeviceFlowError::source(DeviceFlowErrorKind::Encode, error))?;
    let response = transport
        .post_json(CODEX_DEVICE_USER_CODE_URL, &body, cancellation)
        .await?;
    validate_response_size(&response)?;
    if !codex_device_is_success_status(response.status) {
        let kind = if response.status == 404 {
            DeviceFlowErrorKind::EndpointUnavailable
        } else {
            DeviceFlowErrorKind::Rejected
        };
        return Err(DeviceFlowError::status(kind, response.status));
    }
    let parsed: CodexDeviceUserCodeResponse = serde_json::from_slice(&response.body)
        .map_err(|error| DeviceFlowError::source(DeviceFlowErrorKind::Decode, error))?;
    if parsed.device_auth_id.trim().is_empty() || parsed.effective_user_code().is_empty() {
        return Err(DeviceFlowError::new(DeviceFlowErrorKind::MissingFields));
    }
    Ok(parsed)
}

pub async fn poll_codex_device_token(
    transport: &dyn CodexDeviceTransport,
    runtime: &dyn DevicePollRuntime,
    cancellation: &LoginCancellation,
    device_auth_id: &str,
    user_code: &str,
    interval: Duration,
) -> Result<CodexDeviceTokenResponse, DeviceFlowError> {
    let deadline = runtime.now() + CODEX_DEVICE_TIMEOUT;
    loop {
        if cancellation.is_cancelled() {
            return Err(DeviceFlowError::new(DeviceFlowErrorKind::Cancelled));
        }
        if runtime.now() > deadline {
            return Err(DeviceFlowError::new(DeviceFlowErrorKind::Timeout));
        }
        let body = serde_json::to_vec(&CodexDeviceTokenRequest {
            device_auth_id,
            user_code,
        })
        .map_err(|error| DeviceFlowError::source(DeviceFlowErrorKind::Encode, error))?;
        let response = transport
            .post_json(CODEX_DEVICE_TOKEN_URL, &body, cancellation)
            .await?;
        validate_response_size(&response)?;
        if codex_device_is_success_status(response.status) {
            let parsed: CodexDeviceTokenResponse = serde_json::from_slice(&response.body)
                .map_err(|error| DeviceFlowError::source(DeviceFlowErrorKind::Decode, error))?;
            if parsed.authorization_code.trim().is_empty()
                || parsed.code_verifier.trim().is_empty()
                || parsed.code_challenge.trim().is_empty()
            {
                return Err(DeviceFlowError::new(DeviceFlowErrorKind::MissingFields));
            }
            return Ok(parsed);
        }
        if matches!(response.status, 403 | 404) {
            runtime.sleep(interval, cancellation).await?;
            continue;
        }
        return Err(DeviceFlowError::status(
            DeviceFlowErrorKind::Rejected,
            response.status,
        ));
    }
}

fn validate_response_size(response: &DeviceHttpResponse) -> Result<(), DeviceFlowError> {
    if response.body.len() > MAX_DEVICE_RESPONSE_BYTES {
        Err(DeviceFlowError::new(DeviceFlowErrorKind::ResponseTooLarge))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use super::*;

    struct QueueTransport(Mutex<VecDeque<DeviceHttpResponse>>);

    impl CodexDeviceTransport for QueueTransport {
        fn post_json<'a>(
            &'a self,
            _url: &'a str,
            _body: &'a [u8],
            _cancellation: &'a LoginCancellation,
        ) -> DeviceHttpFuture<'a> {
            Box::pin(async move {
                self.0
                    .lock()
                    .unwrap()
                    .pop_front()
                    .ok_or_else(|| DeviceFlowError::new(DeviceFlowErrorKind::Transport))
            })
        }
    }

    struct ImmediateRuntime {
        start: Instant,
        sleeps: Mutex<usize>,
    }

    impl DevicePollRuntime for ImmediateRuntime {
        fn now(&self) -> Instant {
            self.start
        }

        fn sleep<'a>(
            &'a self,
            _duration: Duration,
            _cancellation: &'a LoginCancellation,
        ) -> DeviceSleepFuture<'a> {
            Box::pin(async move {
                *self.sleeps.lock().unwrap() += 1;
                Ok(())
            })
        }
    }

    #[test]
    fn mode_interval_and_status_match_upstream() {
        let mut options = LoginOptions::default();
        assert!(!should_use_codex_device_flow(&options));
        options.metadata.insert(
            CODEX_LOGIN_MODE_METADATA_KEY.to_owned(),
            " Device ".to_owned(),
        );
        assert!(should_use_codex_device_flow(&options));
        assert_eq!(
            parse_codex_device_poll_interval(&Value::String("7".into())),
            Duration::from_secs(7)
        );
        assert_eq!(
            parse_codex_device_poll_interval(&Value::from(3)),
            Duration::from_secs(3)
        );
        assert_eq!(
            parse_codex_device_poll_interval(&Value::from(0)),
            Duration::from_secs(5)
        );
        assert!(codex_device_is_success_status(299));
        assert!(!codex_device_is_success_status(300));
    }

    #[tokio::test]
    async fn user_code_supports_legacy_alias_and_redacts_rejection_body() {
        let fixture = br#"{"device_auth_id":"device","usercode":"CODE","interval":"2"}"#;
        serde_json::from_slice::<CodexDeviceUserCodeResponse>(fixture)
            .expect("valid upstream user-code response fixture");
        let transport = QueueTransport(Mutex::new(VecDeque::from([
            DeviceHttpResponse {
                status: 200,
                body: fixture.to_vec(),
            },
            DeviceHttpResponse {
                status: 401,
                body: b"secret-upstream-body".to_vec(),
            },
        ])));
        let cancellation = LoginCancellation::default();
        let code = request_codex_device_user_code(&transport, &cancellation, "client")
            .await
            .unwrap();
        assert_eq!(code.effective_user_code(), "CODE");
        let error = request_codex_device_user_code(&transport, &cancellation, "client")
            .await
            .unwrap_err();
        assert_eq!(error.status, Some(401));
        assert!(!format!("{error:?} {error}").contains("secret-upstream-body"));
    }

    #[tokio::test]
    async fn poll_retries_pending_then_returns_complete_exchange_tuple() {
        let transport = QueueTransport(Mutex::new(VecDeque::from([
            DeviceHttpResponse {
                status: 403,
                body: Vec::new(),
            },
            DeviceHttpResponse {
                status: 200,
                body: br#"{"authorization_code":"auth","code_verifier":"verifier","code_challenge":"challenge"}"#.to_vec(),
            },
        ])));
        let runtime = ImmediateRuntime {
            start: Instant::now(),
            sleeps: Mutex::new(0),
        };
        let response = poll_codex_device_token(
            &transport,
            &runtime,
            &LoginCancellation::default(),
            "device",
            "code",
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        assert_eq!(response.authorization_code, "auth");
        assert_eq!(*runtime.sleeps.lock().unwrap(), 1);
    }
}
