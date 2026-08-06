// Origin: CTOX module graph for the upstream Kimi auth package.
// License: AGPL-3.0-only

#[path = "kimi.rs"]
mod flow;
mod token;

pub use flow::{
    DeviceFlowClient, KimiAuth, KimiAuthError, KimiAuthErrorKind, KimiClock, KimiDeviceIdentity,
    KimiHttpFuture, KimiHttpRequest, KimiHttpResponse, KimiHttpTransport, KimiRefreshCoordinator,
    KimiSleepFuture, KimiTransportFailure, SystemKimiClock, KIMI_API_BASE_URL, KIMI_CLIENT_ID,
    KIMI_DEFAULT_POLL_INTERVAL, KIMI_DEVICE_CODE_URL, KIMI_HTTP_TIMEOUT, KIMI_MAX_POLL_DURATION,
    KIMI_OAUTH_HOST, KIMI_TOKEN_URL,
};
pub use token::{
    DeviceCodeResponse, KimiAuthBundle, KimiCredentialHandles, KimiSecretHandle, KimiSecretKind,
    KimiSecretStore, KimiSecretStoreError, KimiStoredCredentials, KimiTokenData, KimiTokenError,
    KimiTokenStorage, SecretString, KIMI_REFRESH_THRESHOLD,
};

#[cfg(test)]
mod kimi_proxy_test;
#[cfg(test)]
mod kimi_refresh_test;
