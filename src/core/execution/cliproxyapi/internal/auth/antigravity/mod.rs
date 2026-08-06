// Origin: CTOX
// License: AGPL-3.0-only

mod auth;
mod constants;
#[cfg(feature = "antigravity-http-transport")]
mod transport;

pub use auth::{
    build_auth_url, AntigravityAuth, AntigravityAuthError, AntigravityAuthErrorKind,
    AntigravityCredentialHandles, AntigravityFlowTransport, AntigravityHttpFuture,
    AntigravityHttpMethod, AntigravityHttpRequest, AntigravityHttpResponse,
    AntigravityHttpTransportFailure, AntigravityRefreshCoordinator, AntigravityRefreshError,
    AntigravityRefreshHttpResponse, AntigravityRefreshRequest, AntigravityRefreshTransport,
    AntigravityRefreshTransportFailure, AntigravitySecretHandle, AntigravitySecretKind,
    AntigravitySecretStore, AntigravityStoredCredentials, AntigravityTokenError,
    AntigravityTokenResponse, SecretString,
};
pub use constants::{
    API_ENDPOINT, API_VERSION, AUTH_ENDPOINT, CALLBACK_PORT, DAILY_API_ENDPOINT, REFRESH_SKEW,
    TOKEN_ENDPOINT, USER_INFO_ENDPOINT,
};
#[cfg(feature = "antigravity-http-transport")]
pub use transport::{AntigravityHttpTransport, AntigravityTransportBuildError};

#[cfg(test)]
mod auth_test;
